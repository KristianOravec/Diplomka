/* =============================================================================
 * evaluator.c  -- implementation of the batch Monte Carlo evaluator and helpers.
 *
 * Implements evaluator_run (average metrics over many trials), history_run
 * (historical optimum O_hist), get_regression_params (historical curve fit), and
 * recommend_tuning_length (the batch stopping rule the deployed API mirrors).
 * Public parameter docs live in evaluator.h; the comments here explain the
 * internals (sampling, the Monte Carlo loop, the cost model).
 * ============================================================================= */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "csv.h"
#include "curvefit.h"
#include "evaluator.h"

/* Large sentinel used while searching for minimum total runtime. TO-DO use
 * doubleMAX from float.h or smth */
#define LARGE_RUNTIME_SENTINEL 1e300

/* 3 distinct "salts" -> mixing salt into the per-test seed means the history
 * simulation, main evaluator and regression step all use different random
 * streams even for the same test index - to prevent accidental colleration  */
#define HISTORY_SEED_SALT 0xA5A5A5A5A5A5A5A5ULL
#define EVALUATOR_SEED_SALT 0xC3D2E1F0B4A59687ULL
#define REGRESSION_SEED_SALT 0x9E3779B97F4A7C15ULL

/* -----------------------------------------------------------------------------
 * RNG SELF-CHECK  (build with -DEVALUATOR_RNG_CHECK=1)
 *
 * Prints a short, screenshot-friendly report proving the Monte Carlo sampling is
 * genuinely random: that every trial gets its own seed, that no two trials share
 * one, and that the resulting stop steps actually vary. Off by default and
 * costs nothing when off.
 *
 *   gcc ... -DEVALUATOR_RNG_CHECK=1 ...
 *
 * Output goes to stderr so it never mixes into a harness's results on stdout.
 * --------------------------------------------------------------------------- */
#ifndef EVALUATOR_RNG_CHECK
#define EVALUATOR_RNG_CHECK 0
#endif

#if EVALUATOR_RNG_CHECK
#define EVAL_RNG_LOG(...)                                                      \
    do {                                                                       \
        fprintf(stderr, __VA_ARGS__);                                          \
        fprintf(stderr, "\n");                                                 \
    } while (0)
#else
#define EVAL_RNG_LOG(...)                                                      \
    do {                                                                       \
    } while (0)
#endif

/* One simulated run's results */
typedef struct {
    uint64_t crystal_idx; /* oracle's best stopping step (0-based) */
    uint64_t estimate;    /* estimator's chosen stopping step */
    double extra_runtime; /* estimate_runtime / oracle_runtime - 1 */
    double miss;          /* |crystal_idx - estimate| */
} TestResult;

/* Per-thread scratch buffers allocated once and reused across many tests so we
   don't malloc/free inside the hot loop. tuning_run holds one synthetic run;
   best_configs holds the best-so-far curve the estimator fits to.
*/
typedef struct {
    double *tuning_run;
    double *best_configs;
} MonteCarloWorkspace;

static uint64_t recommend_tuning_length_impl(
    uint64_t default_tuning_steps, const double *tuning_run,
    uint64_t tuning_run_len, uint64_t total_kernel_runs,
    double regression_weight, uint64_t fit_start, uint64_t overhead,
    double hist_a, double hist_b, double *best_configs);

static inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t make_test_seed(uint64_t test_idx, uint64_t salt) {
    uint64_t state = salt ^ (test_idx + 0x9E3779B97F4A7C15ULL);
    return splitmix64_next(&state);
}

static inline uint64_t random_index_from_state(uint64_t n, uint64_t *state) {
    return splitmix64_next(state) % n;
}

static inline int evaluator_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static inline int evaluator_thread_id(void) {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/* How many steps a simulated run should have, smallest of: total_kernel_runs,
 * the data size and hard cap (CURVE_LIMIT_MAX) */
static inline uint64_t get_curve_limit(uint64_t total_kernel_runs,
                                       uint64_t data_size) {
    uint64_t max_steps =
        (total_kernel_runs < data_size) ? total_kernel_runs : data_size;
    return (max_steps < CURVE_LIMIT_MAX) ? max_steps : CURVE_LIMIT_MAX;
}

static inline double total_runtime_for_step(double cumulative_tuning_runtime,
                                            uint64_t step_idx,
                                            uint64_t overhead,
                                            double best_runtime_so_far,
                                            uint64_t total_kernel_runs) {
    double tuning_cost = cumulative_tuning_runtime + (step_idx + 1) * overhead;
    double running_cost = best_runtime_so_far * (total_kernel_runs - step_idx);
    return tuning_cost + running_cost;
}

static void clear_result(EvaluatorResult *result) {
    /* zero out every field, used on any error path so caller never see garbage
     */
    result->avg_extra_runtime = 0;
    result->std_extra_runtime = 0;
    result->avg_crystal_ball = 0;
    result->std_crystal_ball = 0;
    result->avg_estimate = 0;
    result->std_estimate = 0;
    result->avg_miss = 0;
}

static void build_data_path(const char *hw, const char *file_name, char *path,
                            size_t path_len) {
    char benchmark[256];
    const char *safe_hw = hw ? hw : "";
    const char *safe_file_name = file_name ? file_name : "";

    snprintf(benchmark, sizeof(benchmark), "%s", safe_file_name);
    /* Find the "_output.csv" suffix... */
    char *suffix = strstr(benchmark, "_output.csv");
    /* ...and chop it off by writing a '\0' there, leaving just the benchmark
     * name. */
    if (suffix)
        *suffix = '\0';

    /* Put the full path into the caller's buffer */
    snprintf(path, path_len, "raw-data/raw-autotuning-data/%s/%s-%s", benchmark,
             safe_hw, safe_file_name);
}
/*
  get_x_cache -- returns a shared array [0, 1, 2, ..., CURVE_LIMIT_MAX-1] of
  doubles, used as the x-axis (step numbers) when fitting curves. Built once on
  first call and reused forever (the `static` locals persist across calls).
*/
static double *get_x_cache(void) {
    static double x_cache[CURVE_LIMIT_MAX];
    static int initialized = 0;

    if (!initialized) {
        for (uint64_t i = 0; i < CURVE_LIMIT_MAX; i++) {
            x_cache[i] = i;
        }
        initialized = 1;
    }

    return x_cache;
}

/*
    workspace_init -- allocate the 2 scratch buffers for one thread's workspace.
*/
static int workspace_init(MonteCarloWorkspace *ws, uint64_t curve_limit) {
    ws->tuning_run = NULL;
    ws->best_configs = NULL;

    ws->tuning_run = malloc(curve_limit * sizeof(double));
    ws->best_configs = malloc(CURVE_LIMIT_MAX * sizeof(double));
    return ws->tuning_run && ws->best_configs;
}

static void workspace_free(MonteCarloWorkspace *ws) {
    free(ws->tuning_run);
    free(ws->best_configs);
    ws->tuning_run = NULL;
    ws->best_configs = NULL;
}

/*
    run_history_trial -- ONE trial of the "history" simulation

    Draws curve_limit samples, tracks the best-so-far and cumulative cost, and
    returns the step index where the COST MODEL is minimised (the oracle stop).
    This is just the oracle half of a Monte Carlo test, with no estimator
   involved.
*/
static uint64_t run_history_trial(const TuningData *data, uint64_t curve_limit,
                                  uint64_t total_kernel_runs, uint64_t overhead,
                                  uint64_t *rng_state) {
    double cumsum = 0;
    double best = 0; /* best (min) runtime so far */
    double min_runtime = LARGE_RUNTIME_SENTINEL;
    uint64_t crystal_idx = 0;

    for (uint64_t i = 0; i < curve_limit; i++) {
        /* Draw a random runtime from the data (bootstrap sampling).
         *
         * SAMPLING NOTE (applies to every draw site in this file):
         * This samples WITH replacement -- the same configuration can be drawn
         * more than once in a run. The Python reference used pandas .sample(),
         * which defaults to replace=False and therefore never repeats a
         * configuration. Drawing 2000 from a pool of 5788 yields ~1690 distinct
         * values here versus 2000 in Python.
         *
         * Measured effect on the average best-so-far curve: under 1% at every
         * step (680 data, 300 tests), with the sign varying -- i.e. within
         * noise. Kept as-is deliberately: sampling with replacement is standard
         * bootstrap resampling, and switching would change every validated
         * number in the project for a sub-1% difference.
         *
         * Do not "fix" this without re-running the full validation suite. */
        double sample =
            data->data[random_index_from_state(data->size, rng_state)];
        cumsum += sample;
        /* Update best-so-far: on the first step, or whenever we beat the
         * record. */
        if (i == 0 || sample < best)
            best = sample;

        /* Evaluate the cost model at this step. */
        double total = total_runtime_for_step(cumsum, i, overhead, best,
                                              total_kernel_runs);
        /* Track the step with the lowest cost = the oracle's best stopping
         * point. */
        if (total < min_runtime) {
            min_runtime = total;
            crystal_idx = i;
        }
    }
    return crystal_idx;
}

/*
    run_monte_carlo_test() - Main Monte Carlo test loop
    run_monte_carlo_test -- ONE full Monte Carlo test: build a synthetic run,
    ask the estimator where to stop, compute the oracle stop, and score the gap.
 */
static TestResult
run_monte_carlo_test(const TuningData *data, uint64_t curve_limit,
                     uint64_t total_kernel_runs, uint64_t overhead, double k,
                     uint64_t default_tuning_steps, uint64_t fit_start,
                     double hist_a, double hist_b, MonteCarloWorkspace *ws,
                     uint64_t *rng_state) {
    /* Use this thread's reusable scratch buffer for the synthetic run. */
    double *tuning_run = ws->tuning_run;

    /* Build the syntethic run: curve_limit random draws from the real data.
     * With replacement -- see the SAMPLING NOTE above run_oracle_scan(). */
    for (uint64_t i = 0; i < curve_limit; i++)
        tuning_run[i] =
            data->data[random_index_from_state(data->size, rng_state)];

    /* Ask the estimator where it WOULD stop on this run. (Done first because
     * the estimator needs the full tuning_run array, which we just filled.) */
    uint64_t estimate = recommend_tuning_length_impl(
        default_tuning_steps, tuning_run, curve_limit, total_kernel_runs, k,
        fit_start, overhead, hist_a, hist_b, ws->best_configs);

    /* Now compute the ORACLE stop by scanning the same run. */
    double cumsum = 0;
    double best = 0;
    double min_runtime = LARGE_RUNTIME_SENTINEL;
    uint64_t crystal_idx = 0;
    double estimate_runtime = 0;
    /* Only meaningful if the estimator's stop index is within the run. */
    int has_estimate_runtime = (estimate < curve_limit);

    for (uint64_t i = 0; i < curve_limit; i++) {
        double sample = tuning_run[i];
        cumsum += sample;
        if (i == 0 || sample < best)
            best = sample;

        /* Cost model at step i. */
        double total = total_runtime_for_step(cumsum, i, overhead, best,
                                              total_kernel_runs);
        /* Track the oracle's minimum-cost step. */
        if (total < min_runtime) {
            min_runtime = total;
            crystal_idx = i;
        }

        /* When we reach the estimator's chosen step, record its cost so we can
         * compare it to the oracle's minimum. */
        if (has_estimate_runtime && i == estimate) {
            estimate_runtime = total;
        }
    }

    /* extra_runtime = how much worse the estimator's stop is than optimal,
     * as a fraction. 0 means the estimator matched the oracle's cost. Guard
     * against divide-by-zero with the min_runtime > 0 check. */
    double extra_runtime = 0;
    if (has_estimate_runtime) {
        extra_runtime =
            min_runtime > 0 ? estimate_runtime / min_runtime - 1.0 : 0;
    }

    /* Package up the four numbers for this test. miss = |oracle - estimate|. */
    TestResult res = {crystal_idx, estimate, extra_runtime,
                      fabs((double)crystal_idx - (double)estimate)};

    return res;
}

/*
 * local_budget_estimation() - Estimate optimal budget at current tuning step
 *
 * Python equivalent: budget_estimator.local_budget_estimation()
 * (python/budget_estimator.py lines 67-103)
 *
 * 1. Fits curve to best_configs_so_far (lines 72-93 in Python)
 * 2. Finds optimal budget using minimize (line 96 in Python)
 * 3. Returns budget if curve_eval < best_configs[-1], else 0 (lines 100-103 in
 * Python)
 *
 * DIFFERENCES FROM PYTHON:
 * - Python uses scipy.optimize.curve_fit with bounds ([0.1,-inf,-inf],
 * [3,inf,inf]) and scipy.optimize.minimize with bounds
 * - C uses GSL Levenberg-Marquardt implementation (curvefit.c) with different
 *   constraints (CURVEFIT_MIN_DECAY_RATE=0.1, CURVEFIT_MAX_DECAY_RATE=3.0)
 * - Python uses exhaustive search for budget (line 96: minimize), C also uses
 *   exhaustive search (line 110-111)
 * - May produce slightly different curve fit results due to different
 * algorithms
 */
static uint64_t local_budget_estimation(uint64_t current, uint64_t total,
                                        double avg_rt, double *best_cfg,
                                        uint64_t best_len, uint64_t fit_start,
                                        uint64_t overhead, double hist_a,
                                        double hist_b) {
    /* Lines 72-93 in Python: Xvalues, Yvalues setup and curve_fit call */
    double *x_cache = get_x_cache();
    double a, b, c;
    curve_fit(x_cache, best_cfg, best_len, fit_start, &a, &b, &c, hist_a,
              hist_b);

    /* Lines 96-97 in Python: optimize.minimize to find best budget */
    uint64_t best_budget =
        minimize_total_runtime(a, b, c, current, total, avg_rt, overhead);

    /* Lines 100-103 in Python: Return budget if curve_eval < best_cfg[-1], else
     * 0
     */
    uint64_t ib = (uint64_t)(best_budget + 0.5);
    return (curve_eval(current + ib, a, b, c) < best_cfg[best_len - 1]) ? ib
                                                                        : 0;
}

static uint64_t recommend_tuning_length_impl(
    uint64_t default_tuning_steps, const double *tuning_run,
    uint64_t tuning_run_len, uint64_t total_kernel_runs,
    double regression_weight, uint64_t fit_start, uint64_t overhead,
    double hist_a, double hist_b, double *best_configs) {
    /* Lines 28-34 in Python: Initialization */
    uint64_t max_steps = (total_kernel_runs < tuning_run_len)
                             ? total_kernel_runs
                             : tuning_run_len;
    uint64_t budget = max_steps;
    double best_config = tuning_run[0];
    uint64_t best_len = 1;
    double avg_runtime = tuning_run[0];

    best_configs[0] = tuning_run[0];

    /* Line 37 in Python: loop_limit = min(max_tuning_steps, 2000) */
    uint64_t limit =
        (max_steps < CURVE_LIMIT_MAX) ? max_steps : CURVE_LIMIT_MAX;

    /* Main loop: lines 38-61 in Python */
    for (uint64_t i = 1; i < limit; i++) {
        /* Line 41 in Python: average_runtime_so_far = (average_runtime_so_far *
         * i + tuning_run[i]) / (i+1) */
        budget--;
        avg_runtime = (avg_runtime * i + tuning_run[i]) / (i + 1);

        /* Lines 44-55 in Python: local_budget_estimation call and budget update
         */
        if (i > fit_start + 5) {
            uint64_t new_budget = local_budget_estimation(
                i, total_kernel_runs, avg_runtime, best_configs, best_len,
                fit_start, overhead, hist_a, hist_b);

            /* Lines 47-50 in Python: regression_weight blending */
            if (default_tuning_steps > 0) {
                double rw = (double)i / default_tuning_steps;
                if (rw > 1.0)
                    rw = 1.0;
                new_budget = (uint64_t)(rw * new_budget +
                                        (1 - rw) * (default_tuning_steps - i));
            }

            /* Lines 52-55 in Python: Update budget */
            if (tuning_run[i] < best_config) {
                budget = new_budget;
            } else if (new_budget < budget) {
                budget = new_budget;
            }
        }

        /* Lines 57-58 in Python: Update best_config and best_configs_so_far */
        if (tuning_run[i] < best_config)
            best_config = tuning_run[i];
        if (best_len < CURVE_LIMIT_MAX)
            best_configs[best_len++] = best_config;

        /* Lines 60-61 in Python: Early return if budget < 1 */
        if (budget < 1)
            return i;
    }

    /* Lines 63-65 in Python: Return value if loop completes */
    uint64_t res = (max_steps - 1 < 3000) ? max_steps - 1 : 3000;
    return res > 0 ? res : 1;
}

uint64_t
recommend_tuning_length(uint64_t default_tuning_steps, const double *tuning_run,
                        uint64_t tuning_run_len, uint64_t total_kernel_runs,
                        double regression_weight, uint64_t fit_start,
                        uint64_t overhead, double hist_a, double hist_b) {
    double *best_configs = malloc(CURVE_LIMIT_MAX * sizeof(double));
    if (!best_configs)
        return 1;

    uint64_t result = recommend_tuning_length_impl(
        default_tuning_steps, tuning_run, tuning_run_len, total_kernel_runs,
        regression_weight, fit_start, overhead, hist_a, hist_b, best_configs);
    free(best_configs);
    return result;
}

/*
 * history_run() - Monte Carlo simulation to find historical stopping point
 *
 * Python equivalent: evaluator.history() (python/evaluator.py lines 17-50)
 *
 * Runs multiple tests (number_of_tests) and averages the optimal stopping
 * point:
 * 1. Samples tuning_run from historical data (lines 33-34 in Python)
 * 2. Computes best_so_far (lines 35-39 in Python)
 * 3. Computes total_runtimes (lines 41-46 in Python)
 * 4. Finds optimal (line 48 in Python)
 * 5. Returns average + 1 (line 50 in Python)
 */
uint64_t history_run(const char *HW, const char *file_name,
                     uint64_t total_kernel_runs, uint64_t overhead,
                     uint64_t number_of_tests) {
    /* Load historical data */
    char path[512];
    build_data_path(HW, file_name, path, sizeof(path));
    TuningData data = csv_load(path, "Computation duration (us)");

    if (!data.data || data.size == 0) {
        csv_free(&data);
        return DEFAULT_FIT_START;
    }
    uint64_t curve_limit = get_curve_limit(total_kernel_runs, data.size);
    if (curve_limit == 0) {
        csv_free(&data);
        return DEFAULT_FIT_START;
    }

    uint64_t historical_optimum = 0;
    if (number_of_tests == 0) {
        csv_free(&data);
        return DEFAULT_FIT_START;
    }

    int max_threads = evaluator_max_threads();
    uint64_t *thread_sum = calloc((size_t)max_threads, sizeof(uint64_t));
    if (!thread_sum) {
        csv_free(&data);
        return DEFAULT_FIT_START;
    }

    /* Parallelise the trials across threads, but only if there are enough (>32)
     * to make threading worth the overhead. Each trial gets a reproducible
     * seed. */
#pragma omp parallel for if (number_of_tests > 32)
    for (uint64_t test = 0; test < number_of_tests; test++) {
        int tid = evaluator_thread_id();
        uint64_t rng_state = make_test_seed(test, HISTORY_SEED_SALT);
        thread_sum[tid] += run_history_trial(
            &data, curve_limit, total_kernel_runs, overhead, &rng_state);
    }

    for (int t = 0; t < max_threads; t++) {
        historical_optimum += thread_sum[t];
    }
    free(thread_sum);

    csv_free(&data);
    /* Line 50 in Python: return round(historical_optimum / number_of_tests) + 1
     */
    return (uint64_t)llround((double)historical_optimum /
                             (double)number_of_tests) +
           1;
}

/*
 * get_regression_params() - Fit regression parameters from historical data
 *
 * Python equivalent: evaluator.get_history_regression_parameters()
 * (python/evaluator.py lines 60-101)
 *
 * 1. Loads historical data (lines 61-68 in Python)
 * 2. Runs multiple tests computing average best_so_far curve (lines 74-91 in
 * Python)
 * 3. Fits curve to average (lines 93-97 in Python)
 * 4. Returns a, b parameters (lines 99-101 in Python)
 */
/* =============================================================================
 * get_regression_params -- learn the SHAPE of the convergence curve on one
 * piece of hardware, so it can be frozen and reused on another (k=0 mode).
 *
 * THE IDEA
 *   A single simulated tuning run produces a jagged, luck-dependent best-so-far
 *   curve. Averaging many runs point-by-point gives one smooth curve that is
 *   representative of this hardware. We fit f(x) = b/x^a + c to THAT average.
 *   The caller keeps a and b -- the shape -- and transplants them onto a
 *   different GPU, refitting only the floor c against live data. That is the
 *   "frozen curve" of the historical mode.
 *
 * WHY AVERAGE FIRST, THEN FIT (rather than fit each run and average the a,b,c)
 *   Fitting noisy single-run data yields unstable parameters, and the occasional
 *   wild fit would skew the mean. Averaging the DATA first produces a smooth,
 *   well-behaved curve that fits cleanly -- one robust fit beats a thousand
 *   shaky ones.
 *
 *   HW              : [in]  historical hardware id, e.g. "1070".
 *   file_name       : [in]  data file, e.g. "gemm-reduced_output.csv".
 *   total_kernel_runs:[in]  #E, only used to cap the curve length.
 *   fit_start       : [in]  skip this many noisy warmup points before fitting.
 *   number_of_tests : [in]  how many runs to average (typically 1000).
 *   params          : [out] receives the fitted a, b, c. On ANY failure this is
 *                           filled with neutral defaults (a=0.5, b=1, c=0) and
 *                           the function returns early -- callers get usable
 *                           values rather than garbage.
 *
 * Mirrors Python's get_history_regression_parameters(); the "Lines NN" comments
 * below refer to that function.
 * ============================================================================= */
void get_regression_params(const char *HW, const char *file_name,
                           uint64_t total_kernel_runs, uint64_t fit_start,
                           uint64_t number_of_tests, CurveParams *params) {
    /* ---- load the historical hardware's measured runtimes ---- */
    char path[512];
    build_data_path(HW, file_name, path, sizeof(path));
    TuningData data = csv_load(path, "Computation duration (us)");

    /* If any problem, return default params (a=0.5,b=1,c=0)*/
    if (!data.data || data.size == 0) {
        csv_free(&data);
        params->a = 0.5;
        params->b = 1.0;
        params->c = 0;
        return;
    }
    uint64_t curve_limit = get_curve_limit(total_kernel_runs, data.size);
    if (curve_limit == 0) {
        csv_free(&data);
        params->a = 0.5;
        params->b = 1.0;
        params->c = 0;
        return;
    }

    if (number_of_tests == 0) {
        csv_free(&data);
        params->a = 0.5;
        params->b = 1.0;
        params->c = 0;
        return;
    }

    double *avg_curve = calloc(curve_limit, sizeof(double));
    double *tuning_run = malloc(curve_limit * sizeof(double));
    if (!avg_curve || !tuning_run) {
        free(avg_curve);
        free(tuning_run);
        csv_free(&data);
        params->a = 0.5;
        params->b = 1.0;
        params->c = 0;
        return;
    }

    /* ---- build the AVERAGE convergence curve over `number_of_tests` runs ----
     * avg_curve[j] ends up holding the mean best-so-far runtime at step j,
     * averaged across every simulated run. */
    for (uint64_t test = 0; test < number_of_tests; test++) {
        /* Lines 77-78 in Python: draw one synthetic tuning run.
         * With replacement -- see the SAMPLING NOTE earlier in this file.
         *
         * Per-test SALTED seed, same scheme as history_run and the main
         * evaluator loop. Previously this drew from the shared global GSL RNG
         * (random.c), whose state advances across calls -- so identical inputs
         * returned different parameters depending on how many prior calls the
         * process had made (pure execution-order dependence). Seeding each
         * trial from (test, REGRESSION_SEED_SALT) makes the result a pure
         * function of the inputs: reproducible, order-independent, and
         * thread-safe by construction. */
        uint64_t rng_state = make_test_seed(test, REGRESSION_SEED_SALT);
        for (uint64_t i = 0; i < curve_limit; i++)
            tuning_run[i] =
                data.data[random_index_from_state(data.size, &rng_state)];

        /* Lines 84-91 in Python: fold this run's best-so-far curve into the
         * running average. The incremental mean
         *     new_avg = (old_avg * test + value) / (test + 1)
         * lets us average all runs without storing them: `test` is the count of
         * runs already folded in, `test + 1` the count including this one.
         * (avg_curve was calloc'd to zero, so on test 0 this reduces to
         * new_avg = value -- which is why no special first-iteration branch is
         * needed here, unlike the Python.) */
        avg_curve[0] = (avg_curve[0] * test + tuning_run[0]) / (test + 1);
        double running_min = tuning_run[0]; /* best seen so far in THIS run */
        for (uint64_t j = 1; j < curve_limit; j++) {
            if (tuning_run[j] < running_min)
                running_min = tuning_run[j]; /* the curve only ever descends */
            avg_curve[j] = (avg_curve[j] * test + running_min) / (test + 1);
        }
    }

    /* ---- fit f(x) = b/x^a + c to the averaged curve ----
     * Lines 93-97 in Python (scipy.optimize.curve_fit).
     *
     * WHY NO_HISTORICAL_DATA HERE, EVEN THOUGH THIS IS THE HISTORICAL FIT:
     * the two "historical" names mean different things.
     *   TUNER_MODE_HISTORICAL = what the tuner does LATER (reuse a frozen curve)
     *   NO_HISTORICAL_DATA    = "this particular call has no a,b to freeze"
     * This function is what CREATES the frozen curve, so it must fit from
     * scratch -- there is nothing to inherit yet. Both sentinels therefore
     * select curve_fit's "fit everything" mode: a, b AND c all come out of the
     * Levenberg-Marquardt fit.
     *
     * The a,b produced here are stored as hist_a/hist_b by initiate_kernel and
     * passed back into curve_fit on every live tuning step -- that later call
     * DOES supply them, selecting the frozen-curve mode. Same function, opposite
     * arguments: once to MAKE the curve, once to USE it.
     *
     * NOTE: a, b, c are pure OUTPUT parameters -- curve_fit overwrites them in
     * every mode and never reads the values passed in. The initialisers below
     * are therefore redundant; they merely duplicate the starting guesses
     * curve_fit already uses internally (CURVEFIT_INITIAL_DECAY,
     * CURVEFIT_INITIAL_SCALE, and the last point of the curve for c). Kept only
     * so the declaration reads as a complete statement of intent. */
    double *x_cache = get_x_cache();
    double a = 0.5, b = 1.0, c = avg_curve[curve_limit - 1];
    curve_fit(x_cache, avg_curve, curve_limit, fit_start, &a, &b, &c,
              NO_HISTORICAL_DATA, NO_HISTORICAL_DATA);

    /* Hand back all three, though callers typically use only a and b (the
     * shape); c is refit against live data on the target hardware. */
    params->a = a;
    params->b = b;
    params->c = c;
    free(tuning_run);
    free(avg_curve);
    csv_free(&data);
}

void evaluator_run(const EvaluatorParams *params, EvaluatorResult *result) {
    char path[512];
    build_data_path(params->HW, params->file_name, path, sizeof(path));
    TuningData data = csv_load(path, "Computation duration (us)");

    if (!data.data || data.size == 0) {
        clear_result(result);
        return;
    }

    /* Determine how long each simulated run is; bail if it's zero or no tests.
     */
    uint64_t curve_limit =
        get_curve_limit(params->total_kernel_runs, data.size);
    if (curve_limit == 0 || params->number_of_tests == 0) {
        clear_result(result);
        csv_free(&data);
        return;
    }

    double hist_a = NO_HISTORICAL_DATA, hist_b = NO_HISTORICAL_DATA;
    if (params->k == 0.0) {
        CurveParams cp;
        get_regression_params(params->hist_HW, params->file_name,
                              params->total_kernel_runs, params->fit_start,
                              params->number_of_tests, &cp);
        hist_a = cp.a;
        hist_b = cp.b;
    }

    /* Lines 146-147 in Python: k > 0.0 && k < 1.0 case - get
     * historical_tuning_steps */
    uint64_t default_tuning_steps = 0;
    if (params->k > 0.0 && params->k < 1.0) {
        default_tuning_steps = history_run(
            params->hist_HW, params->file_name, params->total_kernel_runs,
            params->overhead, params->number_of_tests);
    }

    /* Allocate one workspace per thread. calloc so pointers start NULL. */
    int max_threads = evaluator_max_threads();
    MonteCarloWorkspace *workspaces =
        calloc((size_t)max_threads, sizeof(MonteCarloWorkspace));
    if (!workspaces) {
        csv_free(&data);
        clear_result(result);
        return;
    }

    /* Init each thread's workspace */
    int ws_ok = 1;
    for (int t = 0; t < max_threads; t++) {
        if (!workspace_init(&workspaces[t], curve_limit)) {
            ws_ok = 0;
            break;
        }
    }
    if (!ws_ok) {
        for (int t = 0; t < max_threads; t++)
            workspace_free(&workspaces[t]);
        free(workspaces);
        csv_free(&data);
        clear_result(result);
        return;
    }

    /* Seven per-thread accumulators: running sums and sums-of-squares for the
     * four metrics (extra, crystal, estimate get a sum-of-squares for stddev;
     * miss only needs a mean). Sums-of-squares let us compute variance later
     * via var = E[x^2] - (E[x])^2. */
    double *sum_extra_th = calloc((size_t)max_threads, sizeof(double));
    double *sum_crystal_th = calloc((size_t)max_threads, sizeof(double));
    double *sum_estimate_th = calloc((size_t)max_threads, sizeof(double));
    double *sum_miss_th = calloc((size_t)max_threads, sizeof(double));
    double *sum_extra_sq_th = calloc((size_t)max_threads, sizeof(double));
    double *sum_crystal_sq_th = calloc((size_t)max_threads, sizeof(double));
    double *sum_estimate_sq_th = calloc((size_t)max_threads, sizeof(double));

    if (!sum_extra_th || !sum_crystal_th || !sum_estimate_th || !sum_miss_th ||
        !sum_extra_sq_th || !sum_crystal_sq_th || !sum_estimate_sq_th) {
        free(sum_extra_th);
        free(sum_crystal_th);
        free(sum_estimate_th);
        free(sum_miss_th);
        free(sum_extra_sq_th);
        free(sum_crystal_sq_th);
        free(sum_estimate_sq_th);
        for (int t = 0; t < max_threads; t++)
            workspace_free(&workspaces[t]);
        free(workspaces);
        csv_free(&data);
        clear_result(result);
        return;
    }
#if EVALUATOR_RNG_CHECK
    /* per-trial seed + stop step, summarised after the loop */
    uint64_t *rngchk_seed = malloc(params->number_of_tests * sizeof(uint64_t));
    double *rngchk_est = malloc(params->number_of_tests * sizeof(double));
#endif

    /* THE MAIN PARALLEL LOOP: run every Monte Carlo test. Each thread writes
     * only to its own bucket (indexed by tid), so there are no data races. */
#pragma omp parallel for if (params->number_of_tests > 32)
    for (uint64_t test = 0; test < params->number_of_tests; test++) {
        int tid = evaluator_thread_id();
        /* Reproducible per-test seed, salted for the evaluator stream. */
        uint64_t rng_state = make_test_seed(test, EVALUATOR_SEED_SALT);
        /* Run one full test (synthetic run + estimator + oracle + scoring). */
        TestResult res = run_monte_carlo_test(
            &data, curve_limit, params->total_kernel_runs, params->overhead,
            params->k, default_tuning_steps, params->fit_start, hist_a, hist_b,
            &workspaces[tid], &rng_state);
#if EVALUATOR_RNG_CHECK
        rngchk_seed[test] = make_test_seed(test, EVALUATOR_SEED_SALT);
        rngchk_est[test] = (double)res.estimate;
#endif
        sum_extra_th[tid] += res.extra_runtime;
        sum_extra_sq_th[tid] += res.extra_runtime * res.extra_runtime;
        sum_crystal_th[tid] += res.crystal_idx + 1;
        sum_crystal_sq_th[tid] += (res.crystal_idx + 1) * (res.crystal_idx + 1);
        sum_estimate_th[tid] += res.estimate;
        sum_estimate_sq_th[tid] += res.estimate * res.estimate;
        sum_miss_th[tid] += res.miss;
    }

#if EVALUATOR_RNG_CHECK
    {
        uint64_t n = params->number_of_tests;

        /* --- CHECK 1: are all per-trial seeds distinct? ---
         * A collision means two trials replayed the IDENTICAL sampled run, so
         * the effective sample count is lower than `n` and the averages are
         * quietly over-confident. O(n^2), so keep trial counts small here. */
        uint64_t dup = 0;
        for (uint64_t i = 0; i < n; i++)
            for (uint64_t j = i + 1; j < n; j++)
                if (rngchk_seed[i] == rngchk_seed[j]) { dup++; break; }

        /* --- CHECK 2: did the sampling actually change the OUTCOME? ---
         * Spread of the stop steps. sd == 0 would mean every trial behaved
         * identically, i.e. the sampling had no effect at all. */
        double mn = rngchk_est[0], mx = rngchk_est[0], sum = 0, sum2 = 0;
        for (uint64_t i = 0; i < n; i++) {
            if (rngchk_est[i] < mn) mn = rngchk_est[i];
            if (rngchk_est[i] > mx) mx = rngchk_est[i];
            sum += rngchk_est[i];
            sum2 += rngchk_est[i] * rngchk_est[i];
        }
        double mean = sum / (double)n;
        double var = sum2 / (double)n - mean * mean;
        double sd = (var > 0) ? sqrt(var) : 0.0;

        /* --- CHECK 3: are the DRAWS THEMSELVES uniform over the pool? ---
         * Checks 1-2 can both pass with a badly skewed generator (e.g. one that
         * only ever returns indices from the first tenth of the data). Here we
         * replay one trial's draw sequence and bucket the indices into deciles.
         * A uniform generator puts ~10% in each; a skewed one clusters.
         * `max_dev` is the largest deviation from the expected 10%. */
        uint64_t buckets[10] = {0};
        uint64_t probe_draws = curve_limit;
        uint64_t probe_state = make_test_seed(0, EVALUATOR_SEED_SALT);
        for (uint64_t i = 0; i < probe_draws; i++) {
            uint64_t idx = random_index_from_state(data.size, &probe_state);
            buckets[(idx * 10) / data.size]++;
        }
        double expect = (double)probe_draws / 10.0;
        double max_dev = 0.0;
        for (int bi = 0; bi < 10; bi++) {
            double dev = fabs((double)buckets[bi] - expect) / expect;
            if (dev > max_dev) max_dev = dev;
        }

        /* --- VERDICT ---
         * All three must hold:
         *   no seed collisions,
         *   the stop steps genuinely spread (sd > 0.5 step, not merely max>min),
         *   the draws are within 25% of uniform in every decile.
         * The 25% band is deliberately loose: with only a few hundred draws,
         * decile counts fluctuate by tens of percent purely by chance. */
        int ok_seeds = (dup == 0);
        int ok_spread = (sd > 0.5);
        int ok_uniform = (max_dev < 0.25);

        EVAL_RNG_LOG("%s", "");
        EVAL_RNG_LOG("+-- RNG SELF-CHECK (evaluator) ----------------------------+");
        EVAL_RNG_LOG("|  config : HW=%s runs=%llu k=%.1f oh=%llu",
                     params->HW, (unsigned long long)params->total_kernel_runs,
                     params->k, (unsigned long long)params->overhead);
        EVAL_RNG_LOG("|  trials : %llu", (unsigned long long)n);
        EVAL_RNG_LOG("|");
        EVAL_RNG_LOG("|  [1] seed uniqueness  (two trials must never share a seed)");
        EVAL_RNG_LOG("|      first 3 seeds  : %llu, %llu, %llu",
                     (unsigned long long)(rngchk_seed[0] % 1000000),
                     (unsigned long long)(rngchk_seed[n > 1 ? 1 : 0] % 1000000),
                     (unsigned long long)(rngchk_seed[n > 2 ? 2 : 0] % 1000000));
        EVAL_RNG_LOG("|      duplicates     : %llu    -> %s",
                     (unsigned long long)dup, ok_seeds ? "PASS" : "FAIL");
        EVAL_RNG_LOG("|");
        EVAL_RNG_LOG("|  [2] outcome spread  (sampling must change the result)");
        EVAL_RNG_LOG("|      stop steps     : min=%.0f max=%.0f mean=%.1f sd=%.1f",
                     mn, mx, mean, sd);
        EVAL_RNG_LOG("|                     -> %s", ok_spread ? "PASS" : "FAIL");
        EVAL_RNG_LOG("|");
        EVAL_RNG_LOG("|  [3] draw uniformity (pool must be sampled evenly)");
        EVAL_RNG_LOG("|      %llu draws by decile:", (unsigned long long)probe_draws);
        EVAL_RNG_LOG("|      %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu  (expect ~%.0f each)",
                     (unsigned long long)buckets[0], (unsigned long long)buckets[1],
                     (unsigned long long)buckets[2], (unsigned long long)buckets[3],
                     (unsigned long long)buckets[4], (unsigned long long)buckets[5],
                     (unsigned long long)buckets[6], (unsigned long long)buckets[7],
                     (unsigned long long)buckets[8], (unsigned long long)buckets[9],
                     expect);
        EVAL_RNG_LOG("|      worst deviation: %.1f%%  -> %s",
                     100.0 * max_dev, ok_uniform ? "PASS" : "FAIL");
        EVAL_RNG_LOG("|");
        EVAL_RNG_LOG("|  VERDICT: %s",
                     (ok_seeds && ok_spread && ok_uniform)
                         ? "RANDOMNESS ACTIVE AND WELL-BEHAVED"
                         : "PROBLEM DETECTED -- see the FAIL line(s) above");
        EVAL_RNG_LOG("+----------------------------------------------------------+");
        EVAL_RNG_LOG("%s", "");
        free(rngchk_seed);
        free(rngchk_est);
    }
#endif

    /* REDUCTION: combine all threads' partial sums into grand totals. */
    double sum_extra = 0, sum_crystal = 0, sum_estimate = 0, sum_miss = 0,
           sum_extra_sq = 0, sum_crystal_sq = 0, sum_estimate_sq = 0;
    for (int t = 0; t < max_threads; t++) {
        sum_extra += sum_extra_th[t];
        sum_extra_sq += sum_extra_sq_th[t];
        sum_crystal += sum_crystal_th[t];
        sum_crystal_sq += sum_crystal_sq_th[t];
        sum_estimate += sum_estimate_th[t];
        sum_estimate_sq += sum_estimate_sq_th[t];
        sum_miss += sum_miss_th[t];
    }

    free(sum_extra_th);
    free(sum_crystal_th);
    free(sum_estimate_th);
    free(sum_miss_th);
    free(sum_extra_sq_th);
    free(sum_crystal_sq_th);
    free(sum_estimate_sq_th);

    /* Lines 190-197 in Python: Compute averages and standard deviations */
    uint64_t n = params->number_of_tests;

    result->avg_extra_runtime = sum_extra / n;
    result->avg_crystal_ball = sum_crystal / n;
    result->avg_estimate = sum_estimate / n;
    result->avg_miss = sum_miss / n;
    result->std_extra_runtime =
        sqrt(sum_extra_sq / n -
             result->avg_extra_runtime * result->avg_extra_runtime);
    result->std_crystal_ball =
        sqrt(sum_crystal_sq / n -
             result->avg_crystal_ball * result->avg_crystal_ball);
    result->std_estimate =
        sqrt(sum_estimate_sq / n - result->avg_estimate * result->avg_estimate);

    /* Final cleanup */
    for (int t = 0; t < max_threads; t++)
        workspace_free(&workspaces[t]);
    free(workspaces);
    csv_free(&data);
}
