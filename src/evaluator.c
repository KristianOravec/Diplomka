/*
 * C Implementation of Evaluator and Budget Estimator
 *
 * Python equivalents:
 * - evaluator() in python/evaluator.py -> evaluator_run() in this file
 * - history() in python/evaluator.py -> history_run() in this file
 * - get_history_regression_parameters() in python/evaluator.py ->
 * get_regression_params() in this file
 * - tuning_length_recommendation() in python/budget_estimator.py ->
 * recommend_tuning_length() in this file
 * - local_budget_estimation() in python/budget_estimator.py ->
 * local_budget_estimation() in this file
 * - total_runtime_remaining() in python/budget_estimator.py ->
 * total_runtime_remaining() in this file
 *
 * KNOWN DIFFERENCES FROM PYTHON:
 * 1. DEFAULT_FIT_START: Python default is 15, C uses 10 (evaluator.h)
 * 2. Curve fitting: Python uses scipy.optimize.curve_fit with bounds,
 *    C uses GSL Levenberg-Marquardt with different constraints
 * 3. Some numerical precision differences may occur due to different libraries
 */

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
#include "random.h"

/* Large sentinel used while searching for minimum total runtime. TO-DO use
 * doubleMAX from float.h or smth */
#define LARGE_RUNTIME_SENTINEL 1e300

/* 3 distinct "salts" -> mixing salt into the per-test seed means the history
 * simulation, main evaluator and regression step all use different random
 * streams even for the same test index - to prevent accidental colleration  */
#define HISTORY_SEED_SALT 0xA5A5A5A5A5A5A5A5ULL
#define EVALUATOR_SEED_SALT 0xC3D2E1F0B4A59687ULL
#define REGRESSION_SEED_SALT 0x9E3779B97F4A7C15ULL

/* One simulated run's results */
typedef struct {
    uint64_t crystal_idx; /* oracle's best stopping step (0-based) */
    uint64_t estimate;    /* estimator's chosen stopping step */
    double extra_runtime; /* estimate_runtime / oracle_runtime - 1 */
    double miss;          /* |crystal_idx - estimate| */
} TestResult;

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

static uint64_t run_history_trial(const TuningData *data, uint64_t curve_limit,
                                  uint64_t total_kernel_runs, uint64_t overhead,
                                  uint64_t *rng_state) {
    double cumsum = 0;
    double best = 0;
    double min_runtime = LARGE_RUNTIME_SENTINEL;
    uint64_t crystal_idx = 0;

    for (uint64_t i = 0; i < curve_limit; i++) {
        double sample =
            data->data[random_index_from_state(data->size, rng_state)];
        cumsum += sample;
        if (i == 0 || sample < best)
            best = sample;

        double total = total_runtime_for_step(cumsum, i, overhead, best,
                                              total_kernel_runs);
        if (total < min_runtime) {
            min_runtime = total;
            crystal_idx = i;
        }
    }
    return crystal_idx;
}

/*
 * run_monte_carlo_test() - Main Monte Carlo test loop
 *
 * Python equivalent: main loop in evaluator() function (python/evaluator.py
 * lines 155-188)
 *
 * Performs a single iteration of:
 * 1. Sampling tuning_run from data (equivalent to lines 157 in Python)
 * 2. Computing best_so_far (lines 158-162 in Python)
 * 3. Computing tuning_costs with cumsum + overhead (lines 164-166 in Python)
 * 4. Computing running_costs (lines 167-169 in Python)
 * 5. Computing total_runtimes and finding crystal_ball minimum (lines 170-171
 * in Python)
 * 6. Calling budget_estimator to get estimate (lines 173-181 in Python)
 * 7. Computing extra_runtime (line 183 in Python)
 * 8. Recording results (lines 185-188 in Python)
 */
static TestResult
run_monte_carlo_test(const TuningData *data, uint64_t curve_limit,
                     uint64_t total_kernel_runs, uint64_t overhead, double k,
                     uint64_t default_tuning_steps, uint64_t fit_start,
                     double hist_a, double hist_b, MonteCarloWorkspace *ws,
                     uint64_t *rng_state) {
    double *tuning_run = ws->tuning_run;

    /* Line 157 in Python: tuning_run = all_config_runtimes.sample(n =
     * curve_limit).values */
    for (uint64_t i = 0; i < curve_limit; i++)
        tuning_run[i] =
            data->data[random_index_from_state(data->size, rng_state)];

    /* Lines 173-181 in Python: budget_estimator.tuning_length_recommendation()
     * call */
    uint64_t estimate = recommend_tuning_length_impl(
        default_tuning_steps, tuning_run, curve_limit, total_kernel_runs, k,
        fit_start, overhead, hist_a, hist_b, ws->best_configs);

    /* Lines 158-171 in Python: best_so_far, tuning_costs, running_costs,
     * crystal_ball */
    double cumsum = 0;
    double best = 0;
    double min_runtime = LARGE_RUNTIME_SENTINEL;
    uint64_t crystal_idx = 0;
    double estimate_runtime = 0;
    int has_estimate_runtime = (estimate < curve_limit);

    for (uint64_t i = 0; i < curve_limit; i++) {
        double sample = tuning_run[i];
        cumsum += sample;
        if (i == 0 || sample < best)
            best = sample;

        double total = total_runtime_for_step(cumsum, i, overhead, best,
                                              total_kernel_runs);

        if (total < min_runtime) {
            min_runtime = total;
            crystal_idx = i;
        }

        if (has_estimate_runtime && i == estimate) {
            estimate_runtime = total;
        }
    }

    /* Line 183 in Python: extra_runtime calculation */
    double extra_runtime = 0;
    if (has_estimate_runtime) {
        extra_runtime =
            min_runtime > 0 ? estimate_runtime / min_runtime - 1.0 : 0;
    }

    /* Lines 185-188 in Python: Recording crystal_balls, extra_runtimes,
     * estimates, misses */
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
void get_regression_params(const char *HW, const char *file_name,
                           uint64_t total_kernel_runs, uint64_t fit_start,
                           uint64_t number_of_tests, CurveParams *params) {
    /* Load historical data */
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

    for (uint64_t test = 0; test < number_of_tests; test++) {
        /* Lines 77-78 in Python: Sample tuning_run */
        for (uint64_t i = 0; i < curve_limit; i++)
            tuning_run[i] = data.data[random_index(data.size)];

        /* Lines 84-91 in Python: Running average of best_so_far */
        avg_curve[0] = (avg_curve[0] * test + tuning_run[0]) / (test + 1);
        double running_min = tuning_run[0];
        for (uint64_t j = 1; j < curve_limit; j++) {
            if (tuning_run[j] < running_min)
                running_min = tuning_run[j];
            avg_curve[j] = (avg_curve[j] * test + running_min) / (test + 1);
        }
    }

    /* Lines 93-97 in Python: curve_fit using scipy.optimize.curve_fit */
    double *x_cache = get_x_cache();
    double a = 0.5, b = 1.0, c = avg_curve[curve_limit - 1];
    curve_fit(x_cache, avg_curve, curve_limit, fit_start, &a, &b, &c,
              NO_HISTORICAL_DATA, NO_HISTORICAL_DATA);

    /* Cleanup */
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
        sum_extra_th[tid] += res.extra_runtime;
        sum_extra_sq_th[tid] += res.extra_runtime * res.extra_runtime;
        sum_crystal_th[tid] += res.crystal_idx + 1;
        sum_crystal_sq_th[tid] += (res.crystal_idx + 1) * (res.crystal_idx + 1);
        sum_estimate_th[tid] += res.estimate;
        sum_estimate_sq_th[tid] += res.estimate * res.estimate;
        sum_miss_th[tid] += res.miss;
    }

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
