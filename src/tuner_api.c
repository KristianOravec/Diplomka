/* =============================================================================
 * tuner_api.c  -- implementation of the deployed three-function tuner API.
 *
 * Combines:
 *   - per-kernel storage (the raw results pushed so far + derived trackers),
 * and
 *   - the stateless per-call decision (fit curve to history -> minimize T(x) ->
 *     remaining budget, 0 = stop), lifted from local_budget_estimation.
 *
 * The estimator math (curve_fit, minimize_total_runtime, curve_eval) and the
 * historical-data functions (get_regression_params, history_run) are reused
 * UNCHANGED from the validated C sources.
 * =============================================================================
 */

#include "tuner_api.h"
#include "curvefit.h" /* curve_fit, minimize_total_runtime_tuner, curve_eval */
#include "evaluator.h" /* CURVE_LIMIT_MAX, NO_HISTORICAL_DATA, history_run, get_regression_params */
#include <stdlib.h>
#include <string.h>

/* Max length (including the '\0' terminator) of the optional debug name. */
#define TUNER_NAME_CAP 128

/* -----------------------------------------------------------------------------
 * DEBUG PRINTS
 *
 * Compile with -DTUNER_DEBUG=1 to turn on tracing; leave it off (the default)
 * and every TUNER_DBG(...) call vanishes at compile time -- zero runtime cost.
 *
 *   gcc ... -DTUNER_DEBUG=1 ...       # traces on
 *   gcc ...                           # traces off (production)
 *
 * Traces go to stderr (so they don't mix into a program's stdout results) and
 * are prefixed with the kernel's debug_name so multiple kernels are
 * distinguishable. They report the per-step budget decision (fresh estimate,
 * new-best resets, countdown, stop), which is what you need to see WHY the
 * estimator stopped where it did -- and, in a harness, to confirm the sampled
 * inputs are actually changing from step to step.
 * ---------------------------------------------------------------------------
 */
#ifndef TUNER_DEBUG
#define TUNER_DEBUG 0
#endif

/* -----------------------------------------------------------------------------
 * RNG SELF-CHECK  (build with -DTUNER_RNG_CHECK=1)
 *
 * Prints ONE summary line per kernel when it stops, proving the samples pushed
 * into the API actually varied. Unlike TUNER_DEBUG this does NOT trace every
 * push, so the output stays short enough to read or screenshot.
 *
 *   gcc ... -DTUNER_RNG_CHECK=1 ...    # one verdict line per kernel
 *   gcc ... -DTUNER_DEBUG=1 ...        # full per-step trace (verbose)
 * ---------------------------------------------------------------------------
 */
#ifndef TUNER_RNG_CHECK
#define TUNER_RNG_CHECK 0
#endif

/* the input-variation tracking is compiled in for EITHER flag */
#if TUNER_DEBUG || TUNER_RNG_CHECK
#define TUNER_TRACK_INPUT 1
#else
#define TUNER_TRACK_INPUT 0
#endif

#if TUNER_RNG_CHECK
#include <stdio.h>
#define TUNER_RNG_LOG(name, ...)                                               \
    do {                                                                       \
        fprintf(stderr, "[rng-check:%s] ", (name) && *(name) ? (name) : "?");  \
        fprintf(stderr, __VA_ARGS__);                                          \
        fprintf(stderr, "\n");                                                 \
    } while (0)
#else
#define TUNER_RNG_LOG(name, ...)                                               \
    do {                                                                       \
    } while (0)
#endif

#if TUNER_DEBUG
#include <stdio.h>
/* name: the kernel's debug label (or "?"); fmt/...: printf-style message. */
#define TUNER_DBG(name, ...)                                                   \
    do {                                                                       \
        fprintf(stderr, "[tuner:%s] ", (name) && *(name) ? (name) : "?");      \
        fprintf(stderr, __VA_ARGS__);                                          \
        fprintf(stderr, "\n");                                                 \
    } while (0)
#else
#define TUNER_DBG(name, ...)                                                   \
    do {                                                                       \
    } while (0)
#endif

struct KernelHandle {
    char debug_name[TUNER_NAME_CAP];

    /* config */
    uint64_t
        total_kernel_runs; /* #E: how many times the chosen config will run */
    uint64_t overhead;  /* fixed per-step overhead. Superseded for the live cost
                         * model by the measured total in push_result (two-value
                         * push); still used for the historical fit at init. */
    uint64_t fit_start; /* warmup: don't fit the curve before this step */
    double k;           /* mode selector value (1.0 / 0.0 / in-between) */
    TunerMode mode;     /* LIVE / HISTORICAL / HYBRID (enum) */

    /* historical inputs resolved at init */
    double hist_a, hist_b;       /* for HISTORICAL (k=0) */
    uint64_t hist_optimal_steps; /* O_hist for HYBRID */

    /* accumulated state */
    double best_configs[CURVE_LIMIT_MAX]; /* best-so-far KERNEL times (curve fit
                                             input) */
    uint64_t best_len;
    double best_config;       /* best KERNEL time so far */
    double avg_runtime;       /* running mean of KERNEL times (curve/legacy) */
    double avg_total_runtime; /* running mean of TOTAL times (kernel+overhead)
                                 -> cost model */
    uint64_t step;            /* index of last sample (0-based) */
    int seeded;

    /* budget countdown -- mirrors the batch estimator's running `budget`.
     * Starts at max_steps, decrements each step, is reset to the fresh estimate
     * when a new best is found (or shrunk if the estimate is smaller), and the
     * kernel signals STOP once budget < 1. This is Python's
     * commit-and-countdown stopping rule, which the earlier raw-budget poll did
     * NOT implement. */
    uint64_t budget;
    int stopped; /* latched once budget < 1 */

    /* ---- input-variation tracking (debug builds only) ----
     * Records the spread of the kernel times actually pushed, so a debug run
     * can PROVE the caller is feeding varied samples rather than a constant.
     * A stuck RNG or a mis-wired harness shows up here as min == max and
     * repeat_count == step. Zero cost when TUNER_DEBUG is off (the fields are
     * only written inside TUNER_DEBUG guards). */
    double dbg_min_kernel, dbg_max_kernel; /* range of pushed kernel times */
    double dbg_sum_kernel;                 /* for the mean */
    double dbg_last_kernel;    /* previous sample, to spot repeats */
    uint64_t dbg_repeat_count; /* consecutive identical samples */
};

/* returns the array [0,1,2,...,CURVE_LIMIT_MAX-1] used as the
 * x-values (step numbers) when fitting the curve. Built once, then reused. */
static const double *tuner_x_axis(void) {
    static double x[CURVE_LIMIT_MAX];
    static int init = 0;
    if (!init) {
        for (uint64_t i = 0; i < CURVE_LIMIT_MAX; i++)
            x[i] = (double)i;
        init = 1;
    }
    return x;
}

/* One per-step decision: fit the curve to the best-so-far history, find the
 * cost-minimising extra budget, and return it -- or 0 if the curve predicts no
 * further improvement.
 *
 *   step_cost = the measured per-step cost (total runtime, overhead already
 *               included). Passed straight to the tuner minimiser, which takes
 *               no separate overhead argument. */
static uint64_t tuner_local_budget(uint64_t current, uint64_t total,
                                   double step_cost, double *best_cfg,
                                   uint64_t best_len, uint64_t fit_start,
                                   double hist_a, double hist_b) {
    const double *x = tuner_x_axis();
    double a, b, c;
    curve_fit((double *)x, best_cfg, best_len, fit_start, &a, &b, &c, hist_a,
              hist_b);
    uint64_t best_budget =
        minimize_total_runtime_tuner(a, b, c, current, total, step_cost);
    uint64_t ib = (uint64_t)(best_budget + 0.5);
    return (curve_eval((double)current + ib, a, b, c) < best_cfg[best_len - 1])
               ? ib
               : 0;
}

KernelHandle *initiate_kernel(const KernelConfig *cfg, const char *debug_name,
                              TunerStatus *out_status) {
    /* Reject a NULL config or a struct_size smaller than expected (the
     * struct_size field is an ABI guard so old callers can be detected). */
    if (!cfg || cfg->struct_size < sizeof(KernelConfig)) {
        if (out_status)
            *out_status = TUNER_ERR_INVALID_ARG;
        return NULL;
    }
    /* calloc zero-initializes the whole struct (so all counters start at 0). */
    KernelHandle *h = calloc(1, sizeof(*h));
    if (!h) {
        if (out_status)
            *out_status = TUNER_ERR_ALLOC;
        return NULL;
    }

    h->total_kernel_runs = cfg->total_kernel_runs;
    h->overhead = cfg->overhead;
    h->fit_start = cfg->fit_start;
    h->k = cfg->k;
    h->mode = cfg->mode;
    h->hist_a = NO_HISTORICAL_DATA;
    h->hist_b = NO_HISTORICAL_DATA;
    h->hist_optimal_steps = 0;

    /* debug name (optional) */
    if (debug_name) {
        size_t n = strnlen(debug_name, TUNER_NAME_CAP - 1);
        memcpy(h->debug_name, debug_name, n);
        h->debug_name[n] = '\0';
    } else {
        h->debug_name[0] = '\0';
    }

    /* Resolve historical data for the non-LIVE modes -- done ONCE here, the
     * same way evaluator_run does it. */
    if (cfg->mode == TUNER_MODE_HISTORICAL) {
        /* k == 0: fit the frozen curve from the historical GPU's raw data --
         * a,b,c are computed here, not loaded. Keep only a,b (the shape, which
         * transfers across hardware); cp.c is the historical GPU's floor and is
         * discarded, since c is refit live on this kernel's own samples. */

        CurveParams cp;
        get_regression_params(cfg->hist_HW, cfg->file_name,
                              cfg->total_kernel_runs, cfg->fit_start,
                              cfg->hist_number_of_tests, &cp);
        h->hist_a = cp.a;
        h->hist_b = cp.b;
    } else if (cfg->mode == TUNER_MODE_HYBRID) {
        /* 0 < k < 1: compute historical optimum O_hist for the backstop. */
        h->hist_optimal_steps =
            history_run(cfg->hist_HW, cfg->file_name, cfg->total_kernel_runs,
                        cfg->overhead, cfg->hist_number_of_tests);
    }
    /* LIVE mode: nothing historical to resolve. */

    /* initialize the budget countdown (same as reset_kernel). */
    h->budget = (h->total_kernel_runs < CURVE_LIMIT_MAX) ? h->total_kernel_runs
                                                         : CURVE_LIMIT_MAX;
    h->stopped = 0;

    TUNER_DBG(h->debug_name,
              "init mode=%d k=%.2f total_runs=%llu fit_start=%llu "
              "hist_a=%.4f hist_b=%.4f O_hist=%llu start_budget=%llu",
              (int)h->mode, h->k, (unsigned long long)h->total_kernel_runs,
              (unsigned long long)h->fit_start, h->hist_a, h->hist_b,
              (unsigned long long)h->hist_optimal_steps,
              (unsigned long long)h->budget);

    if (out_status)
        *out_status = TUNER_OK;
    return h;
}

void release_kernel(KernelHandle *h) { free(h); }

void reset_kernel(KernelHandle *h) {
    if (!h)
        return;
    h->best_len = 0;
    h->best_config = 0.0;
    h->avg_runtime = 0.0;
    h->avg_total_runtime = 0.0;
    h->step = 0;
    h->seeded = 0;
    /* budget starts at the max steps we could ever take (Python: max_steps),
     * = min(total_kernel_runs, CURVE_LIMIT_MAX) in the live setting. */
    h->budget = (h->total_kernel_runs < CURVE_LIMIT_MAX) ? h->total_kernel_runs
                                                         : CURVE_LIMIT_MAX;
    h->stopped = 0;
}

void set_total_runs(KernelHandle *h, uint64_t total_kernel_runs) {
    if (h)
        h->total_kernel_runs = total_kernel_runs;
    /* NOTE: in HYBRID mode this does not recompute O_hist (see header note). */
}

/* ---- push ---- */

void push_result(KernelHandle *h, double kernel_time_us, double total_time_us) {
    if (!h)
        return;

    if (!h->seeded) {
        h->best_config = kernel_time_us; /* best-so-far uses KERNEL time */
        h->avg_runtime = kernel_time_us;
        h->avg_total_runtime = total_time_us; /* cost model uses TOTAL time */
        h->best_configs[0] = kernel_time_us;  /* curve fits to KERNEL times */
        h->best_len = 1;
        h->step = 0;
        h->seeded = 1;
#if TUNER_TRACK_INPUT
        /* start the input-variation tracking */
        h->dbg_min_kernel = h->dbg_max_kernel = kernel_time_us;
        h->dbg_sum_kernel = kernel_time_us;
        h->dbg_last_kernel = kernel_time_us;
        h->dbg_repeat_count = 0;
#endif
        TUNER_DBG(h->debug_name,
                  "push step=0 (seed) kernel=%.1f total=%.1f budget=%llu",
                  kernel_time_us, total_time_us, (unsigned long long)h->budget);
        return;
    }

    uint64_t i = h->step + 1;
#if TUNER_TRACK_INPUT
    /* accumulate the spread of pushed kernel times */
    if (kernel_time_us < h->dbg_min_kernel)
        h->dbg_min_kernel = kernel_time_us;
    if (kernel_time_us > h->dbg_max_kernel)
        h->dbg_max_kernel = kernel_time_us;
    h->dbg_sum_kernel += kernel_time_us;
    if (kernel_time_us == h->dbg_last_kernel)
        h->dbg_repeat_count++;
    h->dbg_last_kernel = kernel_time_us;
#endif
    TUNER_DBG(h->debug_name, "push step=%llu kernel=%.1f total=%.1f",
              (unsigned long long)i, kernel_time_us, total_time_us);
    /* running means: kernel time (for the curve) and total time (for cost) */
    h->avg_runtime =
        (h->avg_runtime * (double)i + kernel_time_us) / (double)(i + 1);
    h->avg_total_runtime =
        (h->avg_total_runtime * (double)i + total_time_us) / (double)(i + 1);
    h->step = i;

    /* ---- mirror the batch estimator's per-step budget logic ---- */
    if (h->budget > 0)
        h->budget--;

    int is_new_best = (kernel_time_us < h->best_config);

    if (i > h->fit_start + 5) {
        uint64_t default_tuning_steps =
            (h->mode == TUNER_MODE_HYBRID) ? h->hist_optimal_steps : 0;

        /* Per-step cost is the measured TOTAL runtime (overhead already folded
         * in), so we hand avg_total_runtime straight to the tuner minimiser --
         * no separate overhead term. This is what makes overhead MEASURED per
         * step rather than a fixed constant. */
        uint64_t new_budget = tuner_local_budget(
            i, h->total_kernel_runs, h->avg_total_runtime, h->best_configs,
            h->best_len, h->fit_start, h->hist_a, h->hist_b);

        if (default_tuning_steps > 0) {
            double rw = (double)i / (double)default_tuning_steps;
            if (rw > 1.0)
                rw = 1.0;
            double hist_remaining = (i < default_tuning_steps)
                                        ? (double)(default_tuning_steps - i)
                                        : 0.0;
            new_budget = (uint64_t)(rw * (double)new_budget +
                                    (1.0 - rw) * hist_remaining);
        }

        if (is_new_best) {
            h->budget =
                new_budget; /* new best -> re-commit to fresh estimate */
            TUNER_DBG(h->debug_name,
                      "  new_best=%.1f -> budget reset to %llu (est=%llu)",
                      kernel_time_us, (unsigned long long)h->budget,
                      (unsigned long long)new_budget);
        } else if (new_budget < h->budget) {
            h->budget = new_budget; /* estimate shrank -> shrink budget */
            TUNER_DBG(h->debug_name, "  budget shrunk to %llu (est=%llu)",
                      (unsigned long long)h->budget,
                      (unsigned long long)new_budget);
        } else {
            TUNER_DBG(
                h->debug_name, "  budget ticked to %llu (est=%llu, no reset)",
                (unsigned long long)h->budget, (unsigned long long)new_budget);
        }
    }

    /* update best-so-far (KERNEL time) + history */
    if (is_new_best)
        h->best_config = kernel_time_us;
    if (h->best_len < CURVE_LIMIT_MAX)
        h->best_configs[h->best_len++] = h->best_config;

    if (h->budget < 1) {
        if (!h->stopped) {
            TUNER_DBG(h->debug_name, "  STOP at step=%llu (budget exhausted)",
                      (unsigned long long)i);
#if TUNER_TRACK_INPUT
            /* INPUT-VARIATION VERDICT: did the caller actually feed varied
             * samples? If min == max the input was constant (stuck RNG or a
             * mis-wired harness) and any "randomness" in the result is fake. */
            {
                double spread = h->dbg_max_kernel - h->dbg_min_kernel;
                double mean = h->dbg_sum_kernel / (double)(i + 1);
                const char *verdict = (spread > 0.0)
                                          ? "VARIED (randomness real)"
                                          : "CONSTANT INPUT -- RNG NOT WORKING";
                /* under TUNER_DEBUG this joins the trace; under TUNER_RNG_CHECK
                 * it is the ONLY line printed, one per kernel. */
                TUNER_DBG(h->debug_name,
                          "  INPUT CHECK: n=%llu min=%.1f max=%.1f mean=%.1f "
                          "spread=%.1f consecutive-repeats=%llu -> %s",
                          (unsigned long long)(i + 1), h->dbg_min_kernel,
                          h->dbg_max_kernel, mean, spread,
                          (unsigned long long)h->dbg_repeat_count, verdict);
                TUNER_RNG_LOG(h->debug_name,
                              "stop=%llu samples n=%llu min=%.0f max=%.0f "
                              "mean=%.0f spread=%.0f repeats=%llu -> %s",
                              (unsigned long long)i,
                              (unsigned long long)(i + 1), h->dbg_min_kernel,
                              h->dbg_max_kernel, mean, spread,
                              (unsigned long long)h->dbg_repeat_count, verdict);
            }
#endif
        }
        h->stopped = 1;
    }
}

/* ---- query (stateless per-call) ---- */

uint64_t find_number_of_steps_that_should_be_tuning(KernelHandle *h) {
    if (!h || !h->seeded)
        return 1; /* not started: keep tuning */
    /* The stop decision is now maintained incrementally in push_result, using
     * Python's commit-and-countdown rule. Here we simply report it:
     *   0  = budget exhausted -> STOP
     *   >0 = remaining committed budget -> keep tuning */
    if (h->stopped)
        return 0;
    return h->budget;
}

/* ---- introspection ---- */
uint64_t tuner_steps_taken(const KernelHandle *h) {
    return h ? (h->seeded ? h->step + 1 : 0) : 0;
}
double tuner_best_so_far(const KernelHandle *h) {
    return (h && h->seeded) ? h->best_config : 0.0;
}
const char *tuner_debug_name(const KernelHandle *h) {
    return h ? h->debug_name : "";
}

/* -----------------------------------------------------------------------------
 * tuner_raw_recommendation -- the FRESH per-step budget given the current
 * history, IGNORING the latched countdown/stop state. This answers "given
 * everything pushed so far, how many more steps does the curve say are worth it
 * RIGHT NOW?" -- which is what an overtune/undertune probe needs (the latched
 * countdown would just report 0 once stopped). Returns 0 if the curve predicts
 * no further improvement, or during warmup returns a positive "keep going". */
uint64_t tuner_raw_recommendation(const KernelHandle *h) {
    if (!h || !h->seeded)
        return 1;
    uint64_t i = h->step;
    if (i <= h->fit_start + 5)
        return 1; /* warmup: not enough data to decide */

    uint64_t default_tuning_steps =
        (h->mode == TUNER_MODE_HYBRID) ? h->hist_optimal_steps : 0;

    uint64_t nb =
        tuner_local_budget(i, h->total_kernel_runs, h->avg_total_runtime,
                           (double *)h->best_configs, h->best_len, h->fit_start,
                           h->hist_a, h->hist_b);

    if (default_tuning_steps > 0) {
        double rw = (double)i / (double)default_tuning_steps;
        if (rw > 1.0)
            rw = 1.0;
        double hist_remaining = (i < default_tuning_steps)
                                    ? (double)(default_tuning_steps - i)
                                    : 0.0;
        nb = (uint64_t)(rw * (double)nb + (1.0 - rw) * hist_remaining);
    }
    return nb;
}
