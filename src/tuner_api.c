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
#include "evaluator.h"
#include "historical_cache.h" /* CURVE_LIMIT_MAX, NO_HISTORICAL_DATA, history_run, get_regression_params */
#include <stdlib.h>
#include <string.h>

/* Max length (including the '\0' terminator) of the optional debug name. */
#define TUNER_NAME_CAP 128

#ifndef TUNER_DEBUG
#define TUNER_DEBUG 0
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
    uint64_t total_kernel_runs;  /* X */
    uint64_t overhead;
    uint64_t fit_start;
    double k;                    /* unused (C5) */
    TunerMode mode;
    double hist_a, hist_b;       /* hist mode: frozen curve a, b */
    uint64_t hist_optimal_steps; /* hybrid mode: O_hist */

    /* state */
    double best_configs[CURVE_LIMIT_MAX]; /* best-so-far kernel time per step (fit input) */
    uint64_t best_len;
    double best_config;          /* best kernel time so far */
    double avg_runtime;          /* mean kernel time, unused (C8) */
    double avg_total_runtime;    /* mean total time = step cost */
    uint64_t step;               /* index of the last push (0-based) */
    int seeded;                  /* first push received */

    /* stop rule: counts down each step, reset on a new best, shrunk when the
     * estimate is smaller; STOP once it reaches 0 */
    uint64_t budget;
    int stopped;                 /* latched once budget < 1 */
};


/* returns the array [0,1,2,...,CURVE_LIMIT_MAX-1] used as the
 * x-values (step numbers) when fitting the curve. */
static const double *tuner_x_axis(void) {
    static double x[CURVE_LIMIT_MAX];
    static int init = 0;
    if (!init) {
        for (uint16_t i = 0; i < CURVE_LIMIT_MAX; i++)
            x[i] = (double)i;
        init = 1;
    }
    return x;
}

/* Fresh estimate: fit the curve, minimise T(x), return the extra steps
 * (0 if the curve predicts no improvement). step_cost already includes
 * the overhead. */
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

static uint64_t blend_with_history(uint64_t live_estimate, uint64_t step,
                                   uint64_t o_hist) {
    double live_weight = (double)step / (double)o_hist;
    if (live_weight > 1.0)
        live_weight = 1.0;
    double history_weight = 1.0 - live_weight;

    double history_estimate = (step < o_hist) ? (double)(o_hist - step) : 0.0;

    double blended = live_weight * (double)live_estimate +
                     history_weight * history_estimate;
    return (uint64_t)blended;
}

/* Fresh budget estimate at `step`: curve fit + T(x) minimum, blended with
 * history in hybrid mode. Used by push_result and tuner_raw_recommendation. */
static uint64_t estimate_budget(const KernelHandle *handle, uint64_t step) {
    uint64_t estimate = tuner_local_budget(
        step, handle->total_kernel_runs, handle->avg_total_runtime,
        (double *)handle->best_configs, handle->best_len, handle->fit_start,
        handle->hist_a, handle->hist_b);

    if (handle->mode == TUNER_MODE_HYBRID && handle->hist_optimal_steps > 0)
        estimate = blend_with_history(estimate, step, handle->hist_optimal_steps);

    return estimate;
}

static bool warmup_done(const KernelHandle *handle) {
    return handle->step > handle->fit_start + 5;
}

KernelHandle *initiate_kernel(const KernelConfig *cfg, const char *debug_name,
                              TunerStatus *out_status) {
    if (!cfg) {
        if (out_status)
            *out_status = TUNER_ERR_INVALID_ARG;
        return NULL;
    }
    /* calloc zero-initializes the whole struct (so all counters start at 0). */
    KernelHandle *handle = calloc(1, sizeof(*handle));
    if (!handle) {
        if (out_status)
            *out_status = TUNER_ERR_ALLOC;
        return NULL;
    }

    handle->total_kernel_runs = cfg->total_kernel_runs;
    handle->overhead = cfg->overhead;
    handle->fit_start = cfg->fit_start;
    handle->k = cfg->k;
    handle->mode = cfg->mode;
    handle->hist_a = NO_HISTORICAL_DATA;
    handle->hist_b = NO_HISTORICAL_DATA;
    handle->hist_optimal_steps = 0;

    /* debug name (optional) */
    if (debug_name) {
        size_t n = strnlen(debug_name, TUNER_NAME_CAP - 1);
        memcpy(handle->debug_name, debug_name, n);
        handle->debug_name[n] = '\0';
    } else {
        handle->debug_name[0] = '\0';
    }

    if (cfg->mode == TUNER_MODE_HISTORICAL) {
        /* k == 0: frozen curve parameters a, b. */
        double cache_a = 0, cache_b = 0;
        if (hist_cache_lookup_regression(
                cfg->hist_HW, cfg->file_name, cfg->total_kernel_runs,
                cfg->fit_start, cfg->hist_number_of_tests, &cache_a, &cache_b)) {
            handle->hist_a = cache_a;
            handle->hist_b = cache_b;
            TUNER_DBG(handle->debug_name, "hist a,b from CACHE: a=%.4f b=%.4f", cache_a, cache_b);
        } else {
            CurveParams cp;
            get_regression_params(cfg->hist_HW, cfg->file_name,
                                  cfg->total_kernel_runs, cfg->fit_start,
                                  cfg->hist_number_of_tests, &cp);
            handle->hist_a = cp.a;
            handle->hist_b = cp.b;
            TUNER_DBG(handle->debug_name,
                      "hist a,b COMPUTED (cache miss): a=%.4f b=%.4f", cp.a, cp.b);
        }
        /* cp.c / the fitted c is discarded either way: it describes the
         * HISTORICAL hardware's floor. c is refit live on this kernel's own
         * samples every step (curve_fit frozen-curve mode). */
    } else if (cfg->mode == TUNER_MODE_HYBRID) {
        /* 0 < k < 1: historical optimum O_hist, the blending backstop. */
        uint64_t o_hist = 0;
        if (hist_cache_lookup_optimum(
                cfg->hist_HW, cfg->file_name, cfg->total_kernel_runs,
                cfg->overhead, cfg->hist_number_of_tests, &o_hist)) {
            handle->hist_optimal_steps = o_hist;
            TUNER_DBG(handle->debug_name, "O_hist from CACHE: %llu",
                      (unsigned long long)o_hist);
        } else {
            handle->hist_optimal_steps =
                history_run(cfg->hist_HW, cfg->file_name,
                            cfg->total_kernel_runs, cfg->overhead,
                            cfg->hist_number_of_tests);
            TUNER_DBG(handle->debug_name, "O_hist COMPUTED (cache miss): %llu",
                      (unsigned long long)handle->hist_optimal_steps);
        }
    }
    /* LIVE mode: nothing historical to resolve. */

    /* initialize the budget countdown (same as reset_kernel). */
    handle->budget = (handle->total_kernel_runs < CURVE_LIMIT_MAX) ? handle->total_kernel_runs
                                                         : CURVE_LIMIT_MAX;
    handle->stopped = 0;

    TUNER_DBG(handle->debug_name,
              "init mode=%d k=%.2f total_runs=%llu fit_start=%llu "
              "hist_a=%.4f hist_b=%.4f O_hist=%llu start_budget=%llu",
              (int)handle->mode, handle->k, (unsigned long long)handle->total_kernel_runs,
              (unsigned long long)handle->fit_start, handle->hist_a, handle->hist_b,
              (unsigned long long)handle->hist_optimal_steps,
              (unsigned long long)handle->budget);

    if (out_status)
        *out_status = TUNER_OK;
    return handle;
}

void release_kernel(KernelHandle *handle) { free(handle); }

void reset_kernel(KernelHandle *handle) {
    if (!handle)
        return;
    handle->best_len = 0;
    handle->best_config = 0.0;
    handle->avg_runtime = 0.0;
    handle->avg_total_runtime = 0.0;
    handle->step = 0;
    handle->seeded = 0;
    /* budget starts at the max steps we could ever take (Python: max_steps),
     * = min(total_kernel_runs, CURVE_LIMIT_MAX) in the live setting. */
    handle->budget = (handle->total_kernel_runs < CURVE_LIMIT_MAX) ? handle->total_kernel_runs
                                                         : CURVE_LIMIT_MAX;
    handle->stopped = 0;
}

void set_total_runs(KernelHandle *handle, uint64_t total_kernel_runs) {
    if (handle)
        handle->total_kernel_runs = total_kernel_runs;
    /* NOTE: in HYBRID mode this does not recompute O_hist (see header note). */
}

/* ---- push ---- */

void push_result(KernelHandle *handle, double kernel_time_us, double total_time_us) {
    if (!handle)
        return;

    if (!handle->seeded) {
        // first push only stores data
        handle->best_config = kernel_time_us; // best so far = the only one we have
        handle->avg_runtime = kernel_time_us; // mean of 1 value = that value
        handle->avg_total_runtime = total_time_us; // same for total time (kernel + overhead)
        handle->best_configs[0] = kernel_time_us;  // first point of the best-so-far curve
        handle->best_len = 1; // curve has 1 point
        handle->step = 0; // this was step 0
        handle->seeded = 1; // from now on we're "running"
        TUNER_DBG(handle->debug_name,
                  "push step=0 (seed) kernel=%.1f total=%.1f budget=%llu",
                  kernel_time_us, total_time_us, (unsigned long long)handle->budget);
        return; // no budget decision on step 0s
    }

    uint64_t i = handle->step + 1;
    TUNER_DBG(handle->debug_name, "push step=%llu kernel=%.1f total=%.1f",
              (unsigned long long)i, kernel_time_us, total_time_us);
    /* i = 0-based index of this push = number of earlier samples,
     * so the new mean is (mean * i + new) / (i + 1). */
    handle->avg_runtime =
        (handle->avg_runtime * (double)i + kernel_time_us) / (double)(i + 1); // It is never read, maybe delete?
    handle->avg_total_runtime =
        (handle->avg_total_runtime * (double)i + total_time_us) / (double)(i + 1);
    handle->step = i;

    // Every push costs one step first, then the estimator may lower the budget further
    if (handle->budget > 0)
        handle->budget--;

    bool found_faster_kernel = (kernel_time_us < handle->best_config);

    if (warmup_done(handle)) {
        uint64_t new_budget = estimate_budget(handle, i);

        if (found_faster_kernel)
            handle->budget = new_budget;      /* new best -> re-commit to fresh estimate */
        else if (new_budget < handle->budget)
            handle->budget = new_budget;      /* estimate shrank -> shrink */
        /* else: keep counting down */
    }

    /* update best-so-far (KERNEL time) + history */
    if (found_faster_kernel)
        handle->best_config = kernel_time_us;
    if (handle->best_len < CURVE_LIMIT_MAX)
        handle->best_configs[handle->best_len++] = handle->best_config;

    if (handle->budget < 1) {
        if (!handle->stopped) {
            TUNER_DBG(handle->debug_name, "  STOP at step=%llu (budget exhausted)",
                      (unsigned long long)i);
        }
        handle->stopped = 1;
    }
}

uint64_t find_number_of_steps_that_should_be_tuning(KernelHandle *handle) {
    if (!handle || !handle->seeded)
        return 1; /* not started: keep tuning */
    if (handle->stopped)
        return 0;
    return handle->budget;
}

uint64_t tuner_steps_taken(const KernelHandle *handle) {
    return handle ? (handle->seeded ? handle->step + 1 : 0) : 0;
}
double tuner_best_so_far(const KernelHandle *handle) {
    return (handle && handle->seeded) ? handle->best_config : 0.0;
}
const char *tuner_debug_name(const KernelHandle *handle) {
    return handle ? handle->debug_name : "";
}

/* Fresh estimate of how many more steps are worth tuning, ignoring the
 * countdown and the stop latch (for overtune/undertune probes).
 * Returns 1 before enough data (warmup), 0 if no improvement is predicted. */
uint64_t tuner_raw_recommendation(const KernelHandle *handle) {
    if (!handle || !handle->seeded)
        return 1;
    if (!warmup_done(handle))
        return 1; /* warmup */
    return estimate_budget(handle, handle->step);
}
