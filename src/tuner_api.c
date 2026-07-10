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
#include "curvefit.h" /* curve_fit, minimize_total_runtime, curve_eval, CurveParams */
#include "evaluator.h" /* CURVE_LIMIT_MAX, NO_HISTORICAL_DATA, history_run, get_regression_params */
#include <stdlib.h>
#include <string.h>

/* Max length (including the '\0' terminator) of the optional debug name. */
#define TUNER_NAME_CAP 128

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

static uint64_t tuner_local_budget(uint64_t current, uint64_t total,
                                   double avg_rt, double *best_cfg,
                                   uint64_t best_len, uint64_t fit_start,
                                   uint64_t overhead, double hist_a,
                                   double hist_b) {
    const double *x = tuner_x_axis();
    double a, b, c;
    curve_fit((double *)x, best_cfg, best_len, fit_start, &a, &b, &c, hist_a,
              hist_b);
    uint64_t best_budget =
        minimize_total_runtime(a, b, c, current, total, avg_rt, overhead);
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
        /* k == 0: fit historical curve params a, b. */
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
        return;
    }

    uint64_t i = h->step + 1;
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

        /* COST MODEL CHANGE: the measured total runtime already includes
         * overhead, so we pass avg_total_runtime as the per-step cost and
         * overhead = 0 (minimize_total_runtime uses (avg_rt + overhead) as the
         * step cost). This makes overhead MEASURED per step rather than a fixed
         * constant. */
        uint64_t new_budget = tuner_local_budget(
            i, h->total_kernel_runs, h->avg_total_runtime, h->best_configs,
            h->best_len, h->fit_start, /*overhead=*/0, h->hist_a, h->hist_b);

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
            h->budget = new_budget;
        } else if (new_budget < h->budget) {
            h->budget = new_budget;
        }
    }

    /* update best-so-far (KERNEL time) + history */
    if (is_new_best)
        h->best_config = kernel_time_us;
    if (h->best_len < CURVE_LIMIT_MAX)
        h->best_configs[h->best_len++] = h->best_config;

    if (h->budget < 1)
        h->stopped = 1;
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
                           /*overhead=*/0, h->hist_a, h->hist_b);

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
