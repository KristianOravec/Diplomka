/* =============================================================================
 * curvefit.h  -- curve fitting and total-runtime minimisation (public API)
 *
 * Provides the two mathematical primitives the estimator is built on:
 *   1. curve_fit / curve_eval -- fit and evaluate the diminishing-returns curve
 *                                f(x) = b/x^a + c.
 *   2. minimize_total_runtime* -- find the tuning budget that minimises the
 *                                projected total application runtime T(x).
 *
 * Detailed parameter documentation lives here in the header (the contract);
 * curvefit.c contains the implementation notes (Levenberg-Marquardt internals,
 * the shared minimiser core, etc.).
 * ============================================================================= */
#ifndef CURVEFIT_H
#define CURVEFIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Levenberg-Marquardt tunables (see curvefit.c for how each is used) ---- */
#define CURVEFIT_LM_ITERATIONS 100      /* fixed number of LM refinement passes */
#define CURVEFIT_LM_DAMPING 0.001       /* lambda: stabilises each LM step */
#define CURVEFIT_MATRIX_DETECTION 1e-10 /* below this |det|, treat matrix singular */
#define CURVEFIT_MIN_DECAY_RATE 0.1     /* lower clamp for decay a (from Python) */
#define CURVEFIT_MAX_DECAY_RATE 3.0     /* upper clamp for a (from Python) */
#define CURVEFIT_MIN_SCALE_FACTOR 0.001 /* floor for scale b (C-port guard) */
#define CURVEFIT_INITIAL_DECAY 0.5      /* starting guess for a */
#define CURVEFIT_INITIAL_SCALE 1.0      /* starting guess for b */

/* The three fitted parameters of f(x) = b/x^a + c. */
typedef struct {
    double a, b, c;
} CurveParams;

/* -----------------------------------------------------------------------------
 * curve_fit -- fit f(x) = b/x^a + c to the points (x[i], y[i]).
 *
 *   x         : [in]  x-values (step numbers), length n.
 *   y         : [in]  y-values (best-so-far runtimes), length n.
 *   n         : [in]  number of points.
 *   fit_start : [in]  index to begin fitting from (earlier points are noisy
 *                     warmup and are skipped).
 *   a, b, c   : [out] the fitted parameters.
 *   hist_a    : [in]  historical decay, or a negative sentinel for "none".
 *   hist_b    : [in]  historical scale, or a negative sentinel for "none".
 *
 * MODES (selected by hist_a / hist_b):
 *   both < 0  -> fit all three a, b, c live            (k=1 live).
 *   both > 0  -> freeze a, b to the historical values, fit only c
 *               (k=0 historical: the "frozen curve").
 * --------------------------------------------------------------------------- */
void curve_fit(double *x, double *y, uint64_t n, uint64_t fit_start, double *a,
               double *b, double *c, double hist_a, double hist_b);

/* -----------------------------------------------------------------------------
 * curve_eval -- evaluate the fitted curve at one point.
 *   x       : [in] the step to evaluate at (x > 0).
 *   a, b, c : [in] curve parameters.
 *   returns : the predicted best-so-far runtime at x.
 * --------------------------------------------------------------------------- */
double curve_eval(double x, double a, double b, double c);

/* -----------------------------------------------------------------------------
 * minimize_total_runtime -- BATCH variant.
 * Find the budget (extra tuning steps) that minimises projected total runtime.
 * Per-step cost is (avg_rt + overhead), with overhead passed SEPARATELY.
 *
 *   a, b, c  : [in] fitted curve parameters.
 *   current  : [in] current step index (how far we've tuned).
 *   total    : [in] total_kernel_runs (#E).
 *   avg_rt   : [in] average per-step runtime (feeds the running-cost term).
 *   overhead : [in] per-step tuning overhead (added to avg_rt).
 *   returns  : the budget (>= 1) that minimises total runtime.
 *
 * NOTE: used by the batch evaluator / Python-comparison path, which is
 * validation scaffolding and will be removed once the tuner ships. Shares its
 * minimiser core with the tuner variant below, so both stay identical.
 * --------------------------------------------------------------------------- */
uint64_t minimize_total_runtime(double a, double b, double c, uint64_t current,
                                uint64_t total, double avg_rt,
                                uint64_t overhead);

/* -----------------------------------------------------------------------------
 * minimize_total_runtime_tuner -- TUNER variant (the deployed path).
 * Same as above, but the per-step cost is passed as a single `step_cost` with no
 * separate overhead argument, because in the live API the overhead is already
 * folded into the measured total runtime the tuner supplies.
 *
 *   a, b, c   : [in] fitted curve parameters.
 *   current   : [in] current step index.
 *   total     : [in] total_kernel_runs (#E).
 *   step_cost : [in] measured per-step cost (total runtime, overhead included).
 *   returns   : the budget (>= 1) that minimises total runtime.
 * --------------------------------------------------------------------------- */
uint64_t minimize_total_runtime_tuner(double a, double b, double c,
                                      uint64_t current, uint64_t total,
                                      double step_cost);

#ifdef __cplusplus
}
#endif
#endif /* CURVEFIT_H */
