/* =============================================================================
 * curvefit.c  -- curve fitting (Levenberg-Marquardt) and total-runtime
 *                minimisation (exhaustive scan + libLBFGS).
 *
 * Implements the primitives declared in curvefit.h. Public parameter docs live
 * in the header; the comments here explain the internals: the LM iteration, the
 * three fit modes, and the shared minimiser core behind the batch/tuner wrappers.
 * ============================================================================= */

#include "curvefit.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <lbfgs.h>
#include <math.h>

/*
    pred(x) = 1/x^a · b + c (Section 2.1 / 3.1)

    [in] a : decay rate
    [in] b : initial scale
    [in] c : floor

    returns: the predicted best-so-far runtime at step x.
    */
static double fitting_function(double x, double a, double b, double c) {
    return b / pow(x, a) + c;
}

/* Trust-region fit helpers. GSL has no bounds, so the decay a is fitted
 * through t with a = MIN + (MAX - MIN) * sigmoid(t): a can never leave
 * [CURVEFIT_MIN_DECAY_RATE, CURVEFIT_MAX_DECAY_RATE], like Python's bounds. */
typedef struct {
    const double *x, *y;
    size_t n;
} FitData;

static inline double a_from_t(double t) {
    return CURVEFIT_MIN_DECAY_RATE +
           (CURVEFIT_MAX_DECAY_RATE - CURVEFIT_MIN_DECAY_RATE) / (1.0 + exp(-t));
}

static inline double t_from_a(double a) {
    double lo = CURVEFIT_MIN_DECAY_RATE, hi = CURVEFIT_MAX_DECAY_RATE;
    double eps = 1e-9 * (hi - lo);
    if (a < lo + eps) a = lo + eps;
    if (a > hi - eps) a = hi - eps;
    return log((a - lo) / (hi - a));
}

/* residuals r_i = f(x_i) - y_i */
static int fit_residuals(const gsl_vector *p, void *data, gsl_vector *r) {
    const FitData *d = (const FitData *)data;
    double a = a_from_t(gsl_vector_get(p, 0));
    double b = gsl_vector_get(p, 1), c = gsl_vector_get(p, 2);
    for (size_t i = 0; i < d->n; i++)
        gsl_vector_set(r, i, fitting_function(d->x[i], a, b, c) - d->y[i]);
    return GSL_SUCCESS;
}

/* Jacobian w.r.t. (t, b, c): d/da = -b * x^-a * ln x, chained with da/dt */
static int fit_jacobian(const gsl_vector *p, void *data, gsl_matrix *J) {
    const FitData *d = (const FitData *)data;
    double t = gsl_vector_get(p, 0), b = gsl_vector_get(p, 1);
    double a = a_from_t(t);
    double s = 1.0 / (1.0 + exp(-t));
    double da_dt = (CURVEFIT_MAX_DECAY_RATE - CURVEFIT_MIN_DECAY_RATE) * s * (1.0 - s);
    for (size_t i = 0; i < d->n; i++) {
        double inv_x_pow_a = 1.0 / pow(d->x[i], a);
        gsl_matrix_set(J, i, 0, -b * inv_x_pow_a * log(d->x[i]) * da_dt);
        gsl_matrix_set(J, i, 1, inv_x_pow_a);
        gsl_matrix_set(J, i, 2, 1.0);
    }
    return GSL_SUCCESS;
}

/* Starting point: for a fixed a the model is linear in b and c, so scan 30
 * values of a, solve b, c by ordinary least squares, keep the best triple.
 * Starting from b = 1 instead can strand the solver in a poor valley. */
static void initial_guess(const double *x, const double *y, size_t n, double *a,
                          double *b, double *c) {
    double best = INFINITY;
    for (int g = 0; g < 30; g++) {
        double ag = CURVEFIT_MIN_DECAY_RATE +
                    (CURVEFIT_MAX_DECAY_RATE - CURVEFIT_MIN_DECAY_RATE) * (g + 0.5) / 30;
        double su = 0, sy = 0, suu = 0, suy = 0;
        for (size_t i = 0; i < n; i++) {
            double u = pow(x[i], -ag);
            su += u; sy += y[i]; suu += u * u; suy += u * y[i];
        }
        double mu = su / n, my = sy / n, var = suu / n - mu * mu;
        if (!(var > 0))
            continue;
        double bg = (suy / n - mu * my) / var, cg = my - bg * mu, sse = 0;
        for (size_t i = 0; i < n; i++) {
            double r = fitting_function(x[i], ag, bg, cg) - y[i];
            sse += r * r;
        }
        if (sse < best) { best = sse; *a = ag; *b = bg; *c = cg; }
    }
}

/*
    Fits a,b,c so f(x[i]) ~ y[i] for all i, by trust-region least squares
    (GSL gsl_multifit_nlinear, Levenberg-Marquardt). Like scipy's curve_fit,
    a step is only accepted if it lowers the error, and the fit stops on
    convergence.
    [in]     x : array of x-values (step numbers), length n
    [in]     y : array of y-values (measured best-so-far runtimes), length n.
    [in]     n : number of points.
    [in/out] a : decay -- pass a starting guess in; fitted value out
    [in/out] b : scale -- starting guess in; fitted value out
    [in/out] c : floor -- starting guess in; fitted value out

    returns    : nothing (a, b, c)
*/
static void curve_fit_lm(double *x, double *y, uint64_t n, double *a, double *b,
                         double *c) {
    if (n < 3)
        return;
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();

    FitData data = {x, y, (size_t)n};
    gsl_multifit_nlinear_fdf fdf = {.f = fit_residuals, .df = fit_jacobian,
                                    .fvv = NULL, .n = (size_t)n, .p = 3,
                                    .params = &data};
    gsl_multifit_nlinear_parameters params =
        gsl_multifit_nlinear_default_parameters();
    gsl_multifit_nlinear_workspace *w = gsl_multifit_nlinear_alloc(
        gsl_multifit_nlinear_trust, &params, (size_t)n, 3);
    gsl_vector *co = gsl_vector_alloc(3);

    initial_guess(x, y, (size_t)n, a, b, c);
    gsl_vector_set(co, 0, t_from_a(*a));
    gsl_vector_set(co, 1, *b);
    gsl_vector_set(co, 2, *c);
    gsl_multifit_nlinear_init(co, &fdf, w);

    int info;
    gsl_multifit_nlinear_driver(CURVEFIT_LM_ITERATIONS * 5, 1e-8, 1e-8, 1e-8,
                                NULL, NULL, &info, w);

    const gsl_vector *p = gsl_multifit_nlinear_position(w);
    double na = a_from_t(gsl_vector_get(p, 0));
    double nb = gsl_vector_get(p, 1), nc = gsl_vector_get(p, 2);
    if (isfinite(na) && isfinite(nb) && isfinite(nc)) {
        *a = na;
        *b = nb;
        *c = nc;
    }

    gsl_vector_free(co);
    gsl_multifit_nlinear_free(w);
    gsl_set_error_handler(old_handler);
}

/*
    [in] x         : x-values (step numbers), length n.
    [in] y         : y-values (best-so-far runtimes), length n
    [in] n         : number of points
    [in] fit_start : index to begin fitting from
    [out] a,b,c     : the fitted curve parameters.
    [in]  hist_a    : historical decay, or NO_HISTORICAL_DATA (-1) for "none".
    [in]  hist_b    : historical scale, or -1 for "none".

    MODES:
      hist_a<0 & hist_b<0 -> fit all three a,b,c live      (k=1 live).
      hist_a>0 & hist_b>0 -> freeze a,b to the historical values, fit only c
                             (k=0 historical: the "frozen curve").
    returns: nothing (results via a,b,c).
*/
void curve_fit(double *x, double *y, uint64_t n, uint64_t fit_start, double *a,
               double *b, double *c, double hist_a, double hist_b) {
    /* How many points we'll actually fit. */
    uint64_t fit_n = n - fit_start;

    /* MODE 1 -- nothing to fit (fit_start == n). Return sensible defaults and
     * use the last observed value as the floor c. */
    if (n <= fit_start) {
        *a = CURVEFIT_INITIAL_DECAY;
        *b = CURVEFIT_INITIAL_SCALE;
        *c = y[n - 1];
        return;
    }

    /* MODE 2 -- historical parameters provided: reuse a and b, only solve for
     * c. c is chosen as the average vertical offset between the data and the
     * fixed historical curve b/x^a over the fitted range. */
    if (hist_a > 0 && hist_b > 0) {
        *a = hist_a;
        *b = hist_b;

        double sum = 0;
        for (uint64_t i = fit_start; i < n; i++)
            sum += y[i] - hist_b / pow(x[i], hist_a);
        *c = sum / (n - fit_start);
        return;
    }

    /* MODE 3 -- full fit from scratch using Levenberg-Marquardt.
     * Start from default guesses; c starts at the last observed value. */
    double a_val = CURVEFIT_INITIAL_DECAY, b_val = CURVEFIT_INITIAL_SCALE,
           c_val = y[n - 1];
    curve_fit_lm(x + fit_start, y + fit_start, fit_n, &a_val, &b_val, &c_val);

    *a = a_val;
    *b = b_val;
    *c = c_val;
}

double curve_eval(double x, double a, double b, double c) {
    return fitting_function(x, a, b, c);
}

typedef struct {
    double a, b, c;          /* the fitted curve parameters */
    uint64_t current, total; /* current tuning step, and total kernel runs */
    double avg_rt;           /* average runtime per step observed so far */
    uint64_t overhead;       /* per-step tuning overhead */
    double lb;               /* lower bound on the budget x */
    double ub;               /* upper bound on the budget x */
} MinimizationData;

/* T(x) = x·Cᵢ + (#E − x)·pred(x) (Section 2.2/2.3)
    Cost if you stop at step X
    Cᵢ: d->avg_rt + d->overhead
    #E: d->total
*/
static inline double total_runtime_eval(double x, const MinimizationData *d) {
    return (d->avg_rt + d->overhead) * x +
           curve_eval(d->current + x, d->a, d->b, d->c) *
               (d->total - d->current - x);
}

/*
 * minimize_total_runtime - Find optimal budget using libLBFGS minimizer
 *
 * Replaces O(n) exhaustive search in local_budget_estimation with O(log n)
 * minimization using libLBFGS (similar to scipy L-BFGS-B).
 *
 * Note: libLBFGS doesn't have native bounds, so we enforce bounds in the
 * callback. Uses 'delta' parameter for gradient-based convergence (similar to
 * scipy PGTOL).
 *
 * Parameters:
 *   a, b, c   - curve fit parameters from fitting_function: b/x^a + c
 *   current   - current tuning step
 *   total     - total kernel runs
 *   avg_rt    - average runtime so far
 *   overhead  - tuning overhead per step
 *
 * Returns optimal budget (x) that minimizes total_runtime_remaining()
 */

/* libLBFGS callback function - computes objective and gradient */
static lbfgsfloatval_t lbfgs_evaluate(void *data, const lbfgsfloatval_t *x,
                                      lbfgsfloatval_t *g, const int n,
                                      const lbfgsfloatval_t step) {
    (void)n;
    (void)step;

    MinimizationData *d = (MinimizationData *)data;

    /* Clamp x to bounds (since libLBFGS doesn't support bounds natively) */
    double x_clamped = x[0];
    if (x_clamped < d->lb)
        x_clamped = d->lb;
    if (x_clamped > d->ub)
        x_clamped = d->ub;

    /* Compute objective: total_runtime_remaining */
    double result = total_runtime_eval(x_clamped, d);

    /* Compute gradient numerically (central differences) */
    double eps = 1e-8;
    double x_plus = x_clamped + eps;
    if (x_plus > d->ub)
        x_plus = d->ub;
    double x_minus = x_clamped - eps;
    if (x_minus < d->lb)
        x_minus = d->lb;

    if (g) {
        double f_plus = total_runtime_eval(x_plus, d);
        double f_minus = total_runtime_eval(x_minus, d);
        g[0] = (f_plus - f_minus) / (x_plus - x_minus);
    }

    return result;
}

/* Progress callback (optional, not used) */
static int lbfgs_progress(void *data, const lbfgsfloatval_t *x,
                          const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                          const lbfgsfloatval_t xnorm,
                          const lbfgsfloatval_t gnorm,
                          const lbfgsfloatval_t step, int n, int k, int ls) {
    (void)data;
    (void)x;
    (void)g;
    (void)fx;
    (void)xnorm;
    (void)gnorm;
    (void)step;
    (void)n;
    (void)k;
    (void)ls;
    return 0;
}

/* =============================================================================
 * minimise_core -- the actual minimiser, shared by both public variants.
 *
 * Finds the budget x (number of extra tuning steps) that minimises
 * total_runtime_eval, returned as a whole number >= 1.
 *
 * The per-step cost is passed as a single combined `step_cost`; the objective
 * (total_runtime_eval) then uses (avg_rt + overhead) with overhead == 0, so
 * `step_cost` IS the per-step cost. The two public wrappers below differ only in
 * how they build this value:
 *   - batch  : step_cost = avg_rt + overhead   (overhead is a separate constant)
 *   - tuner  : step_cost = measured total       (overhead already folded in)
 * Keeping one core guarantees the two paths stay numerically identical.
 *
 * Strategy: exhaustive scan for small ranges (<=200: fast and exact), libLBFGS
 * for larger ranges (O(log n) rather than O(n)).
 * ---------------------------------------------------------------------------
 */
static uint64_t minimise_core(double a, double b, double c, uint64_t current,
                              uint64_t total, double step_cost) {
    uint64_t max_budget = total - current;
    if (max_budget < 1)
        return 1; /* no room to tune further */

    /* Bundle the objective's inputs. overhead is 0 because step_cost already
     * carries the full per-step cost. */
    MinimizationData data = {a,         b,   c,   current, total,
                             step_cost, 0,   1.0, (double)max_budget};

    /* --- small range: exhaustive search (exact, and cheap here) --- */
    if (max_budget <= 200) {
        lbfgsfloatval_t best_fx = total_runtime_eval(1.0, &data);
        uint64_t best_budget = 1;
        for (uint64_t bgt = 2; bgt <= max_budget; bgt++) {
            lbfgsfloatval_t fx = total_runtime_eval((double)bgt, &data);
            if (fx < best_fx) {
                best_fx = fx;
                best_budget = bgt;
            }
        }
        return best_budget;
    }

    /* --- large range: libLBFGS, tuned to resemble scipy's L-BFGS-B --- */
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.delta = 1e-5;         /* gradient-norm convergence (~ scipy pgtol) */
    param.ftol = 2.22e-9;       /* function-change tolerance (~ scipy ftol) */
    param.max_iterations = 100; /* iteration cap */
    param.max_linesearch = 20;  /* line-search cap (default) */

    lbfgsfloatval_t x[1] = {1.0}; /* initial guess, start at 1 (like scipy) */
    lbfgsfloatval_t fx;
    int ret = lbfgs(1, x, &fx, lbfgs_evaluate, lbfgs_progress, &data, &param);
    (void)ret;

    /* round + clamp into [1, max_budget] */
    uint64_t opt_result = (uint64_t)(x[0] + 0.5);
    if (opt_result < 1)
        opt_result = 1;
    if (opt_result > max_budget)
        opt_result = max_budget;
    return opt_result;
}

/* BATCH variant: overhead supplied separately; combine it with avg_rt. Used by
 * the batch evaluator / Python-comparison path (temporary validation scaffold). */
uint64_t minimize_total_runtime(double a, double b, double c, uint64_t current,
                                uint64_t total, double avg_rt,
                                uint64_t overhead) {
    return minimise_core(a, b, c, current, total, avg_rt + (double)overhead);
}

/* TUNER variant: step_cost is the measured total runtime, which already includes
 * overhead -- so pass it straight through. Used by the deployed push/query path. */
uint64_t minimize_total_runtime_tuner(double a, double b, double c,
                                      uint64_t current, uint64_t total,
                                      double step_cost) {
    return minimise_core(a, b, c, current, total, step_cost);
}
