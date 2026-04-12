#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <lbfgs.h>
#include "curvefit.h"
static double fitting_function(double x, double a, double b, double c) { return b / pow(x, a) + c; }
static void curve_fit_lm(double* x, double* y, uint64_t n, double* a, double* b, double* c) {
    gsl_matrix* J = gsl_matrix_alloc(n, 3);
    gsl_vector* yv = gsl_vector_alloc(n);
    gsl_vector* co = gsl_vector_alloc(3);
    gsl_vector_set(co, 0, *a); gsl_vector_set(co, 1, *b); gsl_vector_set(co, 2, *c);
    for (uint64_t iter = 0; iter < CURVEFIT_LM_ITERATIONS; iter++) {
        for (uint64_t i = 0; i < n; i++) {
            double xi = x[i], ai = gsl_vector_get(co, 0), bi = gsl_vector_get(co, 1);
            gsl_matrix_set(J, i, 0, -bi * pow(xi, -ai) * log(xi));
            gsl_matrix_set(J, i, 1, 1.0 / pow(xi, ai));
            gsl_matrix_set(J, i, 2, 1.0);
            gsl_vector_set(yv, i, y[i] - (bi / pow(xi, ai) + gsl_vector_get(co, 2)));
        }
        double lambda = CURVEFIT_LM_DAMPING;
        gsl_matrix* JtJ = gsl_matrix_alloc(3, 3);
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, J, J, 0.0, JtJ);
        for (int i = 0; i < 3; i++) gsl_matrix_set(JtJ, i, i, gsl_matrix_get(JtJ, i, i) * (1.0 + lambda) + lambda);
        gsl_vector* Jty = gsl_vector_alloc(3);
        gsl_blas_dgemv(CblasTrans, 1.0, J, yv, 0.0, Jty);
        gsl_permutation* p = gsl_permutation_alloc(3);
        int signum;
        if (gsl_linalg_LU_decomp(JtJ, p, &signum) == 0 && fabs(gsl_linalg_LU_det(JtJ, signum)) > CURVEFIT_MATRIX_DETECTION) {
            gsl_vector* delta = gsl_vector_alloc(3);
            gsl_linalg_LU_solve(JtJ, p, Jty, delta);
            gsl_vector_add(co, delta);
            double ca = gsl_vector_get(co, 0);
            if (ca < CURVEFIT_MIN_DECAY_RATE || ca > CURVEFIT_MAX_DECAY_RATE || gsl_vector_get(co, 1) < CURVEFIT_MIN_SCALE_FACTOR)
                gsl_vector_set(co, 0, ca < CURVEFIT_MIN_DECAY_RATE ? CURVEFIT_MIN_DECAY_RATE : (ca > CURVEFIT_MAX_DECAY_RATE ? CURVEFIT_MAX_DECAY_RATE : ca));
            gsl_vector_free(delta);
        }
        gsl_permutation_free(p); gsl_vector_free(Jty); gsl_matrix_free(JtJ);
    }
    *a = gsl_vector_get(co, 0); *b = gsl_vector_get(co, 1); *c = gsl_vector_get(co, 2);
    gsl_matrix_free(J); gsl_vector_free(yv); gsl_vector_free(co);
}
void curve_fit(double* x, double* y, uint64_t n, uint64_t fit_start, double* a, double* b, double* c, double hist_a, double hist_b) {
    uint64_t fit_n = n - fit_start;
    if (fit_n == 0) { *a = CURVEFIT_INITIAL_DECAY; *b = CURVEFIT_INITIAL_SCALE; *c = y[n-1]; return; }
    if (hist_a > 0 && hist_b > 0) {
        *a = hist_a; *b = hist_b;
        double sum = 0;
        for (uint64_t i = fit_start; i < n; i++) sum += y[i] - hist_b / pow(x[i], hist_a);
        *c = sum / (n - fit_start);
        return;
    }
    double a_val = CURVEFIT_INITIAL_DECAY, b_val = CURVEFIT_INITIAL_SCALE, c_val = y[n-1];
    curve_fit_lm(x + fit_start, y + fit_start, fit_n, &a_val, &b_val, &c_val);
    *a = a_val; *b = b_val; *c = c_val;
}
double curve_eval(double x, double a, double b, double c) { return fitting_function(x, a, b, c); }

typedef struct {
    double a, b, c;
    uint64_t current, total;
    double avg_rt;
    uint64_t overhead;
    double lb;  /* lower bound */
    double ub;  /* upper bound */
} MinimizationData;

/*
 * minimize_total_runtime - Find optimal budget using libLBFGS minimizer
 * 
 * Replaces O(n) exhaustive search in local_budget_estimation with O(log n) 
 * minimization using libLBFGS (similar to scipy L-BFGS-B).
 *
 * Note: libLBFGS doesn't have native bounds, so we enforce bounds in the callback.
 * Uses 'delta' parameter for gradient-based convergence (similar to scipy PGTOL).
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
    
    MinimizationData* d = (MinimizationData*)data;
    
    /* Clamp x to bounds (since libLBFGS doesn't support bounds natively) */
    double x_clamped = x[0];
    if (x_clamped < d->lb) x_clamped = d->lb;
    if (x_clamped > d->ub) x_clamped = d->ub;
    
    /* Compute objective: total_runtime_remaining */
    double result = (d->avg_rt + d->overhead) * x_clamped + 
                    curve_eval(d->current + x_clamped, d->a, d->b, d->c) * (d->total - d->current - x_clamped);
    
    /* Compute gradient numerically (central differences) */
    double eps = 1e-8;
    double x_plus = x_clamped + eps;
    if (x_plus > d->ub) x_plus = d->ub;
    double x_minus = x_clamped - eps;
    if (x_minus < d->lb) x_minus = d->lb;
    
    double f_plus = (d->avg_rt + d->overhead) * x_plus + 
                    curve_eval(d->current + x_plus, d->a, d->b, d->c) * (d->total - d->current - x_plus);
    double f_minus = (d->avg_rt + d->overhead) * x_minus + 
                     curve_eval(d->current + x_minus, d->a, d->b, d->c) * (d->total - d->current - x_minus);
    
    g[0] = (f_plus - f_minus) / (x_plus - x_minus);
    
    return result;
}

/* Progress callback (optional, not used) */
static int lbfgs_progress(void *data, const lbfgsfloatval_t *x, 
                          const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                          const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                          const lbfgsfloatval_t step, int n, int k, int ls) {
    (void)data; (void)x; (void)g; (void)fx; (void)xnorm; (void)gnorm; (void)step; (void)n; (void)k; (void)ls;
    return 0;
}

uint64_t minimize_total_runtime(double a, double b, double c, uint64_t current, 
                                 uint64_t total, double avg_rt, uint64_t overhead) {
    uint64_t max_budget = total - current;
    if (max_budget < 1) return 1;  /* Edge case: no room for budget */
    
    /* For small budgets, use exhaustive search (faster and reliable) */
    if (max_budget <= 200) {
        MinimizationData data = {a, b, c, current, total, avg_rt, overhead, 1.0, (double)max_budget};
        lbfgsfloatval_t fx = lbfgs_evaluate(&data, (lbfgsfloatval_t[]){1.0}, NULL, 1, 0.0);
        uint64_t best_budget = 1;
        lbfgsfloatval_t best_fx = fx;
        for (uint64_t bgt = 2; bgt <= max_budget; bgt++) {
            fx = lbfgs_evaluate(&data, (lbfgsfloatval_t[]){(lbfgsfloatval_t)bgt}, NULL, 1, 0.0);
            if (fx < best_fx) {
                best_fx = fx;
                best_budget = bgt;
            }
        }
        return best_budget;
    }
    
    MinimizationData data = {a, b, c, current, total, avg_rt, overhead, 1.0, (double)max_budget};
    
    /* Initialize parameters - match scipy L-BFGS-B defaults */
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    
    /* Set parameters similar to scipy:
     * - delta: convergence on gradient norm (similar to PGTOL, default 1e-5)
     * - ftol: function change tolerance (default 1e-4)
     * - max_iterations: max iterations */
    param.delta = 1e-5;       /* ≈ pgtol (1e-5) */
    param.ftol = 2.22e-9;    /* ≈ ftol (2.22e-9) */
    param.max_iterations = 100;  /* Limit iterations */
    param.max_linesearch = 20;   /* Default */
    
    /* Initial guess - start at 1 (like scipy) */
    lbfgsfloatval_t x[1];
    x[0] = 1.0;
    
    /* Run optimization */
    lbfgsfloatval_t fx;
    int ret = lbfgs(1, x, &fx, lbfgs_evaluate, lbfgs_progress, &data, &param);
    
    /* Clamp result to valid range */
    uint64_t opt_result = (uint64_t)(x[0] + 0.5);
    if (opt_result < 1) opt_result = 1;
    if (opt_result > max_budget) opt_result = max_budget;
    
    return opt_result;
}
