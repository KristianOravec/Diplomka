#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
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
