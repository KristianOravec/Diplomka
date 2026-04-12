#ifndef CURVEFIT_H
#define CURVEFIT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CURVEFIT_LM_ITERATIONS 100
#define CURVEFIT_LM_DAMPING 0.001
#define CURVEFIT_MATRIX_DETECTION 1e-10
#define CURVEFIT_MIN_DECAY_RATE 0.1
#define CURVEFIT_MAX_DECAY_RATE 3.0
#define CURVEFIT_MIN_SCALE_FACTOR 0.001
#define CURVEFIT_INITIAL_DECAY 0.5
#define CURVEFIT_INITIAL_SCALE 1.0
typedef struct { double a, b, c; } CurveParams;
void curve_fit(double* x, double* y, uint64_t n, uint64_t fit_start, double* a, double* b, double* c, double hist_a, double hist_b);
double curve_eval(double x, double a, double b, double c);
uint64_t minimize_total_runtime(double a, double b, double c, uint64_t current, uint64_t total, double avg_rt, uint64_t overhead);
#ifdef __cplusplus
}
#endif
#endif
