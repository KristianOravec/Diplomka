#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "curvefit.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#define CURVE_LIMIT_MAX 2000
#define DEFAULT_FIT_START 15
#define NO_HISTORICAL_DATA -1.0

typedef struct {
    const char *HW;             /* hardware id for the main data, e.g. "1070" */
    const char *file_name;      /* e.g. "gemm-reduced_output.csv" */
    uint64_t total_kernel_runs; /* how many times the final config will run */
    const char *hist_HW;        /* hardware id for the HISTORICAL data */
    double k;                 /* estimator variant selector (see evaluator.c) */
    uint64_t overhead;        /* per-step tuning overhead */
    uint64_t fit_start;       /* index to begin curve fitting from */
    uint64_t number_of_tests; /* how many Monte Carlo trials to run */
} EvaluatorParams;

/* avg_* are means over all
 * tests; std_* are their standard deviations. */
typedef struct {
    double avg_extra_runtime,
        std_extra_runtime; /* the headline metric + spread */
    double avg_crystal_ball,
        std_crystal_ball;              /* oracle stop step, mean + spread */
    double avg_estimate, std_estimate; /* estimator stop step, mean + spread */
    double avg_miss;                   /* mean |oracle - estimate| */
} EvaluatorResult;
void evaluator_run(const EvaluatorParams *params, EvaluatorResult *result);
uint64_t history_run(const char *HW, const char *file_name,
                     uint64_t total_kernel_runs, uint64_t overhead,
                     uint64_t number_of_tests);
void get_regression_params(const char *HW, const char *file_name,
                           uint64_t total_kernel_runs, uint64_t fit_start,
                           uint64_t number_of_tests, CurveParams *params);
uint64_t
recommend_tuning_length(uint64_t default_tuning_steps, const double *tuning_run,
                        uint64_t tuning_run_len, uint64_t total_kernel_runs,
                        double regression_weight, uint64_t fit_start,
                        uint64_t overhead, double hist_a, double hist_b);
#ifdef __cplusplus
}
#endif
#endif
