#ifndef EVALUATOR_H
#define EVALUATOR_H
#include <stdint.h>
#include "curvefit.h"
#ifdef __cplusplus
extern "C" {
#endif
#define CURVE_LIMIT_MAX 2000
#define DEFAULT_FIT_START 10
#define NO_HISTORICAL_DATA -1.0
typedef struct {
    const char* HW;
    const char* file_name;
    uint64_t total_kernel_runs;
    const char* hist_HW;
    double k;
    uint64_t overhead;
    uint64_t fit_start;
    uint64_t number_of_tests;
} EvaluatorParams;
typedef struct {
    double avg_extra_runtime, std_extra_runtime;
    double avg_crystal_ball, std_crystal_ball;
    double avg_estimate, std_estimate;
    double avg_miss;
} EvaluatorResult;
void evaluator_run(const EvaluatorParams* params, EvaluatorResult* result);
uint64_t history_run(const char* HW, const char* file_name, uint64_t total_kernel_runs, uint64_t overhead, uint64_t number_of_tests);
void get_regression_params(const char* HW, const char* file_name, uint64_t total_kernel_runs, uint64_t fit_start, uint64_t number_of_tests, CurveParams* params);
uint64_t recommend_tuning_length(uint64_t default_tuning_steps, const double* tuning_run, uint64_t tuning_run_len, uint64_t total_kernel_runs, double regression_weight, uint64_t fit_start, uint64_t overhead, double hist_a, double hist_b);
#ifdef __cplusplus
}
#endif
#endif
