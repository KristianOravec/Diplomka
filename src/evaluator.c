/*
 * C Implementation of Evaluator and Budget Estimator
 * 
 * Python equivalents:
 * - evaluator() in python/evaluator.py -> evaluator_run() in this file
 * - history() in python/evaluator.py -> history_run() in this file
 * - get_history_regression_parameters() in python/evaluator.py -> get_regression_params() in this file
 * - tuning_length_recommendation() in python/budget_estimator.py -> recommend_tuning_length() in this file
 * - local_budget_estimation() in python/budget_estimator.py -> local_budget_estimation() in this file
 * - total_runtime_remaining() in python/budget_estimator.py -> total_runtime_remaining() in this file
 * 
 * KNOWN DIFFERENCES FROM PYTHON:
 * 1. DEFAULT_FIT_START: Python default is 15, C uses 10 (evaluator.h)
 * 2. Curve fitting: Python uses scipy.optimize.curve_fit with bounds,
 *    C uses GSL Levenberg-Marquardt with different constraints
 * 3. Some numerical precision differences may occur due to different libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "evaluator.h"
#include "csv.h"
#include "random.h"
#include "curvefit.h"

/* Structure to hold test results - corresponds to collecting results in evaluator.py lines 138-148 */
typedef struct { uint64_t crystal_idx, estimate; double extra_runtime, miss; } TestResult;

/*
 * run_monte_carlo_test() - Main Monte Carlo test loop
 * 
 * Python equivalent: main loop in evaluator() function (python/evaluator.py lines 155-188)
 * 
 * Performs a single iteration of:
 * 1. Sampling tuning_run from data (equivalent to lines 157 in Python)
 * 2. Computing best_so_far (lines 158-162 in Python)
 * 3. Computing tuning_costs with cumsum + overhead (lines 164-166 in Python)
 * 4. Computing running_costs (lines 167-169 in Python)
 * 5. Computing total_runtimes and finding crystal_ball minimum (lines 170-171 in Python)
 * 6. Calling budget_estimator to get estimate (lines 173-181 in Python)
 * 7. Computing extra_runtime (line 183 in Python)
 * 8. Recording results (lines 185-188 in Python)
 */
static TestResult run_monte_carlo_test(const TuningData* data, uint64_t curve_limit, uint64_t total_kernel_runs, uint64_t overhead, double k, double hist_a, double hist_b) {
    double* tuning_run = malloc(curve_limit * sizeof(double));
    double* best_so_far = malloc(curve_limit * sizeof(double));
    double* tuning_costs = malloc(curve_limit * sizeof(double));
    double* running_costs = malloc(curve_limit * sizeof(double));
    
    /* Line 157 in Python: tuning_run = all_config_runtimes.sample(n = curve_limit).values */
    for (uint64_t i = 0; i < curve_limit; i++) tuning_run[i] = data->data[random_index(data->size)];
    
    /* Lines 158-162 in Python: best_so_far.append() loop */
    best_so_far[0] = tuning_run[0];
    for (uint64_t i = 1; i < curve_limit; i++) best_so_far[i] = (tuning_run[i] < best_so_far[i-1]) ? tuning_run[i] : best_so_far[i-1];
    
    /* Lines 164-166 in Python: tuning_runtimes = np.cumsum(), tuning_overhead, tuning_costs */
    double cumsum = 0;
    for (uint64_t i = 0; i < curve_limit; i++) { cumsum += tuning_run[i]; tuning_costs[i] = cumsum + (i + 1) * overhead; }
    
    /* Lines 167-169 in Python: running_costs computation */
    for (uint64_t i = 0; i < curve_limit; i++) running_costs[i] = best_so_far[i] * (total_kernel_runs - i);
    
    /* Lines 170-171 in Python: total_runtimes and crystal_ball = min(total_runtimes) */
    double min_runtime = 1e300; uint64_t crystal_idx = 0;
    for (uint64_t i = 0; i < curve_limit; i++) { double total = tuning_costs[i] + running_costs[i]; if (total < min_runtime) { min_runtime = total; crystal_idx = i; } }

    /* Lines 146-147 in Python: k > 0.0 && k < 1.0 case - get historical_tuning_steps */
    uint64_t default_tuning_steps = 0;
    if (k > 0.0 && k < 1.0) {
        default_tuning_steps = history_run(0, "", total_kernel_runs, overhead, 1);  /* Calls history() in Python */
    }

    /* Lines 173-181 in Python: budget_estimator.tuning_length_recommendation() call */
    uint64_t estimate = recommend_tuning_length(default_tuning_steps, tuning_run, curve_limit, NULL, NULL, total_kernel_runs, k, DEFAULT_FIT_START, overhead, hist_a, hist_b);
    
    /* Line 183 in Python: extra_runtime calculation */
    double extra_runtime = 0;
    if (estimate < curve_limit) { double est_runtime = tuning_costs[estimate] + running_costs[estimate]; extra_runtime = min_runtime > 0 ? est_runtime / min_runtime - 1.0 : 0; }
    
    /* Lines 185-188 in Python: Recording crystal_balls, extra_runtimes, estimates, misses */
    TestResult res = {crystal_idx, estimate, extra_runtime, fabs((double)crystal_idx - (double)estimate)};
    
    free(tuning_run); free(best_so_far); free(tuning_costs); free(running_costs);
    return res;
}

/*
 * total_runtime_remaining() - Calculate remaining runtime for given budget
 * 
 * Python equivalent: budget_estimator.total_runtime_remaining() (python/budget_estimator.py lines 21-22)
 * Formula: (average_runtime_so_far + overhead) * x + fitting_function(starting_step+x, a, b, c) * (total_steps-starting_step-x)
 */
static double total_runtime_remaining(double x, double a, double b, double c, uint64_t start, uint64_t total, double avg_rt, uint64_t overhead) {
    return (avg_rt + overhead) * x + curve_eval(start + x, a, b, c) * (total - start - x);
}

/*
 * local_budget_estimation() - Estimate optimal budget at current tuning step
 * 
 * Python equivalent: budget_estimator.local_budget_estimation() (python/budget_estimator.py lines 67-103)
 * 
 * 1. Fits curve to best_configs_so_far (lines 72-93 in Python)
 * 2. Finds optimal budget using minimize (line 96 in Python)
 * 3. Returns budget if curve_eval < best_configs[-1], else 0 (lines 100-103 in Python)
 * 
 * DIFFERENCES FROM PYTHON:
 * - Python uses scipy.optimize.curve_fit with bounds ([0.1,-inf,-inf], [3,inf,inf])
 *   and scipy.optimize.minimize with bounds
 * - C uses GSL Levenberg-Marquardt implementation (curvefit.c) with different
 *   constraints (CURVEFIT_MIN_DECAY_RATE=0.1, CURVEFIT_MAX_DECAY_RATE=3.0)
 * - Python uses exhaustive search for budget (line 96: minimize), C also uses
 *   exhaustive search (line 110-111)
 * - May produce slightly different curve fit results due to different algorithms
 */
static uint64_t local_budget_estimation(uint64_t current, uint64_t total, double avg_rt, double* best_cfg, uint64_t best_len, uint64_t fit_start, uint64_t overhead, double hist_a, double hist_b) {
    /* Lines 72-93 in Python: Xvalues, Yvalues setup and curve_fit call */
    double* X = malloc(best_len * sizeof(double));
    for (uint64_t i = 0; i < best_len; i++) X[i] = i;
    double a, b, c; curve_fit(X, best_cfg, best_len, fit_start, &a, &b, &c, hist_a, hist_b); free(X);
    
    /* Lines 96-97 in Python: optimize.minimize to find best budget */
    double best_budget = 1, best_runtime = total_runtime_remaining(1, a, b, c, current, total, avg_rt, overhead);
    for (uint64_t bgt = 2; bgt <= total - current; bgt++) { double rt = total_runtime_remaining(bgt, a, b, c, current, total, avg_rt, overhead); if (rt < best_runtime) { best_runtime = rt; best_budget = bgt; } }
    
    /* Lines 100-103 in Python: Return budget if curve_eval < best_cfg[-1], else 0 */
    uint64_t ib = (uint64_t)(best_budget + 0.5);
    return (curve_eval(current + ib, a, b, c) < best_cfg[best_len-1]) ? ib : 0;
}

/*
 * recommend_tuning_length() - Main budget estimation algorithm
 * 
 * Python equivalent: budget_estimator.tuning_length_recommendation() (python/budget_estimator.py lines 26-65)
 * 
 * Core loop (lines 38-61 in Python):
 * 1. Updates average_runtime_so_far (line 41 in Python)
 * 2. Calls local_budget_estimation when i > fit_start+5 (lines 44-55 in Python)
 * 3. Applies regression_weight blending when default_tuning_steps > 0 (lines 47-50 in Python)
 * 4. Updates best_config and best_configs_so_far (lines 57-58 in Python)
 * 5. Returns early if budget < 1 (lines 60-61 in Python)
 */
uint64_t recommend_tuning_length(uint64_t default_tuning_steps, const double* tuning_run, uint64_t tuning_run_len, const char* HW, const char* file_name, uint64_t total_kernel_runs, double regression_weight, uint64_t fit_start, uint64_t overhead, double hist_a, double hist_b) {
    (void)HW; (void)file_name;
    
    /* Lines 28-34 in Python: Initialization */
    uint64_t max_steps = (total_kernel_runs < tuning_run_len) ? total_kernel_runs : tuning_run_len;
    uint64_t budget = max_steps; double best_config = tuning_run[0];
    double* best_configs = malloc(CURVE_LIMIT_MAX * sizeof(double));
    best_configs[0] = tuning_run[0]; uint64_t best_len = 1; double avg_runtime = tuning_run[0];
    
    /* Line 37 in Python: loop_limit = min(max_tuning_steps, 2000) */
    uint64_t limit = (max_steps < CURVE_LIMIT_MAX) ? max_steps : CURVE_LIMIT_MAX;
    
    /* Main loop: lines 38-61 in Python */
    for (uint64_t i = 1; i < limit; i++) {
        /* Line 41 in Python: average_runtime_so_far = (average_runtime_so_far * i + tuning_run[i]) / (i+1) */
        budget--; avg_runtime = (avg_runtime * i + tuning_run[i]) / (i + 1);
        
        /* Lines 44-55 in Python: local_budget_estimation call and budget update */
        if (i > fit_start + 5) {
            uint64_t new_budget = local_budget_estimation(i, total_kernel_runs, avg_runtime, best_configs, best_len, fit_start, overhead, hist_a, hist_b);
            /* Lines 47-50 in Python: regression_weight blending */
            if (default_tuning_steps > 0) { double rw = (double)i / default_tuning_steps; if (rw > 1.0) rw = 1.0; new_budget = (uint64_t)(rw * new_budget + (1 - rw) * (default_tuning_steps - i)); }
            /* Lines 52-55 in Python: Update budget */
            if (tuning_run[i] < best_config) budget = new_budget; else if (new_budget < budget) budget = new_budget;
        }
        
        /* Lines 57-58 in Python: Update best_config and best_configs_so_far */
        if (tuning_run[i] < best_config) best_config = tuning_run[i];
        if (best_len < CURVE_LIMIT_MAX) best_configs[best_len++] = best_config;
        
        /* Lines 60-61 in Python: Early return if budget < 1 */
        if (budget < 1) { free(best_configs); return i; }
    }
    
    /* Lines 63-65 in Python: Return value if loop completes */
    free(best_configs);
    uint64_t res = (max_steps - 1 < 3000) ? max_steps - 1 : 3000;
    return res > 0 ? res : 1;
}

/*
 * history_run() - Monte Carlo simulation to find historical stopping point
 * 
 * Python equivalent: evaluator.history() (python/evaluator.py lines 17-50)
 * 
 * Runs multiple tests (number_of_tests) and averages the optimal stopping point:
 * 1. Samples tuning_run from historical data (lines 33-34 in Python)
 * 2. Computes best_so_far (lines 35-39 in Python)
 * 3. Computes total_runtimes (lines 41-46 in Python)
 * 4. Finds optimal (line 48 in Python)
 * 5. Returns average + 1 (line 50 in Python)
 */
uint64_t history_run(const char* HW, const char* file_name, uint64_t total_kernel_runs, uint64_t overhead, uint64_t number_of_tests) {
    /* Lines 18-20 in Python: get_full_path and pd.read_csv */
    char path[512], benchmark[256];
    strcpy(benchmark, file_name);
    char* suffix = strstr(benchmark, "_output.csv");
    if (suffix) *suffix = '\0';
    snprintf(path, sizeof(path), "raw-data/raw-autotuning-data/%s/%s-%s", benchmark, HW, file_name);
    TuningData data = csv_load(path, "Computation duration (us)");
    
    /* Line 28 in Python: max_tuning_steps = min(total_kernel_runs, len(all_config_runtimes)) */
    if (!data.data || data.size == 0) { csv_free(&data); return DEFAULT_FIT_START; }
    uint64_t curve_limit = (total_kernel_runs < data.size) ? total_kernel_runs : data.size;
    if (curve_limit > CURVE_LIMIT_MAX) curve_limit = CURVE_LIMIT_MAX;
    
    /* Lines 30-50 in Python: Main simulation loop */
    uint64_t historical_optimum = 0;
    for (uint64_t test = 0; test < number_of_tests; test++) { 
        TestResult res = run_monte_carlo_test(&data, curve_limit, total_kernel_runs, overhead, 1.0, NO_HISTORICAL_DATA, NO_HISTORICAL_DATA); 
        historical_optimum += res.crystal_idx; 
    }
    
    csv_free(&data);
    /* Line 50 in Python: return round(historical_optimum / number_of_tests) + 1 */
    return (historical_optimum / number_of_tests) + 1;
}

/*
 * get_regression_params() - Fit regression parameters from historical data
 * 
 * Python equivalent: evaluator.get_history_regression_parameters() (python/evaluator.py lines 60-101)
 * 
 * 1. Loads historical data (lines 61-68 in Python)
 * 2. Runs multiple tests computing average best_so_far curve (lines 74-91 in Python)
 * 3. Fits curve to average (lines 93-97 in Python)
 * 4. Returns a, b parameters (lines 99-101 in Python)
 */
void get_regression_params(const char* HW, const char* file_name, uint64_t total_kernel_runs, uint64_t fit_start, uint64_t number_of_tests, CurveParams* params) {
    /* Lines 61-68 in Python: Load CSV data */
    char path[512], benchmark[256];
    strcpy(benchmark, file_name);
    char* suffix = strstr(benchmark, "_output.csv");
    if (suffix) *suffix = '\0';
    snprintf(path, sizeof(path), "raw-data/raw-autotuning-data/%s/%s-%s", benchmark, HW, file_name);
    TuningData data = csv_load(path, "Computation duration (us)");
    
    /* Line 72 in Python: curve_limit = min(max_tuning_steps, curve_limit_max) */
    if (!data.data || data.size == 0) { csv_free(&data); params->a = 0.5; params->b = 1.0; params->c = 0; return; }
    uint64_t curve_limit = (total_kernel_runs < data.size) ? total_kernel_runs : data.size;
    if (curve_limit > CURVE_LIMIT_MAX) curve_limit = CURVE_LIMIT_MAX;
    
    /* Lines 74-91 in Python: Compute average curve over multiple tests */
    double* avg_curve = calloc(curve_limit, sizeof(double));
    
    for (uint64_t test = 0; test < number_of_tests; test++) {
        /* Lines 77-78 in Python: Sample tuning_run */
        double* tuning_run = malloc(curve_limit * sizeof(double));
        double* best_so_far = malloc(curve_limit * sizeof(double));
        
        for (uint64_t i = 0; i < curve_limit; i++) tuning_run[i] = data.data[random_index(data.size)];
        
        /* Lines 79-82 in Python: best_so_far computation */
        best_so_far[0] = tuning_run[0];
        for (uint64_t j = 1; j < curve_limit; j++) best_so_far[j] = (tuning_run[j] < best_so_far[j-1]) ? tuning_run[j] : best_so_far[j-1];
        
        /* Lines 84-91 in Python: Running average of best_so_far */
        avg_curve[0] = (avg_curve[0] * test + tuning_run[0]) / (test + 1);
        for (uint64_t j = 1; j < curve_limit; j++) avg_curve[j] = (avg_curve[j] * test + best_so_far[j]) / (test + 1);
        free(tuning_run); free(best_so_far);
    }
    
    /* Lines 93-97 in Python: curve_fit using scipy.optimize.curve_fit */
    double* X = malloc(curve_limit * sizeof(double));
    for (uint64_t i = 0; i < curve_limit; i++) X[i] = i;
    double a = 0.5, b = 1.0, c = avg_curve[curve_limit - 1];
    curve_fit(X, avg_curve, curve_limit, fit_start, &a, &b, &c, NO_HISTORICAL_DATA, NO_HISTORICAL_DATA);
    
    /* Lines 99-101 in Python: Return a, b */
    params->a = a; params->b = b; params->c = c;
    free(X); free(avg_curve); csv_free(&data);
}

/*
 * evaluator_run() - Main entry point for evaluator
 * 
 * Python equivalent: evaluator.evaluator() (python/evaluator.py lines 122-198)
 * 
 * 1. Loads CSV data (lines 124-131 in Python)
 * 2. Determines max_tuning_steps and curve_limit (lines 133-136 in Python)
 * 3. Gets historical regression params if k == 0 (line 153 in Python)
 * 4. Runs Monte Carlo tests (lines 155-188 in Python)
 * 5. Computes averages and standard deviations (lines 190-197 in Python)
 * 
 * DIFFERENCES FROM PYTHON:
 * - DEFAULT_FIT_START: Python default is 15 (line 122), C uses 10 (evaluator.h line 9)
 *   This may cause different behavior when fit_start is not explicitly provided.
 */
void evaluator_run(const EvaluatorParams* params, EvaluatorResult* result) {
    /* Lines 124-131 in Python: Load CSV and extract runtimes */
    char path[512], benchmark[256];
    strcpy(benchmark, params->file_name);
    char* suffix = strstr(benchmark, "_output.csv");
    if (suffix) *suffix = '\0';
    snprintf(path, sizeof(path), "raw-data/raw-autotuning-data/%s/%s-%s", benchmark, params->HW, params->file_name);
    TuningData data = csv_load(path, "Computation duration (us)");
    
    /* Lines 133-136 in Python: max_tuning_steps and curve_limit */
    if (!data.data || data.size == 0) { result->avg_extra_runtime = result->std_extra_runtime = 0; result->avg_crystal_ball = result->std_crystal_ball = 0; result->avg_estimate = result->std_estimate = result->avg_miss = 0; return; }
    
    uint64_t max_steps = (params->total_kernel_runs < data.size) ? params->total_kernel_runs : data.size;
    uint64_t curve_limit = (max_steps < CURVE_LIMIT_MAX) ? max_steps : CURVE_LIMIT_MAX;
    
    /* Line 153 in Python: Get regression parameters if k == 0.0 */
    double hist_a = NO_HISTORICAL_DATA, hist_b = NO_HISTORICAL_DATA;
    if (params->k == 0.0) { CurveParams cp; get_regression_params(params->hist_HW, params->file_name, params->total_kernel_runs, params->fit_start, params->number_of_tests, &cp); hist_a = cp.a; hist_b = cp.b; }
    
    /* Lines 138-148 in Python: Initialize accumulators */
    double sum_extra = 0, sum_crystal = 0, sum_estimate = 0, sum_miss = 0, sum_extra_sq = 0, sum_crystal_sq = 0, sum_estimate_sq = 0;
    
    /* Lines 155-188 in Python: Main Monte Carlo loop over number_of_tests */
    for (uint64_t test = 0; test < params->number_of_tests; test++) {
        TestResult res = run_monte_carlo_test(&data, curve_limit, params->total_kernel_runs, params->overhead, params->k, hist_a, hist_b);
        sum_extra += res.extra_runtime; sum_extra_sq += res.extra_runtime * res.extra_runtime;
        sum_crystal += res.crystal_idx + 1; sum_crystal_sq += (res.crystal_idx + 1) * (res.crystal_idx + 1);
        sum_estimate += res.estimate; sum_estimate_sq += res.estimate * res.estimate;
        sum_miss += res.miss;
    }
    
    /* Lines 190-197 in Python: Compute averages and standard deviations */
    uint64_t n = params->number_of_tests;
    
    result->avg_extra_runtime = sum_extra / n; result->avg_crystal_ball = sum_crystal / n;
    result->avg_estimate = sum_estimate / n; result->avg_miss = sum_miss / n;
    result->std_extra_runtime = sqrt(sum_extra_sq / n - result->avg_extra_runtime * result->avg_extra_runtime);
    result->std_crystal_ball = sqrt(sum_crystal_sq / n - result->avg_crystal_ball * result->avg_crystal_ball);
    result->std_estimate = sqrt(sum_estimate_sq / n - result->avg_estimate * result->avg_estimate);
    
    csv_free(&data);
}
