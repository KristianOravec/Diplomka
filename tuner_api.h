#ifndef TUNER_API_H
#define TUNER_API_H

#include <stddef.h>



/**
 * budget_estimator.tuning_length_recommendation
 */
typedef struct {
    const double* runtimes;      
    const double* best_so_far;   
    size_t current_step;         
    size_t total_kernel_runs;   
    double avg_runtime_so_far; 
    double overhead;             
} TuningState;


typedef struct {
    int default_tuning_steps;    // Result from 'history()'
    double hist_a;               // Decay param from 'get_history_regression_parameters'
    double hist_b;               // Scale param from 'get_history_regression_parameters'
} HistoricalPriors;

/**
 * @brief Statistical results from an evaluation batch.
 */
typedef struct {
    double avg_extra_runtime;
    double std_extra_runtime;
    double avg_crystal_ball;
    double avg_estimate;
    double avg_miss;
} EvalResult;

/* --- Estimator Interface --- */

/**
 * @brief Predicts when to stop tuning.
 * * Corresponds to 'tuning_length_recommendation' in budget_estimator.py.
 * Returns the recommended number of additional steps (0 = STOP).
 */
int predict_budget(
    const TuningState* state, 
    const HistoricalPriors* priors, 
    double k_weight, 
    int fit_start
);

/* --- Evaluator Interface --- */

/**
 * input for evaluator.evalutor in python
 */
EvalResult run_evaluator(
    const double* full_dataset, 
    size_t dataset_size, 
    size_t total_runs, 
    double k, 
    double overhead, 
    int fit_start,
    const HistoricalPriors* priors
);

/**
 * history() func in evaluator
 */
int calculate_historical_steps(const double* hist_data, size_t size, size_t total_runs, double overhead);


/** 
 * get_history_regression_parameters() in evaluator.py
 */
void calculate_historical_params(const double* hist_data, size_t size, double* out_a, double* out_b);

#endif // TUNER_API_H