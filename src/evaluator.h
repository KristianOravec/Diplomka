/* =============================================================================
 * evaluator.h  -- top-level estimator interface (shared with the Python ctypes
 *                 caller during validation)
 *
 * Declares the batch Monte Carlo evaluator, the historical-data helpers, and
 * the batch budget recommender, plus the input/output structs shared across the
 * C/Python boundary. Detailed parameter documentation lives here in the header
 * (the contract); evaluator.c holds the implementation notes.
 * =============================================================================
 */
#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "curvefit.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- constants ---- */
#define CURVE_LIMIT_MAX 2000 /* hard cap on simulated run length / curve size  \
                              */
#define DEFAULT_FIT_START 15 /* fallback fit-start (NOTE: tuner uses 10) */
#define NO_HISTORICAL_DATA -1.0 /* sentinel: "no historical a/b supplied", also "this particular call has no a,b to freeze"  */


/* -----------------------------------------------------------------------------
 * EvaluatorParams -- INPUT to evaluator_run.
 * Field order/types MUST match the ctypes Structure on the Python side exactly,
 * or the data is misread across the boundary.
 *
 *   HW                : [in] hardware id for the MAIN data, e.g. "1070".
 *   file_name         : [in] data file, e.g. "gemm-reduced_output.csv".
 *   total_kernel_runs : [in] #E, how many times the final config will run.
 *   hist_HW           : [in] hardware id for the HISTORICAL data (k=0 /
 * hybrid). k                 : [in] estimator variant: 1=live, 0=historical,
 * (0,1)=hybrid. overhead          : [in] per-step tuning overhead. fit_start :
 * [in] index to begin curve fitting from. number_of_tests   : [in] how many
 * Monte Carlo trials to average over.
 * ---------------------------------------------------------------------------
 */
typedef struct {
    const char *HW;
    const char *file_name;
    uint64_t total_kernel_runs;
    const char *hist_HW;
    double k;
    uint64_t overhead;
    uint64_t fit_start;
    uint64_t number_of_tests;
} EvaluatorParams;

/* -----------------------------------------------------------------------------
 * EvaluatorResult -- OUTPUT from evaluator_run (also mirrored on the Python
 * side). avg_* are means over all trials; std_* are their standard deviations.
 *
 *   avg/std_extra_runtime : [out] performance decline (headline metric) +
 * spread. avg/std_crystal_ball  : [out] ORACLE stop step (mean + spread).
 *   avg/std_estimate      : [out] ESTIMATOR stop step (mean + spread).
 *   avg_miss              : [out] mean |oracle - estimate|.
 * ---------------------------------------------------------------------------
 */
typedef struct {
    double avg_extra_runtime, std_extra_runtime;
    double avg_crystal_ball, std_crystal_ball;
    double avg_estimate, std_estimate;
    double avg_miss;
} EvaluatorResult;

/* -----------------------------------------------------------------------------
 * evaluator_run -- run the full Monte Carlo evaluation for one configuration
 * and average the metrics (mirrors Python's evaluator()). params : [in]  all
 * inputs (see EvaluatorParams). result : [out] filled with the averaged metrics
 * (see EvaluatorResult).
 * ---------------------------------------------------------------------------
 */
void evaluator_run(const EvaluatorParams *params, EvaluatorResult *result);

/* -----------------------------------------------------------------------------
 * history_run -- compute the historical optimum O_hist (the hybrid backstop) by
 * Monte-Carlo-averaging the oracle stop over the historical data.
 *   HW                : [in] historical hardware id.
 *   file_name         : [in] historical data file.
 *   total_kernel_runs : [in] #E for the projection.
 *   overhead          : [in] per-step overhead.
 *   number_of_tests   : [in] trials to average.
 *   returns           : O_hist, the historical optimal number of tuning steps.
 * ---------------------------------------------------------------------------
 */
uint64_t history_run(const char *HW, const char *file_name,
                     uint64_t total_kernel_runs, uint64_t overhead,
                     uint64_t number_of_tests);

/* -----------------------------------------------------------------------------
 * get_regression_params -- fit the historical convergence curve and return its
 * a, b (used to FREEZE the curve shape in k=0 mode).
 *   HW, file_name, total_kernel_runs, fit_start, number_of_tests : [in] as
 * above. params : [out] receives the fitted a, b, c (a, b are reused as
 * hist_a/hist_b).
 * ---------------------------------------------------------------------------
 */
void get_regression_params(const char *HW, const char *file_name,
                           uint64_t total_kernel_runs, uint64_t fit_start,
                           uint64_t number_of_tests, CurveParams *params);

/* -----------------------------------------------------------------------------
 * recommend_tuning_length -- the BATCH estimator: given a whole tuning run,
 * return the step it would stop at. (The reference the deployed API reproduces
 * incrementally.)
 *   default_tuning_steps : [in] O_hist backstop for hybrid; 0 if unused.
 *   tuning_run           : [in] array of per-step runtimes (the whole run).
 *   tuning_run_len       : [in] length of that array.
 *   total_kernel_runs    : [in] #E.
 *   regression_weight    : [in] k (1 live / 0 historical / between = hybrid).
 *   fit_start            : [in] index to begin fitting.
 *   overhead             : [in] per-step overhead.
 *   hist_a, hist_b       : [in] historical curve params, or -1 for none.
 *   returns              : the recommended stopping step.
 * ---------------------------------------------------------------------------
 */
uint64_t
recommend_tuning_length(uint64_t default_tuning_steps, const double *tuning_run,
                        uint64_t tuning_run_len, uint64_t total_kernel_runs,
                        double regression_weight, uint64_t fit_start,
                        uint64_t overhead, double hist_a, double hist_b);

#ifdef __cplusplus
}
#endif
#endif /* EVALUATOR_H */
