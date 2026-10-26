/* =============================================================================
 * historical_cache.h -- lookup of precomputed historical parameters
 *
 * Both lookups return true on a hit (outputs written) and false on a miss, in
 * which case the caller computes the value itself.
 * ============================================================================= */
#ifndef HISTORICAL_CACHE_H
#define HISTORICAL_CACHE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Precomputed curve shape a, b for hist mode, so get_regression_params doesn't have to run.
 * Looked up by: old GPU, data file, total_kernel_runs, fit_start, number_of_tests.
 * Returns true and fills out_a, out_b if found; false = not in table, compute it yourself. */
bool hist_cache_lookup_regression(const char *hist_hw, const char *file_name,
                                  uint64_t total_kernel_runs,
                                  uint64_t fit_start, uint64_t number_of_tests,
                                  double *out_a, double *out_b);


/* Precomputed O_hist (best number of tuning steps on the old GPU) for hybrid mode,
 * so history_run doesn't have to run.
 * Looked up by: old GPU, data file, total_kernel_runs, overhead, number_of_tests.
 * Returns true and fills out_optimal_steps if found; false = not in table. */
bool hist_cache_lookup_optimum(const char *hist_hw, const char *file_name,
                               uint64_t total_kernel_runs, uint64_t overhead,
                               uint64_t number_of_tests,
                               uint64_t *out_optimal_steps);

/* How many rows the table has (0 = no cache file loaded). Only for debugging. */
int hist_cache_row_count(void);

#ifdef __cplusplus
}
#endif
#endif /* HISTORICAL_CACHE_H */
