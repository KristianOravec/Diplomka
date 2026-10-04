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

/* Frozen curve a, b for historical mode (instead of get_regression_params).
 * Key: hist_hw, file_name, total_kernel_runs, fit_start, number_of_tests. */
bool hist_cache_lookup_regression(const char *hist_hw, const char *file_name,
                                  uint64_t total_kernel_runs,
                                  uint64_t fit_start, uint64_t number_of_tests,
                                  double *out_a, double *out_b);

/* O_hist for hybrid mode (instead of history_run).
 * Key: hist_hw, file_name, total_kernel_runs, overhead, number_of_tests. */
bool hist_cache_lookup_optimum(const char *hist_hw, const char *file_name,
                               uint64_t total_kernel_runs, uint64_t overhead,
                               uint64_t number_of_tests,
                               uint64_t *out_optimal_steps);

/* Number of loaded rows (0 = no cache file). For diagnostics. */
int hist_cache_row_count(void);

#ifdef __cplusplus
}
#endif
#endif /* HISTORICAL_CACHE_H */
