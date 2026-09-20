/* =============================================================================
 * historical_cache.h -- lookup of PRECOMPUTED historical parameters.
 *
 * The historical inputs for k=0 and 0<k<1 depend only on the historical data,
 * never on the live run, so they are computed once offline by
 * precompute_historical.py and loaded from historical_params.csv instead of
 * being recomputed at every initiate_kernel().
 *
 * Both functions return 1 on a hit (outputs written) and 0 on a miss, in which
 * case the caller should compute the value the old way. A missing or incomplete
 * cache therefore costs performance, never correctness.
 *
 * The table is loaded lazily on first lookup from:
 *   $TUNER_HISTORICAL_PARAMS   if set, else   ./historical_params.csv
 * ============================================================================= */
#ifndef HISTORICAL_CACHE_H
#define HISTORICAL_CACHE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * Frozen-curve parameters for k=0 (replaces get_regression_params).
 * KEY: hist_hw, file_name, total_kernel_runs, fit_start, number_of_tests.
 *      `overhead` is deliberately NOT part of the key -- the curve fit does not
 *      depend on it.
 *   out_a, out_b : [out] the frozen decay and scale.
 *   returns      : 1 = found, 0 = not cached (compute it instead).
 * --------------------------------------------------------------------------- */
int historical_cache_lookup_regression(const char *hist_hw,
                                       const char *file_name,
                                       uint64_t total_kernel_runs,
                                       uint64_t fit_start,
                                       uint64_t number_of_tests, double *out_a,
                                       double *out_b);

/* -----------------------------------------------------------------------------
 * Historical optimum O_hist for hybrid (replaces history_run).
 * KEY: hist_hw, file_name, total_kernel_runs, overhead, number_of_tests.
 *      `fit_start` is deliberately NOT part of the key -- the oracle scan does
 *      not depend on it.
 *   out_optimal_steps : [out] O_hist, the historical optimal tuning length.
 *   returns           : 1 = found, 0 = not cached (compute it instead).
 * --------------------------------------------------------------------------- */
int historical_cache_lookup_optimum(const char *hist_hw, const char *file_name,
                                    uint64_t total_kernel_runs,
                                    uint64_t overhead,
                                    uint64_t number_of_tests,
                                    uint64_t *out_optimal_steps);

/* How many rows were loaded (0 = no cache file found). For diagnostics. */
int historical_cache_row_count(void);

#ifdef __cplusplus
}
#endif
#endif /* HISTORICAL_CACHE_H */
