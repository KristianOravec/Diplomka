/* =============================================================================
 * test_historical_cache.c -- unit test of the lookup rules in historical_cache.c
 *
 *   REG: one row (runs = -1) for any X >= 2000 -> must hit; X < 2000 -> must
 *        miss (the fit really differs). hist_hw, fit_start, tests must match.
 *   OPT: X and overhead must match exactly.
 *
 * Build and run (from the repo root):
 *   gcc -O2 -Isrc -o test_historical_cache \
 *       testing/test_historical_cache.c src/historical_cache.c
 *   ./test_historical_cache [TABLE_CSV] [NUMBER_OF_TESTS]
 *
 * Without TABLE_CSV: $TUNER_HISTORICAL_PARAMS, else ./historical_params.csv.
 * Exit code: 0 = all passed, 1 = at least one failed.
 * ============================================================================= */

#include "historical_cache.h"
#include <stdio.h>
#include <stdlib.h>

#define CURVE_LIMIT 2000 /* must match CURVE_LIMIT_MAX in evaluator.h */
#define HIST_HW "1070"
#define FILE_NAME "gemm-reduced_output.csv"
#define FIT_START 10

static uint64_t number_of_tests = 1000; /* must match the table */
static int checks = 0;
static int failures = 0;

/* ---- helpers ---------------------------------------------------------------- */

static void check(const char *description, bool passed) {
    checks++;
    if (!passed)
        failures++;
    printf("  [%s] %s\n", passed ? "PASS" : "FAIL", description);
}

/* REG lookup with the default hw, file and number_of_tests */
static bool lookup_reg(uint64_t runs, uint64_t fit_start, double *a, double *b) {
    return hist_cache_lookup_regression(HIST_HW, FILE_NAME, runs, fit_start,
                                        number_of_tests, a, b);
}

/* OPT lookup with the default hw, file and number_of_tests */
static bool lookup_opt(uint64_t runs, uint64_t overhead, uint64_t *o_hist) {
    return hist_cache_lookup_optimum(HIST_HW, FILE_NAME, runs, overhead,
                                     number_of_tests, o_hist);
}

/* ---- tests ------------------------------------------------------------------ */

/* REG rows do not depend on X as long as X >= CURVE_LIMIT. */
static void test_reg_any_runs(double *a_reference) {
    printf("\nREG -- one row for any X >= %d:\n", CURVE_LIMIT);

    double a_10k = 0, b_10k = 0;
    double a_50k = 0, b_50k = 0;
    double a_huge = 0, b_huge = 0;
    double a_small = 0, b_small = 0;

    bool hit_10k = lookup_reg(10000, FIT_START, &a_10k, &b_10k);
    bool hit_50k = lookup_reg(50000, FIT_START, &a_50k, &b_50k);
    bool hit_huge = lookup_reg(999999999, FIT_START, &a_huge, &b_huge);
    bool hit_small = lookup_reg(500, FIT_START, &a_small, &b_small);

    check("X = 10000 (in the grid)      -> hit", hit_10k);
    check("X = 50000 (not in the grid)  -> hit", hit_50k);
    check("X = 999999999                -> hit", hit_huge);
    check("X = 500 (below 2000)         -> miss", !hit_small);

    bool same_values = (a_10k == a_50k && a_10k == a_huge &&
                        b_10k == b_50k && b_10k == b_huge);
    check("all hits return the same a, b", same_values);

    printf("         (a = %.4f, b = %.1f)\n", a_10k, b_10k);
    *a_reference = a_10k;
}

/* The other REG key fields must still select the right row. */
static void test_reg_key_fields(double a_fit_start_10) {
    printf("\nREG -- hist_hw, fit_start, number_of_tests must match:\n");

    double a = 0, b = 0;

    bool hit_fs15 = lookup_reg(10000, 15, &a, &b);
    check("fit_start = 15               -> hit", hit_fs15);
    check("fit_start = 15 gives a different a than 10",
          hit_fs15 && a != a_fit_start_10);

    bool hit_unknown_hw = hist_cache_lookup_regression(
        "NO_SUCH_HW", FILE_NAME, 10000, FIT_START, number_of_tests, &a, &b);
    check("unknown hist_hw              -> miss", !hit_unknown_hw);

    bool hit_unknown_tests = hist_cache_lookup_regression(
        HIST_HW, FILE_NAME, 10000, FIT_START, 12345, &a, &b);
    check("number_of_tests = 12345      -> miss", !hit_unknown_tests);
}

/* OPT rows depend on X and overhead, so both must match exactly. */
static void test_opt_exact_match(void) {
    printf("\nOPT -- X and overhead must match exactly:\n");

    uint64_t o_hist = 0;

    bool hit_in_grid = lookup_opt(10000, 10000, &o_hist);
    check("X = 10000, overhead = 10000  -> hit", hit_in_grid);
    printf("         (O_hist = %llu)\n", (unsigned long long)o_hist);

    bool hit_other_runs = lookup_opt(50000, 10000, &o_hist);
    check("X = 50000 (not in the grid)  -> miss", !hit_other_runs);

    bool hit_other_overhead = lookup_opt(10000, 777, &o_hist);
    check("overhead = 777               -> miss", !hit_other_overhead);
}

/* ---- main ------------------------------------------------------------------- */

int main(int argc, char **argv) {
    /* optional arguments: table path, number_of_tests */
    if (argc > 1)
        setenv("TUNER_HISTORICAL_PARAMS", argv[1], 1);
    if (argc > 2)
        number_of_tests = strtoull(argv[2], NULL, 10);

    const char *table = getenv("TUNER_HISTORICAL_PARAMS");
    int rows = hist_cache_row_count();

    printf("table: %s (%d rows)\n",
           table ? table : "./historical_params.csv", rows);
    if (rows == 0) {
        printf("no rows loaded -- generate the table with "
               "precompute_historical.py or pass its path\n");
        return 1;
    }

    double a_fit_start_10 = 0;
    test_reg_any_runs(&a_fit_start_10);
    test_reg_key_fields(a_fit_start_10);
    test_opt_exact_match();

    printf("\n%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
