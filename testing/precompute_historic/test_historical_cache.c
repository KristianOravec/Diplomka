/* =============================================================================
 * test_historical_cache.c -- verify historical_cache.c answers lookups the way
 *                            the tuner needs, especially the REG key rule.
 *
 * WHY THIS TEST EXISTS
 *   REG rows (the frozen curve a,b for k=0) do NOT depend on total_kernel_runs.
 *   The fit uses curve_limit = min(runs, pool size, CURVE_LIMIT_MAX), so every
 *   runs >= CURVE_LIMIT_MAX is the SAME computation. The precompute scripts
 *   therefore emit ONE row per (hist_hw, fit_start) with total_kernel_runs
 *   stored as -1, meaning "matches any runs >= CURVE_LIMIT_MAX".
 *
 *   Getting the lookup wrong has two failure modes, and this test targets both:
 *
 *     TOO STRICT -- the lookup insists on an exact runs match, so the canonical
 *       -1 row never matches anything. Every REG lookup misses and the library
 *       recomputes at startup, which is exactly what the cache was built to
 *       avoid. Caught by cases 2 and 3 below.
 *
 *     TOO LOOSE -- the lookup accepts the -1 row for ANY runs, including
 *       runs < CURVE_LIMIT_MAX. Those genuinely DO change the fit, so the
 *       library would silently serve a wrong curve to small-X kernels. Caught
 *       by case 4, which is the one most likely to be wrong.
 *
 *   Note the -1 subtlety: the CSV holds "-1" but the struct field is uint64_t
 *   and the loader reads it with %llu, so it lands as UINT64_MAX. A comparison
 *   written against a signed -1 will not match.
 *
 * USAGE
 *   gcc -O2 -Isrc -o test_historical_cache \
 *       testing/test_historical_cache.c src/historical_cache.c
 *   ./test_historical_cache [TABLE_CSV] [NUMBER_OF_TESTS]
 *
 *   With no argument it uses $TUNER_HISTORICAL_PARAMS, else
 *   ./historical_params.csv -- the same resolution historical_cache.c itself
 *   performs. Run from wherever that table lives.
 *
 * EXIT CODE
 *   0 = all checks passed, 1 = at least one failed (usable in CI / a script).
 * ============================================================================= */

#include "historical_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mirrors CURVE_LIMIT_MAX in evaluator.h. Declared locally so this test does
 * not drag the whole evaluator in; if that constant ever changes, change it
 * here too (the test will start failing loudly, which is the point). */
#define TEST_CURVE_LIMIT_MAX 2000

#define HW        "1070"
#define FILE_NAME "gemm-reduced_output.csv"
#define FIT_START 10

/* number_of_tests is part of every key, so it must match the table being
 * tested. Default 1000 (the precompute default); override with argv[2] when
 * checking a table generated with --tests N. */
static uint64_t TESTS = 1000;

static int failures = 0;
static int checks = 0;

/* ---- one assertion: did the lookup hit or miss as expected? ---- */
static void expect(const char *what, int got_hit, int want_hit,
                   double a, double b) {
    checks++;
    int ok = (got_hit == want_hit);
    if (!ok) failures++;
    printf("  [%s] %-58s", ok ? "PASS" : "FAIL", what);
    if (got_hit)
        printf(" hit  a=%.4f b=%.1f\n", a, b);
    else
        printf(" miss\n");
    if (!ok)
        printf("         expected %s\n", want_hit ? "HIT" : "MISS");
}

int main(int argc, char **argv) {
    /* Point the library at the table. Passing a path on the command line just
     * sets the same env var historical_cache.c reads, so the test exercises the
     * real resolution path rather than a special case. */
    if (argc > 1)
        setenv("TUNER_HISTORICAL_PARAMS", argv[1], 1);
    if (argc > 2)
        TESTS = strtoull(argv[2], NULL, 10);

    const char *tbl = getenv("TUNER_HISTORICAL_PARAMS");
    printf("historical_cache test\n");
    printf("  table: %s\n", tbl ? tbl : "./historical_params.csv (default)");

    int rows = historical_cache_row_count();
    printf("  rows loaded: %d   (matching number_of_tests=%llu)\n\n", rows,
           (unsigned long long)TESTS);
    if (rows == 0) {
        printf("  NO ROWS LOADED -- the table was not found or is empty.\n");
        printf("  Generate it first:  python3 precompute_historical.py\n");
        printf("  then run this test from the directory holding the CSV, or\n");
        printf("  pass the path:      ./test_historical_cache <table.csv>\n");
        return 1;
    }

    double a = 0, b = 0;
    int hit;

    /* ---------------------------------------------------------------------
     * REG: the runs-independent key
     * ------------------------------------------------------------------- */
    printf("REG rows (frozen curve) -- total_kernel_runs must NOT be part of\n"
           "the key for any runs >= %d:\n", TEST_CURVE_LIMIT_MAX);

    /* 1. a runs value that IS in the precompute grid */
    a = b = 0;
    hit = historical_cache_lookup_regression(HW, FILE_NAME, 10000, FIT_START,
                                             TESTS, &a, &b);
    expect("runs=10000 (in the grid)", hit, 1, a, b);

    /* 2. a runs value that is NOT in the grid -- the whole point of the
     *    canonical row. A tuner may be configured with any X. */
    double a2 = 0, b2 = 0;
    hit = historical_cache_lookup_regression(HW, FILE_NAME, 50000, FIT_START,
                                             TESTS, &a2, &b2);
    expect("runs=50000 (NOT in the grid, must still hit)", hit, 1, a2, b2);

    /* 3. an absurdly large runs value -- still >= CURVE_LIMIT_MAX */
    double a3 = 0, b3 = 0;
    hit = historical_cache_lookup_regression(HW, FILE_NAME, 999999999ULL,
                                             FIT_START, TESTS, &a3, &b3);
    expect("runs=999999999 (huge, must still hit)", hit, 1, a3, b3);

    /* 4. THE IMPORTANT NEGATIVE CASE: below CURVE_LIMIT_MAX the fit genuinely
     *    differs, so the canonical row must NOT be served. If this hits, the
     *    lookup is too loose and small-X kernels get a wrong curve. */
    double a4 = 0, b4 = 0;
    hit = historical_cache_lookup_regression(HW, FILE_NAME, 500, FIT_START,
                                             TESTS, &a4, &b4);
    expect("runs=500 (< CURVE_LIMIT_MAX, must MISS)", hit, 0, a4, b4);

    /* 5. consistency: every hit above must return the SAME a,b -- they are all
     *    the one canonical row. */
    checks++;
    if (a == a2 && a == a3 && b == b2 && b == b3) {
        printf("  [PASS] all runs values returned the identical a,b\n");
    } else {
        failures++;
        printf("  [FAIL] hits disagreed: a=%.6f/%.6f/%.6f  b=%.3f/%.3f/%.3f\n",
               a, a2, a3, b, b2, b3);
        printf("         the canonical row must be the only REG row matched\n");
    }

    /* ---------------------------------------------------------------------
     * REG: the parts of the key that DO matter
     * ------------------------------------------------------------------- */
    printf("\nREG -- fields that MUST still discriminate:\n");

    double ax = 0, bx = 0;
    hit = historical_cache_lookup_regression(HW, FILE_NAME, 10000, 15, TESTS,
                                             &ax, &bx);
    expect("fit_start=15 (different curve, should hit its own row)", hit, 1,
           ax, bx);
    if (hit) {
        checks++;
        if (ax != a) {
            printf("  [PASS] fit_start=15 returned a different curve than 10\n");
        } else {
            failures++;
            printf("  [FAIL] fit_start 10 and 15 returned the SAME a -- "
                   "fit_start is not discriminating\n");
        }
    }

    hit = historical_cache_lookup_regression("NO_SUCH_HW", FILE_NAME, 10000,
                                             FIT_START, TESTS, &ax, &bx);
    expect("unknown hist_hw (must MISS)", hit, 0, ax, bx);

    hit = historical_cache_lookup_regression(HW, FILE_NAME, 10000, FIT_START,
                                             12345, &ax, &bx);
    expect("unknown number_of_tests (must MISS)", hit, 0, ax, bx);

    /* ---------------------------------------------------------------------
     * OPT: O_hist DOES depend on runs -- exact matching must be preserved
     * ------------------------------------------------------------------- */
    printf("\nOPT rows (historical optimum) -- runs and overhead BOTH matter:\n");

    uint64_t steps = 0;
    hit = historical_cache_lookup_optimum(HW, FILE_NAME, 10000, 10000, TESTS,
                                          &steps);
    checks++;
    if (hit) {
        printf("  [PASS] runs=10000 oh=10000 (in the grid)            "
               "hit  O_hist=%llu\n", (unsigned long long)steps);
    } else {
        failures++;
        printf("  [FAIL] runs=10000 oh=10000 (in the grid)            miss\n");
    }

    /* unlike REG, a runs value outside the grid SHOULD miss here: O_hist
     * genuinely changes with runs, so there is nothing valid to serve. */
    uint64_t s2 = 0;
    hit = historical_cache_lookup_optimum(HW, FILE_NAME, 50000, 10000, TESTS,
                                          &s2);
    expect("runs=50000 (NOT in the grid, must MISS -- O_hist needs runs)",
           hit, 0, 0, 0);

    uint64_t s3 = 0;
    hit = historical_cache_lookup_optimum(HW, FILE_NAME, 10000, 777, TESTS,
                                          &s3);
    expect("overhead=777 (not in the grid, must MISS)", hit, 0, 0, 0);

    /* ---------------------------------------------------------------------- */
    printf("\n%d checks, %d failed\n", checks, failures);
    if (failures == 0) {
        printf("ALL PASSED -- the cache serves the canonical REG row for any\n"
               "runs >= %d, keeps fit_start/hw/tests discriminating, and still\n"
               "matches OPT rows exactly on runs and overhead.\n",
               TEST_CURVE_LIMIT_MAX);
    } else {
        printf("FAILURES -- see the [FAIL] lines above. The most common cause\n"
               "is the REG comparison: it must treat a stored total_kernel_runs\n"
               "of UINT64_MAX (written as -1 in the CSV) as matching any\n"
               "requested runs >= %d, and nothing else.\n",
               TEST_CURVE_LIMIT_MAX);
    }
    return failures ? 1 : 0;
}
