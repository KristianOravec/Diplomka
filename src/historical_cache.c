/* =============================================================================
 * historical_cache.c -- load PRECOMPUTED historical parameters from disk
 *                       instead of recomputing them at every initiate_kernel().
 *
 * THE PROBLEM THIS SOLVES
 *   The historical inputs for k=0 and 0<k<1 depend only on the historical data
 *   and a few config values -- never on the live tuning run. Yet
 *   initiate_kernel() recomputed them every single time:
 *     k=0   : get_regression_params() -- averages `number_of_tests` simulated
 *             runs, then fits a curve.
 *     0<k<1 : history_run() -- simulates `number_of_tests` COMPLETE tuning
 *             sessions to find the average oracle stop.
   With number_of_tests = 1000 that is seconds to minutes PER KERNEL, and the
   answer is identical every time (each Monte-Carlo trial is seeded from its
   own test index + salt, so the result is a pure function of the inputs).
   In a deployed tuner that stall is unacceptable.
 *
 *   So precompute_historical.py computes them once, offline, and writes
 *   historical_params.csv. This module loads that table and answers lookups in
 *   O(n) over a handful of rows.
 *
 * TWO KEYS, NOT ONE
 *   The two quantities depend on DIFFERENT parameters, so they cannot share a
 *   cache key:
 *     REG (a, b)    : hist_hw, file_name, total_kernel_runs, fit_start, tests
 *                     -- overhead is irrelevant to the curve fit.
 *     OPT (O_hist)  : hist_hw, file_name, total_kernel_runs, overhead, tests
 *                     -- fit_start is irrelevant to the oracle scan.
 *
 * FALLBACK
 *   A miss (no file, or no matching row) returns 0 and the caller computes as
 *   before. A missing cache therefore costs performance, never correctness.
 *
 * FILE LOCATION
 *   $TUNER_HISTORICAL_PARAMS if set, else ./historical_params.csv
 * ============================================================================= */

#include "historical_cache.h"
#include <stdio.h>
#include "evaluator.h"
#include <stdlib.h>
#include <string.h>

#define HC_MAX_ROWS 512
#define HC_STR_CAP 128

typedef struct {
    char kind[4]; /* "REG" or "OPT" */
    char hist_hw[16];
    char file_name[HC_STR_CAP];
    uint64_t total_kernel_runs;
    long fit_start; /* -1 when not applicable */
    long overhead;  /* -1 when not applicable */
    uint64_t number_of_tests;
    double a, b;
    long optimal_steps; /* -1 when not applicable */
} HcRow;

static HcRow hc_rows[HC_MAX_ROWS];
static int hc_count = 0;
static int hc_loaded = 0; /* 0 = not tried yet, 1 = tried (success or not) */

/* -----------------------------------------------------------------------------
 * hc_load -- read the CSV on first use. Called lazily so a program that never
 * uses historical modes pays nothing.
 * --------------------------------------------------------------------------- */
static void hc_load(void) {
    if (hc_loaded)
        return;
    hc_loaded = 1; /* set first: a failed load must not be retried per lookup */

    const char *path = getenv("TUNER_HISTORICAL_PARAMS");
    if (!path || !*path)
        path = "historical_params.csv";

    FILE *fp = fopen(path, "r");
    if (!fp)
        return; /* no cache -> every lookup misses -> caller computes */

    char line[1024];
    if (!fgets(line, sizeof(line), fp)) { /* skip the header */
        fclose(fp);
        return;
    }

    while (fgets(line, sizeof(line), fp) && hc_count < HC_MAX_ROWS) {
        HcRow r;
        memset(&r, 0, sizeof(r));
        char kind[32] = "", hw[64] = "", fname[HC_STR_CAP] = "";
        unsigned long long runs = 0, tests = 0;
        long fs = -1, oh = -1, opt = -1;
        double a = -1, b = -1;

        /* kind,hist_hw,file_name,total_kernel_runs,fit_start,overhead,
         * number_of_tests,a,b,optimal_steps */
        int n = sscanf(line, "%31[^,],%63[^,],%127[^,],%llu,%ld,%ld,%llu,%lf,%lf,%ld",
                       kind, hw, fname, &runs, &fs, &oh, &tests, &a, &b, &opt);
        if (n < 10)
            continue; /* malformed row: ignore rather than abort */

        snprintf(r.kind, sizeof(r.kind), "%s", kind);
        snprintf(r.hist_hw, sizeof(r.hist_hw), "%s", hw);
        snprintf(r.file_name, sizeof(r.file_name), "%s", fname);
        r.total_kernel_runs = (uint64_t)runs;
        r.fit_start = fs;
        r.overhead = oh;
        r.number_of_tests = (uint64_t)tests;
        r.a = a;
        r.b = b;
        r.optimal_steps = opt;
        hc_rows[hc_count++] = r;
    }
    fclose(fp);
}

/* shared key fields; the caller checks the discriminating one */
static int hc_base_match(const HcRow *r, const char *kind, const char *hist_hw,
                         const char *file_name, uint64_t total_kernel_runs,
                         uint64_t number_of_tests) {
    return strcmp(r->kind, kind) == 0 &&
           strcmp(r->hist_hw, hist_hw ? hist_hw : "") == 0 &&
           strcmp(r->file_name, file_name ? file_name : "") == 0 &&
           r->total_kernel_runs == total_kernel_runs &&
           r->number_of_tests == number_of_tests;
}

int historical_cache_lookup_regression(const char *hist_hw,
                                       const char *file_name,
                                       uint64_t total_kernel_runs,
                                       uint64_t fit_start,
                                       uint64_t number_of_tests, double *out_a,
                                       double *out_b) {
    hc_load();
    for (int i = 0; i < hc_count; i++) {
        HcRow *r = &hc_rows[i];
        if (strcmp(r->kind, "REG") != 0)
            continue;
        if (strcmp(r->hist_hw, hist_hw ? hist_hw : "") != 0)
            continue;
        if (strcmp(r->file_name, file_name ? file_name : "") != 0)
            continue;
        if (r->number_of_tests != number_of_tests)
            continue;
        if (r->fit_start != (long)fit_start)
            continue;

        /* total_kernel_runs is NOT part of the REG key. The fit uses
         * curve_limit = min(runs, pool size, CURVE_LIMIT_MAX), so every
         * runs >= CURVE_LIMIT_MAX produces the same a,b. A row stored with
         * total_kernel_runs = -1 is the canonical "any large runs" entry and
         * matches any such request; rows with a concrete value still require
         * an exact match, because runs < CURVE_LIMIT_MAX genuinely changes
         * the result. */
        int matches = (r->total_kernel_runs == (uint64_t)-1)
                          ? (total_kernel_runs >= CURVE_LIMIT_MAX)
                          : (r->total_kernel_runs == total_kernel_runs);
        if (!matches)
            continue;

        if (out_a) *out_a = r->a;
        if (out_b) *out_b = r->b;
        return 1;
    }
    return 0; /* miss -> caller must compute */
}

int historical_cache_lookup_optimum(const char *hist_hw, const char *file_name,
                                    uint64_t total_kernel_runs,
                                    uint64_t overhead,
                                    uint64_t number_of_tests,
                                    uint64_t *out_optimal_steps) {
    hc_load();
    for (int i = 0; i < hc_count; i++) {
        HcRow *r = &hc_rows[i];
        /* overhead discriminates OPT rows; fit_start is not part of the key */
        if (hc_base_match(r, "OPT", hist_hw, file_name, total_kernel_runs,
                          number_of_tests) &&
            r->overhead == (long)overhead) {
            if (out_optimal_steps && r->optimal_steps >= 0)
                *out_optimal_steps = (uint64_t)r->optimal_steps;
            return 1;
        }
    }
    return 0;
}

int historical_cache_row_count(void) {
    hc_load();
    return hc_count;
}
