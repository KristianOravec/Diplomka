/* =============================================================================
 * historical_cache.c -- lookup of precomputed historical parameters
 *
 * Reads historical_params.csv (written by precompute_historical.py) on first
 * lookup. Path: $TUNER_HISTORICAL_PARAMS if set, else ./historical_params.csv.
 * A miss (no file, no matching row) returns false; the caller then computes.
 *
 * Row kinds and their keys:
 *   REG (a, b)   : hist_hw, file_name, total_kernel_runs, fit_start, tests
 *   OPT (O_hist) : hist_hw, file_name, total_kernel_runs, overhead,  tests
 * ============================================================================= */

#include "historical_cache.h"
#include "evaluator.h" /* CURVE_LIMIT_MAX */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIST_CACHE_MAX_ROWS 512
#define HIST_CACHE_STR_CAP 128

/* -1 in the CSV: column not used by this row kind */
#define HIST_CACHE_NA (-1L)
/* -1 in total_kernel_runs (read as UINT64_MAX): REG row valid for any
 * runs >= CURVE_LIMIT_MAX */
#define HIST_CACHE_ANY_RUNS UINT64_MAX

typedef struct {
    char kind[4]; /* "REG" or "OPT" */
    char hist_hw[16];
    char file_name[HIST_CACHE_STR_CAP];
    uint64_t total_kernel_runs; /* or HIST_CACHE_ANY_RUNS */
    long fit_start;             /* or HIST_CACHE_NA */
    long overhead;              /* or HIST_CACHE_NA */
    uint64_t number_of_tests;
    double a, b;
    long optimal_steps; /* or HIST_CACHE_NA */
} HistCacheRow;

static HistCacheRow hist_cache_rows[HIST_CACHE_MAX_ROWS];
static int hist_cache_count = 0;
static bool hist_cache_tried = false;

static void hist_cache_load(void) {
    if (hist_cache_tried)
        return;
    hist_cache_tried = true;

    const char *path = getenv("TUNER_HISTORICAL_PARAMS");
    if (!path || !*path)
        path = "historical_params.csv";

    FILE *fp = fopen(path, "r");
    if (!fp)
        return;

    char line[1024];
    if (!fgets(line, sizeof(line), fp)) { /* header */
        fclose(fp);
        return;
    }

    while (hist_cache_count < HIST_CACHE_MAX_ROWS &&
           fgets(line, sizeof(line), fp)) {
        HistCacheRow row = {0};

        /* kind,hist_hw,file_name,total_kernel_runs,fit_start,overhead,
         * number_of_tests,a,b,optimal_steps   ("-1" in an unsigned column
         * wraps to the max value, the "any runs" marker) */
        int n = sscanf(line,
                       "%3[^,],%15[^,],%127[^,],%" SCNu64 ",%ld,%ld,%" SCNu64
                       ",%lf,%lf,%ld",
                       row.kind, row.hist_hw, row.file_name,
                       &row.total_kernel_runs, &row.fit_start, &row.overhead,
                       &row.number_of_tests, &row.a, &row.b,
                       &row.optimal_steps);
        if (n < 10)
            continue; /* malformed or over-long field: skip the row */

        hist_cache_rows[hist_cache_count++] = row;
    }
    fclose(fp);
}

/* Key fields shared by REG and OPT rows. total_kernel_runs is checked by each
 * lookup itself, because REG and OPT match it differently. */
static bool hist_cache_base_match(const HistCacheRow *row, const char *kind,
                                  const char *hist_hw, const char *file_name,
                                  uint64_t number_of_tests) {
    return strcmp(row->kind, kind) == 0 &&
           strcmp(row->hist_hw, hist_hw ? hist_hw : "") == 0 &&
           strcmp(row->file_name, file_name ? file_name : "") == 0 &&
           row->number_of_tests == number_of_tests;
}

bool hist_cache_lookup_regression(const char *hist_hw, const char *file_name,
                                  uint64_t total_kernel_runs,
                                  uint64_t fit_start, uint64_t number_of_tests,
                                  double *out_a, double *out_b) {
    hist_cache_load();
    for (int i = 0; i < hist_cache_count; i++) {
        const HistCacheRow *row = &hist_cache_rows[i];
        if (!hist_cache_base_match(row, "REG", hist_hw, file_name, number_of_tests) ||
            row->fit_start != (long)fit_start)
            continue;

        /* The fit uses at most CURVE_LIMIT_MAX steps, so every
         * runs >= CURVE_LIMIT_MAX gives the same a, b.
         * REG(a,b) needs to fit an average curve, and the fit never uses more than 2000 steps.
         * Beyond 2000 steps `total_kernel_runs` has no effect on a, b */
        bool  total_kernel_runs_match;
        if (row->total_kernel_runs == HIST_CACHE_ANY_RUNS)
            total_kernel_runs_match = (total_kernel_runs >= CURVE_LIMIT_MAX);
        else
            total_kernel_runs_match = (row->total_kernel_runs == total_kernel_runs);
        if (!total_kernel_runs_match)
            continue;

        if (out_a)
            *out_a = row->a;
        if (out_b)
            *out_b = row->b;
        return true;
    }
    return false;
}

bool hist_cache_lookup_optimum(const char *hist_hw, const char *file_name,
                               uint64_t total_kernel_runs, uint64_t overhead,
                               uint64_t number_of_tests,
                               uint64_t *out_optimal_steps) {
    hist_cache_load();
    for (int i = 0; i < hist_cache_count; i++) {
        const HistCacheRow *row = &hist_cache_rows[i];
        if (!hist_cache_base_match(row, "OPT", hist_hw, file_name, number_of_tests) ||
            row->total_kernel_runs != total_kernel_runs ||
            row->overhead != (long)overhead)
            continue;

        if (out_optimal_steps && row->optimal_steps != HIST_CACHE_NA)
            *out_optimal_steps = (uint64_t)row->optimal_steps;
        return true;
    }
    return false;
}

int hist_cache_row_count(void) {
    hist_cache_load();
    return hist_cache_count;
}
