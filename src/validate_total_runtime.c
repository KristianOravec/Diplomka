/* =============================================================================
 * validate_total_runtime.c
 *
 * Tests the DEPLOYED API on the metric that actually matters: TOTAL RUNTIME,
 * not just the stopping step. For each trial it drives initiate/push/find to a
 * stop point, then computes:
 *
 *     extra% = total_runtime(estimator's stop) / total_runtime(oracle's stop) -
 * 1
 *
 * i.e. how much worse the estimator's stop is than the best-possible stop, in
 * total application runtime. This is exactly Python's "Average extra runtime"
 * (Performance decline), so we compare against those ground-truth values.
 *
 * WHY THIS IS DIFFERENT FROM validate_tuner_api.c:
 *   That harness checks WHERE the estimator stops (step count). This one checks
 *   WHAT IT COSTS to stop there (total runtime). Two estimators can stop at
 *   different steps yet have nearly identical total runtime if the cost curve
 * is flat near the optimum -- so total runtime is the meaningful correctness
 * test.
 *
 * USAGE:  ./validate_total_runtime [trials] [mode] [outfile]
 *   trials  : MC trials per config (default 300)
 *   mode    : live | hist | hybrid | all   (default all)
 *   outfile : results file (default total_runtime_results.txt). Rows are
 *             written and flushed as each config finishes, so an interrupted
 *             run keeps its completed results and you can watch progress with
 *             `tail -f`.
 *
 * Run from the dir containing raw-data/.
 * =============================================================================
 */

#include "tuner_api.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- CSV column loader (same as the other harness) ---- */
static double *load_column(const char *path, uint64_t *out_n) {
    *out_n = 0;
    FILE *fp = fopen(path, "r");
    if (!fp)
        return NULL;
    char line[8192];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return NULL;
    }
    int target = -1, idx = 0;
    char *save = NULL;
    for (char *t = strtok_r(line, ",\r\n", &save); t;
         t = strtok_r(NULL, ",\r\n", &save)) {
        if (!strcmp(t, "Computation duration (us)") ||
            !strcmp(t, "Kernel duration (us)")) {
            target = idx;
            break;
        }
        idx++;
    }
    if (target < 0) {
        fclose(fp);
        return NULL;
    }
    uint64_t cap = 1024, n = 0;
    double *d = malloc(cap * sizeof(double));
    while (fgets(line, sizeof(line), fp)) {
        int col = 0;
        char *s2 = NULL;
        double v = 0;
        int got = 0;
        for (char *t = strtok_r(line, ",\r\n", &s2); t;
             t = strtok_r(NULL, ",\r\n", &s2)) {
            if (col == target) {
                v = atof(t);
                got = 1;
                break;
            }
            col++;
        }
        if (!got)
            continue;
        if (n == cap) {
            cap *= 2;
            d = realloc(d, cap * sizeof(double));
        }
        d[n++] = v;
    }
    fclose(fp);
    *out_n = n;
    return d;
}


/* SplitMix64: fast PRNG, state is the single uint64_t at `s`. Advances the
 * state by a fixed constant, then scrambles via multiply-xor-shift so
 * consecutive values look independent. Same seed -> same sequence. */
static uint64_t splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* The paper's cost model T(x) (Section 2.2/2.3)
 * paper: T(x) = x * Ci + (#E - x)*pred(x)
 * here: total = cumulative_total + remaining * best_runtime
 * x * Ci      -> cumulative_total (actual time spent, not steps x per-step cost)
 * #E − x      -> remaining        (total_kernel_runs − step_idx)
 * pred(x)     -> best_runtime     (the real best found, not the fitted curve)
 */
static double total_runtime_for_step(double cumulative_total, uint64_t step_idx,
                                     double best_runtime,
                                     uint64_t total_kernel_runs) {
    double remaining = (total_kernel_runs > step_idx)
                           ? (double)(total_kernel_runs - step_idx)
                           : 0.0;
    return cumulative_total + best_runtime * remaining;
}

typedef struct {
    const char *hw;
    uint64_t runs;
    const char *k;
    uint64_t oh;
    double py_decline;
    const char *hist_hw; /* historical HW used to generate this reference row */
} Ref;
/* Reference "Performance decline (%)" values.
 * 680 rows: from the original benchmarking_gemm-reduced_output.csv.
 * 750/1070/2080 rows: from merged_JARDA_vs_KRISTIAN_full.csv (Kristian column),
 * which also adds the extra overhead settings 100 / 1000 / 100000. */
static Ref refs[] = {
    {"680", 10000ULL, "0.0", 10000ULL, 6.673845, "680"},
    {"680", 10000ULL, "0.0", 1000000ULL, 37.520920, "680"},
    {"680", 10000ULL, "0.5", 10000ULL, 6.717256, "680"},
    {"680", 10000ULL, "0.5", 1000000ULL, 37.128403, "680"},
    {"680", 10000ULL, "1.0", 10000ULL, 17.038522, "680"},
    {"680", 10000ULL, "1.0", 1000000ULL, 36.381331, "680"},
    {"680", 10000000ULL, "0.0", 10000ULL, 4.784461, "680"},
    {"680", 10000000ULL, "0.0", 1000000ULL, 6.623326, "680"},
    {"680", 10000000ULL, "0.5", 10000ULL, 0.913113, "680"},
    {"680", 10000000ULL, "0.5", 1000000ULL, 2.991621, "680"},
    {"680", 10000000ULL, "1.0", 10000ULL, 28.236678, "680"},
    {"680", 10000000ULL, "1.0", 1000000ULL, 26.158967, "680"},
    {"750", 10000ULL, "0.0", 100ULL, 16.587014, "1070"},
    {"750", 10000ULL, "0.0", 1000ULL, 16.409663, "1070"},
    {"750", 10000ULL, "0.0", 10000ULL, 14.706002, "1070"},
    {"750", 10000ULL, "0.0", 100000ULL, 11.040693, "1070"},
    {"750", 10000ULL, "0.0", 1000000ULL, 25.312916, "1070"},
    {"750", 10000ULL, "0.5", 100ULL, 5.445235, "1070"},
    {"750", 10000ULL, "0.5", 1000ULL, 6.623712, "1070"},
    {"750", 10000ULL, "0.5", 10000ULL, 10.691242, "1070"},
    {"750", 10000ULL, "0.5", 100000ULL, 10.677386, "1070"},
    {"750", 10000ULL, "0.5", 1000000ULL, 24.666943, "1070"},
    {"750", 10000ULL, "1.0", 100ULL, 22.042021, "1070"},
    {"750", 10000ULL, "1.0", 1000ULL, 21.653146, "1070"},
    {"750", 10000ULL, "1.0", 10000ULL, 18.576328, "1070"},
    {"750", 10000ULL, "1.0", 100000ULL, 10.677386, "1070"},
    {"750", 10000ULL, "1.0", 1000000ULL, 24.666943, "1070"},
    {"750", 1000000ULL, "0.0", 100000ULL, 20.341683, "1070"},
    {"750", 1000000ULL, "0.0", 1000000ULL, 15.943181, "1070"},
    {"750", 1000000ULL, "0.5", 100000ULL, 6.688852, "1070"},
    {"750", 1000000ULL, "0.5", 1000000ULL, 11.736089, "1070"},
    {"750", 1000000ULL, "1.0", 100000ULL, 28.311820, "1070"},
    {"750", 1000000ULL, "1.0", 1000000ULL, 21.011898, "1070"},
    {"750", 10000000ULL, "0.0", 100ULL, 21.834966, "1070"},
    {"750", 10000000ULL, "0.0", 1000ULL, 22.022469, "1070"},
    {"750", 10000000ULL, "0.0", 10000ULL, 21.981411, "1070"},
    {"750", 10000000ULL, "0.0", 1000000ULL, 20.565778, "1070"},
    {"750", 10000000ULL, "0.5", 100ULL, 1.346436, "1070"},
    {"750", 10000000ULL, "0.5", 1000ULL, 1.347873, "1070"},
    {"750", 10000000ULL, "0.5", 10000ULL, 1.429335, "1070"},
    {"750", 10000000ULL, "0.5", 1000000ULL, 6.684293, "1070"},
    {"750", 10000000ULL, "1.0", 100ULL, 31.491458, "1070"},
    {"750", 10000000ULL, "1.0", 1000ULL, 31.487553, "1070"},
    {"750", 10000000ULL, "1.0", 10000ULL, 31.496248, "1070"},
    {"750", 10000000ULL, "1.0", 1000000ULL, 28.454284, "1070"},
    {"1070", 10000ULL, "0.0", 100ULL, 6.711087, "1070"},
    {"1070", 10000ULL, "0.0", 1000ULL, 7.599889, "1070"},
    {"1070", 10000ULL, "0.0", 10000ULL, 9.326852, "1070"},
    {"1070", 10000ULL, "0.0", 100000ULL, 14.902479, "1070"},
    {"1070", 10000ULL, "0.0", 1000000ULL, 127.332489, "1070"},
    {"1070", 10000ULL, "0.5", 100ULL, 7.841139, "1070"},
    {"1070", 10000ULL, "0.5", 1000ULL, 8.930474, "1070"},
    {"1070", 10000ULL, "0.5", 10000ULL, 11.583142, "1070"},
    {"1070", 10000ULL, "0.5", 100000ULL, 14.949595, "1070"},
    {"1070", 10000ULL, "0.5", 1000000ULL, 121.585266, "1070"},
    {"1070", 10000ULL, "1.0", 100ULL, 31.222440, "1070"},
    {"1070", 10000ULL, "1.0", 1000ULL, 28.759448, "1070"},
    {"1070", 10000ULL, "1.0", 10000ULL, 18.210415, "1070"},
    {"1070", 10000ULL, "1.0", 100000ULL, 14.949595, "1070"},
    {"1070", 10000ULL, "1.0", 1000000ULL, 121.585266, "1070"},
    {"1070", 1000000ULL, "0.0", 100000ULL, 6.705876, "1070"},
    {"1070", 1000000ULL, "0.0", 1000000ULL, 9.448730, "1070"},
    {"1070", 1000000ULL, "0.5", 100000ULL, 6.998256, "1070"},
    {"1070", 1000000ULL, "0.5", 1000000ULL, 11.479724, "1070"},
    {"1070", 1000000ULL, "1.0", 100000ULL, 33.007488, "1070"},
    {"1070", 1000000ULL, "1.0", 1000000ULL, 19.182440, "1070"},
    {"1070", 10000000ULL, "0.0", 100ULL, 3.685144, "1070"},
    {"1070", 10000000ULL, "0.0", 1000ULL, 4.204576, "1070"},
    {"1070", 10000000ULL, "0.0", 10000ULL, 4.397003, "1070"},
    {"1070", 10000000ULL, "0.0", 1000000ULL, 6.876644, "1070"},
    {"1070", 10000000ULL, "0.5", 100ULL, 1.212492, "1070"},
    {"1070", 10000000ULL, "0.5", 1000ULL, 1.232419, "1070"},
    {"1070", 10000000ULL, "0.5", 10000ULL, 1.385980, "1070"},
    {"1070", 10000000ULL, "0.5", 1000000ULL, 7.005274, "1070"},
    {"1070", 10000000ULL, "1.0", 100ULL, 41.808577, "1070"},
    {"1070", 10000000ULL, "1.0", 1000ULL, 41.784298, "1070"},
    {"1070", 10000000ULL, "1.0", 10000ULL, 41.639431, "1070"},
    {"1070", 10000000ULL, "1.0", 1000000ULL, 33.070035, "1070"},
    {"2080", 10000ULL, "0.0", 100ULL, 14.132291, "1070"},
    {"2080", 10000ULL, "0.0", 1000ULL, 13.785644, "1070"},
    {"2080", 10000ULL, "0.0", 10000ULL, 11.773454, "1070"},
    {"2080", 10000ULL, "0.0", 100000ULL, 8.048280, "1070"},
    {"2080", 10000ULL, "0.0", 1000000ULL, 27.526621, "1070"},
    {"2080", 10000ULL, "0.5", 100ULL, 6.357583, "1070"},
    {"2080", 10000ULL, "0.5", 1000ULL, 6.903232, "1070"},
    {"2080", 10000ULL, "0.5", 10000ULL, 9.846525, "1070"},
    {"2080", 10000ULL, "0.5", 100000ULL, 8.156045, "1070"},
    {"2080", 10000ULL, "0.5", 1000000ULL, 26.693601, "1070"},
    {"2080", 10000ULL, "1.0", 100ULL, 20.199566, "1070"},
    {"2080", 10000ULL, "1.0", 1000ULL, 19.535357, "1070"},
    {"2080", 10000ULL, "1.0", 10000ULL, 15.650991, "1070"},
    {"2080", 10000ULL, "1.0", 100000ULL, 8.156045, "1070"},
    {"2080", 10000ULL, "1.0", 1000000ULL, 26.693601, "1070"},
    {"2080", 1000000ULL, "0.0", 100000ULL, 17.444143, "1070"},
    {"2080", 1000000ULL, "0.0", 1000000ULL, 12.629180, "1070"},
    {"2080", 1000000ULL, "0.5", 100000ULL, 7.312042, "1070"},
    {"2080", 1000000ULL, "0.5", 1000000ULL, 10.649537, "1070"},
    {"2080", 1000000ULL, "1.0", 100000ULL, 24.905201, "1070"},
    {"2080", 1000000ULL, "1.0", 1000000ULL, 17.363949, "1070"},
    {"2080", 10000000ULL, "0.0", 100ULL, 18.186497, "1070"},
    {"2080", 10000000ULL, "0.0", 1000ULL, 17.991728, "1070"},
    {"2080", 10000000ULL, "0.0", 10000ULL, 18.359469, "1070"},
    {"2080", 10000000ULL, "0.0", 1000000ULL, 17.313064, "1070"},
    {"2080", 10000000ULL, "0.5", 100ULL, 1.331163, "1070"},
    {"2080", 10000000ULL, "0.5", 1000ULL, 1.337845, "1070"},
    {"2080", 10000000ULL, "0.5", 10000ULL, 1.409713, "1070"},
    {"2080", 10000000ULL, "0.5", 1000000ULL, 7.260816, "1070"},
    {"2080", 10000000ULL, "1.0", 100ULL, 27.784365, "1070"},
    {"2080", 10000000ULL, "1.0", 1000ULL, 27.780459, "1070"},
    {"2080", 10000000ULL, "1.0", 10000ULL, 27.756297, "1070"},
    {"2080", 10000000ULL, "1.0", 1000000ULL, 25.006884, "1070"},
};

static TunerMode mode_of(const char *k) {
    if (!strcmp(k, "1.0"))
        return TUNER_MODE_LIVE;
    if (!strcmp(k, "0.0"))
        return TUNER_MODE_HISTORICAL;
    return TUNER_MODE_HYBRID;
}

/* Returns 1 if the key `k` matches the mode `m` (e.g. "1.0" matches "live"),
 * 0 otherwise. */
static int want(const char *k, const char *m) {
    if (!strcmp(m, "all"))
        return 1;
    if (!strcmp(m, "live"))
        return !strcmp(k, "1.0");
    if (!strcmp(m, "hist"))
        return !strcmp(k, "0.0");
    if (!strcmp(m, "hybrid"))
        return !strcmp(k, "0.5");
    return 0;
}

int main(int argc, char **argv) {
    /* Input parsing */
    uint64_t trials = (argc > 1) ? strtoull(argv[1], NULL, 10) : 300;
    const char *mode = (argc > 2) ? argv[2] : "all";
    const char *outpath = (argc > 3) ? argv[3] : "total_runtime_results.txt";
    uint64_t fit_start = 10, hist_tests = 1000;

    /* Results are written to this file AS THEY ARE COMPUTED (each row flushed
     * immediately), so an interrupted run keeps its completed rows and you can
     * watch progress live with: tail -f total_runtime_results.txt */
    FILE *out = fopen(outpath, "w");
    if (!out) {
        fprintf(stderr, "ERROR: cannot open output file '%s'\n", outpath);
        return 1;
    }

    printf("DEPLOYED tuner_api TOTAL-RUNTIME test vs Python decline%%  "
           "(trials=%lu, mode=%s)\n",
           (unsigned long)trials, mode);
    printf("==================================================================="
           "===============\n");
    printf("%-24s %12s %12s %9s\n", "HW/runs/k/oh", "C_decl%", "Py_decl%",
           "diff(pp)");
    printf("-------------------------------------------------------------------"
           "---------------\n");

    fprintf(out,
            "DEPLOYED tuner_api TOTAL-RUNTIME test vs Python decline%%  "
            "(trials=%lu, mode=%s)\n",
            (unsigned long)trials, mode);
    fprintf(out, "==========================================================="
                 "=======================\n");
    fprintf(out, "%-24s %12s %12s %9s\n", "HW/runs/k/oh", "C_decl%", "Py_decl%",
            "diff(pp)");
    fprintf(out, "-----------------------------------------------------------"
                 "-----------------------\n");
    fflush(out);

    int n = (int)(sizeof(refs) / sizeof(*refs));
    double sum_abs = 0;
    int cnt = 0;
    char loaded_hw[8] = "";
    double *data = NULL;
    uint64_t ndata = 0;

    /* one iteration = one configuration from the refs[] table */
    for (int i = 0; i < n; i++) {
        Ref *r = &refs[i];
        if (!want(r->k, mode))
            continue;

        /* Load this GPU's runtimes, but only when the hardware CHANGES. refs[]
        * is sorted by HW, so consecutive rows usually share a file -- this
        * caches it instead of re-reading ~5788 rows per config. */
        if (strcmp(loaded_hw, r->hw) != 0) {
            free(data); /* release the previous GPU's array */
            char path[512];
            snprintf(path, sizeof(path),
                     "raw-data/raw-autotuning-data/gemm-reduced/"
                     "%s-gemm-reduced_output.csv",
                     r->hw);
            data = load_column(path, &ndata);
            if (!data) {
                /* missing file: skip every row for this GPU, don't abort */
                printf("[skip %s: no data]\n", r->hw);
                loaded_hw[0] = '\0';
                continue;
            }
            snprintf(loaded_hw, sizeof(loaded_hw), "%s", r->hw);
        }

        /* How long one simulated run is: you cannot tune for more steps than
        * the app will run (r->runs) or than there are configs (ndata), and
        * 2000 is the project-wide cap (CURVE_LIMIT_MAX). */
        uint64_t curve_limit = (r->runs < ndata) ? r->runs : ndata;
        if (curve_limit > 2000)
            curve_limit = 2000;

        /* build the config; historical modes fit their data inside initiate */
        KernelConfig cfg = {0};
        cfg.struct_size = sizeof(cfg);
        cfg.total_kernel_runs = r->runs;
        cfg.overhead = r->oh;
        cfg.fit_start = fit_start;
        cfg.mode = mode_of(r->k); /* "1.0"/"0.0"/"0.5" -> enum */
        cfg.k = atof(r->k); /* same value as a number */
        cfg.hist_HW = r->hist_hw;
        cfg.file_name = "gemm-reduced_output.csv";
        cfg.hist_number_of_tests = hist_tests;

        /* create once per config (historical fit happens here, once) */
        KernelHandle *h = initiate_kernel(&cfg, NULL, NULL);
        if (!h) {
            printf("[init failed %s/%s]\n", r->hw, r->k);
            continue;
        }

        double sum_extra = 0; /* sum of per-trial extra%, averaged at the end */
        /* Seed differs per config but is FIXED across runs, so results are
         * reproducible. Pass a seed_base argument to draw a different sample. */
        uint64_t rng =
            0xDA7A1234C0FFEEULL ^ ((uint64_t)i << 1); /* per-config seed */

        for (uint64_t t = 0; t < trials; t++) {
            reset_kernel(h);

            /* Pre-generate the run so the estimator and the oracle see the
             * SAME data, then walk it ONCE: feed the estimator until it stops
             * (flag, not break -- the oracle must keep scanning to the end). */
            /* generate the run */
            static double run[2000]; /* kernel times */
            for (uint64_t s = 0; s < curve_limit; s++)
                run[s] = data[splitmix(&rng) % ndata];

            /* single pass: drive the estimator AND compute the oracle */
            double cumulative_total = 0, best = run[0];
            double oracle_total = INFINITY, est_total = 0;
            uint64_t est_stop = curve_limit - 1;
            int stopped = 0; /* has the estimator quit yet? */

            for (uint64_t s = 0; s < curve_limit; s++) {
                /* --- estimator side: only until it stops --- */
                if (!stopped) {
                    push_result(h, run[s], run[s] + (double)r->oh);
                    if (find_number_of_steps_that_should_be_tuning(h) == 0) {
                        est_stop = s;
                        stopped = 1; /* record it, but KEEP LOOPING */
                    }
                }

                /* --- oracle side: always, to the end --- */
                if (run[s] < best)
                    best = run[s];
                cumulative_total +=
                    run[s] + (double)r->oh; /* total spent through step s */
                double tot =
                    total_runtime_for_step(cumulative_total, s, best, r->runs);
                if (tot < oracle_total)
                    oracle_total = tot; /* best possible stop */
                if (s == est_stop)
                    est_total = tot; /* estimator's stop */
            }
            /* extra runtime fraction for this trial */
            if (oracle_total > 0)
                sum_extra += (est_total / oracle_total - 1.0);
        }
        release_kernel(h);

        double c_decl = 100.0 * (sum_extra / (double)trials);
        double diff = c_decl - r->py_decline;
        sum_abs += fabs(diff);
        cnt++;

        char cfgname[40];
        snprintf(cfgname, sizeof(cfgname), "%s/%lu/%s/%lu", r->hw,
                 (unsigned long)r->runs, r->k, (unsigned long)r->oh);
        printf("%-24s %12.3f %12.3f %9.2f\n", cfgname, c_decl, r->py_decline,
               diff);
        fflush(stdout);
        /* write this row NOW (flushed) so completed results survive an
         * interrupted run */
        fprintf(out, "%-24s %12.3f %12.3f %9.2f\n", cfgname, c_decl,
                r->py_decline, diff);
        fflush(out);
    }
    free(data);
    printf("-------------------------------------------------------------------"
           "---------------\n");
    if (cnt)
        printf("mean |C - Py| performance-decline difference: %.2f pp over %d "
               "configs (mode=%s)\n",
               sum_abs / cnt, cnt, mode);
    printf("decline%% = total_runtime(estimator stop)/total_runtime(oracle "
           "stop) - 1\n");

    fprintf(out, "-----------------------------------------------------------"
                 "-----------------------\n");
    if (cnt)
        fprintf(out,
                "mean |C - Py| performance-decline difference: %.2f pp over %d "
                "configs (mode=%s)\n",
                sum_abs / cnt, cnt, mode);
    fprintf(out, "decline%% = total_runtime(estimator stop)/total_runtime("
                 "oracle stop) - 1\n");
    fclose(out);
    printf("\nresults written to: %s\n", outpath);
    return 0;
}
