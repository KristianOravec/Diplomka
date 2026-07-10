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
 * USAGE:  ./validate_total_runtime [trials] [mode]
 *   trials : MC trials per config (default 300)
 *   mode   : live | hist | hybrid | all   (default all)
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

static uint64_t splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* The cost model, replicated from evaluator.c's total_runtime_for_step():
 *   total = (sum of tuning-step total runtimes) + best_so_far * (remaining
 * runs) cumulative_total = sum over steps 0..step_idx of the total runtime
 * spent best_runtime     = best kernel time found by step_idx
 * (total_kernel_runs - step_idx) = how many of the #E runs remain after tuning
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
} Ref;
/* Python "Performance decline (%)" from benchmarking_gemm-reduced_output.csv.
 */
static Ref refs[] = {
    {"680", 10000ULL, "1.0", 10000ULL, 17.038522},
    {"680", 10000ULL, "1.0", 1000000ULL, 36.381331},
    {"680", 10000ULL, "0.0", 10000ULL, 6.673845},
    {"680", 10000ULL, "0.0", 1000000ULL, 37.520920},
    {"680", 10000ULL, "0.5", 10000ULL, 6.717256},
    {"680", 10000ULL, "0.5", 1000000ULL, 37.128403},
    {"680", 10000000ULL, "1.0", 10000ULL, 28.236678},
    {"680", 10000000ULL, "1.0", 1000000ULL, 26.158967},
    {"680", 10000000ULL, "0.0", 10000ULL, 4.784461},
    {"680", 10000000ULL, "0.0", 1000000ULL, 6.623326},
    {"680", 10000000ULL, "0.5", 10000ULL, 0.913113},
    {"680", 10000000ULL, "0.5", 1000000ULL, 2.991621},
    {"750", 10000ULL, "1.0", 10000ULL, 18.007170},
    {"750", 10000ULL, "1.0", 1000000ULL, 25.339461},
    {"750", 10000ULL, "0.0", 10000ULL, 9.188003},
    {"750", 10000ULL, "0.0", 1000000ULL, 25.452136},
    {"750", 10000ULL, "0.5", 10000ULL, 8.643246},
    {"750", 10000ULL, "0.5", 1000000ULL, 25.085750},
    {"750", 10000000ULL, "1.0", 10000ULL, 30.671944},
    {"750", 10000000ULL, "1.0", 1000000ULL, 28.219934},
    {"750", 10000000ULL, "0.0", 10000ULL, 11.103594},
    {"750", 10000000ULL, "0.0", 1000000ULL, 10.807085},
    {"750", 10000000ULL, "0.5", 10000ULL, 1.435452},
    {"750", 10000000ULL, "0.5", 1000000ULL, 4.199739},
    {"1070", 10000ULL, "1.0", 10000ULL, 17.551271},
    {"1070", 10000ULL, "1.0", 1000000ULL, 127.467788},
    {"1070", 10000ULL, "0.0", 10000ULL, 13.618893},
    {"1070", 10000ULL, "0.0", 1000000ULL, 127.480775},
    {"1070", 10000ULL, "0.5", 10000ULL, 10.176324},
    {"1070", 10000ULL, "0.5", 1000000ULL, 128.016067},
    {"1070", 10000000ULL, "1.0", 10000ULL, 40.405348},
    {"1070", 10000000ULL, "1.0", 1000000ULL, 33.004625},
    {"1070", 10000000ULL, "0.0", 10000ULL, 0.253486},
    {"1070", 10000000ULL, "0.0", 1000000ULL, 7.067604},
    {"1070", 10000000ULL, "0.5", 10000ULL, 1.186747},
    {"1070", 10000000ULL, "0.5", 1000000ULL, 5.101135},
    {"2080", 10000ULL, "1.0", 10000ULL, 15.815015},
    {"2080", 10000ULL, "1.0", 1000000ULL, 26.972473},
    {"2080", 10000ULL, "0.0", 10000ULL, 7.119833},
    {"2080", 10000ULL, "0.0", 1000000ULL, 28.495403},
    {"2080", 10000ULL, "0.5", 10000ULL, 8.106003},
    {"2080", 10000ULL, "0.5", 1000000ULL, 27.455944},
    {"2080", 10000000ULL, "1.0", 10000ULL, 26.602315},
    {"2080", 10000000ULL, "1.0", 1000000ULL, 24.751806},
    {"2080", 10000000ULL, "0.0", 10000ULL, 7.760319},
    {"2080", 10000000ULL, "0.0", 1000000ULL, 8.848318},
    {"2080", 10000000ULL, "0.5", 10000ULL, 1.328840},
    {"2080", 10000000ULL, "0.5", 1000000ULL, 5.360631},
};

static TunerMode mode_of(const char *k) {
    if (!strcmp(k, "1.0"))
        return TUNER_MODE_LIVE;
    if (!strcmp(k, "0.0"))
        return TUNER_MODE_HISTORICAL;
    return TUNER_MODE_HYBRID;
}
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
    uint64_t trials = (argc > 1) ? strtoull(argv[1], NULL, 10) : 300;
    const char *mode = (argc > 2) ? argv[2] : "all";
    uint64_t fit_start = 10, hist_tests = 1000;

    printf("DEPLOYED tuner_api TOTAL-RUNTIME test vs Python decline%%  "
           "(trials=%lu, mode=%s)\n",
           (unsigned long)trials, mode);
    printf("==================================================================="
           "===============\n");
    printf("%-24s %12s %12s %9s\n", "HW/runs/k/oh", "C_decl%", "Py_decl%",
           "diff(pp)");
    printf("-------------------------------------------------------------------"
           "---------------\n");

    int n = (int)(sizeof(refs) / sizeof(*refs));
    double sum_abs = 0;
    int cnt = 0;
    char loaded_hw[8] = "";
    double *data = NULL;
    uint64_t ndata = 0;

    for (int i = 0; i < n; i++) {
        Ref *r = &refs[i];
        if (!want(r->k, mode))
            continue;

        if (strcmp(loaded_hw, r->hw) != 0) {
            free(data);
            char path[512];
            snprintf(path, sizeof(path),
                     "raw-data/raw-autotuning-data/gemm-reduced/"
                     "%s-gemm-reduced_output.csv",
                     r->hw);
            data = load_column(path, &ndata);
            if (!data) {
                printf("[skip %s: no data]\n", r->hw);
                loaded_hw[0] = '\0';
                continue;
            }
            snprintf(loaded_hw, sizeof(loaded_hw), "%s", r->hw);
        }

        uint64_t curve_limit = (r->runs < ndata) ? r->runs : ndata;
        if (curve_limit > 2000)
            curve_limit = 2000;

        /* build the config; historical modes fit their data inside initiate */
        KernelConfig cfg = {0};
        cfg.struct_size = sizeof(cfg);
        cfg.total_kernel_runs = r->runs;
        cfg.overhead = r->oh;
        cfg.fit_start = fit_start;
        cfg.mode = mode_of(r->k);
        cfg.k = atof(r->k);
        cfg.hist_HW = "680";
        cfg.file_name = "gemm-reduced_output.csv";
        cfg.hist_number_of_tests = hist_tests;

        /* create once per config (historical fit happens here, once) */
        KernelHandle *h = initiate_kernel(&cfg, NULL, NULL);
        if (!h) {
            printf("[init failed %s/%s]\n", r->hw, r->k);
            continue;
        }

        double sum_extra = 0; /* sum of per-trial extra% */
        uint64_t rng =
            0xDA7A1234C0FFEEULL ^ ((uint64_t)i << 1); /* per-config seed */

        for (uint64_t t = 0; t < trials; t++) {
            reset_kernel(h);

            /* We must replay the SAME sampled run twice in spirit: once to
             * drive the estimator (to find its stop step), and to compute total
             * runtime at every step so we can also get the oracle minimum. So
             * we pre-generate the run, then (a) feed it to the API and (b) scan
             * it for the oracle. */
            /* generate the run */
            static double run[2000]; /* kernel times */
            for (uint64_t s = 0; s < curve_limit; s++)
                run[s] = data[splitmix(&rng) % ndata];

            /* (a) drive the estimator to its stop step */
            uint64_t est_stop = curve_limit - 1;
            for (uint64_t s = 0; s < curve_limit; s++) {
                push_result(h, run[s], run[s] + (double)r->oh);
                if (find_number_of_steps_that_should_be_tuning(h) == 0) {
                    est_stop = s;
                    break;
                }
            }

            /* (b) scan the same run for cumulative cost, best-so-far, and the
             * oracle (minimum total runtime over all stop points). */
            double cumulative_total = 0, best = run[0];
            double oracle_total = INFINITY, est_total = 0;
            for (uint64_t s = 0; s < curve_limit; s++) {
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
    return 0;
}
