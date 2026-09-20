/* =============================================================================
 * validate_tuner_api.c  -- DIRECT validation of the DEPLOYED API vs Python,
 *                          for ALL THREE modes (LIVE k=1, HISTORICAL k=0,
 *                          HYBRID k=0.5).
 *
 * Drives the shipped interface -- initiate_kernel / push_result /
 * find_number_of_steps_that_should_be_tuning -- exactly as the C++ tuner will,
 * over many Monte Carlo trials, and compares the AVERAGED stopping step to
 * Python's "Average estimated TS".
 *
 * USAGE:  ./validate_tuner_api [trials] [mode]
 *   trials : MC trials per config (default 300; 1000 ~ Python)
 *   mode   : live (k=1, default) | hist (k=0) | hybrid (k=0.5) | all
 *
 * EFFICIENCY NOTE (historical modes): k=0 and k=0.5 make initiate_kernel read
 * and FIT historical data at init (get_regression_params / history_run, each
 * 1000 internal trials). Those historical params do NOT change between trials,
 * so we call initiate_kernel ONCE per config and reset_kernel() between trials
 * -- this clears the accumulated live data while keeping the historical fit,
 * avoiding hundreds of redundant refits.
 *
 * CAVEAT: for k=0/k=0.5 the live run is sampled by THIS harness while the
 * historical fit uses the API's own internal RNG -> slightly noisier comparison
 * than LIVE. Expected; not a bug.
 *
 * DESIGN NOTE: the deployed API uses STATELESS per-call query; Python's
 * evaluator uses the BATCH walk. They differ by design, so expect a systematic
 * (documented) offset, especially in LIVE mode. This harness MEASURES it.
 *
 * Build:
 *   gcc -O2 -fopenmp -I<inc> -o validate_tuner_api validate_tuner_api.c \
 *       tuner_api.c evaluator.c curvefit.c csv.c random.c minicsv.c \
 *       -lgsl -lgslcblas -llbfgs -lm
 * Run from the dir containing raw-data/.
 * ============================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "tuner_api.h"

/* ---- load one column from the CSV (self-contained) ---- */
static double* load_column(const char* path, uint64_t* out_n) {
    *out_n = 0;
    FILE* fp = fopen(path, "r");
    if (!fp) return NULL;
    char line[8192];
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return NULL; }
    int target = -1, idx = 0; char* save = NULL;
    for (char* t = strtok_r(line, ",\r\n", &save); t; t = strtok_r(NULL, ",\r\n", &save)) {
        if (!strcmp(t, "Computation duration (us)") || !strcmp(t, "Kernel duration (us)")) { target = idx; break; }
        idx++;
    }
    if (target < 0) { fclose(fp); return NULL; }
    uint64_t cap = 1024, n = 0; double* d = malloc(cap * sizeof(double));
    while (fgets(line, sizeof(line), fp)) {
        int col = 0; char* s2 = NULL; double v = 0; int got = 0;
        for (char* t = strtok_r(line, ",\r\n", &s2); t; t = strtok_r(NULL, ",\r\n", &s2)) {
            if (col == target) { v = atof(t); got = 1; break; } col++;
        }
        if (!got) continue;
        if (n == cap) { cap *= 2; d = realloc(d, cap * sizeof(double)); }
        d[n++] = v;
    }
    fclose(fp); *out_n = n; return d;
}

static uint64_t splitmix(uint64_t* s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

typedef struct {
    const char* hw; uint64_t runs; const char* k; uint64_t oh; double py_est_ts;
} Ref;

/* Python "Average estimated TS" for ALL modes, from extra_results_gemm-reduced. */
static Ref refs[] = {
 {"680",10000ULL,"1.0",10000ULL,22.475},   {"680",10000ULL,"1.0",1000000ULL,16.711},
 {"680",10000ULL,"0.0",10000ULL,63.607},   {"680",10000ULL,"0.0",1000000ULL,16.861},
 {"680",10000ULL,"0.5",10000ULL,60.432},   {"680",10000ULL,"0.5",1000000ULL,16.693},
 {"680",10000000ULL,"1.0",10000ULL,59.119},{"680",10000000ULL,"1.0",1000000ULL,30.749},
 {"680",10000000ULL,"0.0",10000ULL,1093.833},{"680",10000000ULL,"0.0",1000000ULL,270.338},
 {"680",10000000ULL,"0.5",10000ULL,1314.916},{"680",10000000ULL,"0.5",1000000ULL,401.369},
 {"750",10000ULL,"1.0",10000ULL,22.568},   {"750",10000ULL,"1.0",1000000ULL,16.839},
 {"750",10000ULL,"0.0",10000ULL,54.295},   {"750",10000ULL,"0.0",1000000ULL,16.821},
 {"750",10000ULL,"0.5",10000ULL,59.660},   {"750",10000ULL,"0.5",1000000ULL,16.819},
 {"750",10000000ULL,"1.0",10000ULL,69.628},{"750",10000000ULL,"1.0",1000000ULL,36.612},
 {"750",10000000ULL,"0.0",10000ULL,379.125},{"750",10000000ULL,"0.0",1000000ULL,178.806},
 {"750",10000000ULL,"0.5",10000ULL,1386.348},{"750",10000000ULL,"0.5",1000000ULL,465.039},
 {"1070",10000ULL,"1.0",10000ULL,19.614},  {"1070",10000ULL,"1.0",1000000ULL,16.612},
 {"1070",10000ULL,"0.0",10000ULL,113.188}, {"1070",10000ULL,"0.0",1000000ULL,17.023},
 {"1070",10000ULL,"0.5",10000ULL,52.415},  {"1070",10000ULL,"0.5",1000000ULL,16.634},
 {"1070",10000000ULL,"1.0",10000ULL,51.362},{"1070",10000000ULL,"1.0",1000000ULL,26.366},
 {"1070",10000000ULL,"0.0",10000ULL,1999.0},{"1070",10000000ULL,"0.0",1000000ULL,574.460},
 {"1070",10000000ULL,"0.5",10000ULL,1375.436},{"1070",10000000ULL,"0.5",1000000ULL,345.225},
 {"2080",10000ULL,"1.0",10000ULL,21.939},  {"2080",10000ULL,"1.0",1000000ULL,16.724},
 {"2080",10000ULL,"0.0",10000ULL,66.422},  {"2080",10000ULL,"0.0",1000000ULL,16.848},
 {"2080",10000ULL,"0.5",10000ULL,61.927},  {"2080",10000ULL,"0.5",1000000ULL,16.716},
 {"2080",10000000ULL,"1.0",10000ULL,58.297},{"2080",10000000ULL,"1.0",1000000ULL,31.826},
 {"2080",10000000ULL,"0.0",10000ULL,469.194},{"2080",10000000ULL,"0.0",1000000ULL,210.281},
 {"2080",10000000ULL,"0.5",10000ULL,1329.125},{"2080",10000000ULL,"0.5",1000000ULL,419.451},
};

static TunerMode mode_of(const char* k) {
    if (!strcmp(k, "1.0")) return TUNER_MODE_LIVE;
    if (!strcmp(k, "0.0")) return TUNER_MODE_HISTORICAL;
    return TUNER_MODE_HYBRID;
}

static int want(const char* k, const char* mode) {
    if (!strcmp(mode, "all")) return 1;
    if (!strcmp(mode, "live"))   return !strcmp(k, "1.0");
    if (!strcmp(mode, "hist"))   return !strcmp(k, "0.0");
    if (!strcmp(mode, "hybrid")) return !strcmp(k, "0.5");
    return 0;
}

int main(int argc, char** argv) {
    uint64_t trials = (argc > 1) ? strtoull(argv[1], NULL, 10) : 300;
    const char* mode = (argc > 2) ? argv[2] : "live";
    uint64_t fit_start = 10;
    uint64_t hist_tests = 1000;  /* trials for the API's internal historical fit */

    printf("DEPLOYED tuner_api (push/query) vs Python avg estimated TS  (trials=%lu, mode=%s)\n",
           (unsigned long)trials, mode);
    printf("=================================================================\n");
    printf("%-24s %10s %10s %9s\n", "HW/runs/k/oh", "C_estTS", "Py_estTS", "diff");
    printf("-----------------------------------------------------------------\n");

    int n = (int)(sizeof(refs)/sizeof(*refs));
    double sum_abs = 0; int cnt = 0;
    char loaded_hw[8] = ""; double* data = NULL; uint64_t ndata = 0;

    for (int i = 0; i < n; i++) {
        Ref* r = &refs[i];
        if (!want(r->k, mode)) continue;

        if (strcmp(loaded_hw, r->hw) != 0) {
            free(data);
            char path[512];
            snprintf(path, sizeof(path),
                     "raw-data/raw-autotuning-data/gemm-reduced/%s-gemm-reduced_output.csv", r->hw);
            data = load_column(path, &ndata);
            if (!data) { printf("[skip %s: no data at %s]\n", r->hw, path); loaded_hw[0]='\0'; continue; }
            snprintf(loaded_hw, sizeof(loaded_hw), "%s", r->hw);
        }

        uint64_t curve_limit = (r->runs < ndata) ? r->runs : ndata;
        if (curve_limit > 2000) curve_limit = 2000;

        /* Build the config; historical modes read/fit data inside initiate_kernel. */
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

        /* init ONCE per config (does the costly historical fit once), then
         * reset_kernel between trials to reuse the historical params. */
        KernelHandle* h = initiate_kernel(&cfg, NULL, NULL);
        if (!h) { printf("[init failed for %s/%s]\n", r->hw, r->k); continue; }

        double sum_est = 0;
        uint64_t rng = 0xC3D2E1F0B4A59687ULL ^ ((uint64_t)i << 1);
        for (uint64_t t = 0; t < trials; t++) {
            reset_kernel(h);  /* clear live data, keep historical fit */
            uint64_t stop = curve_limit - 1;
            for (uint64_t s = 0; s < curve_limit; s++) {
                double sample = data[splitmix(&rng) % ndata];
                push_result(h, sample, sample + (double)r->oh);  /* total = kernel + measured overhead */
                if (find_number_of_steps_that_should_be_tuning(h) == 0) { stop = s; break; }
            }
            sum_est += (double)(stop + 1);  /* +1: index -> count, like Python */
        }
        release_kernel(h);

        double c_est = sum_est / (double)trials;
        double diff = c_est - r->py_est_ts;
        sum_abs += fabs(diff); cnt++;

        char cfgname[40];
        snprintf(cfgname, sizeof(cfgname), "%s/%lu/%s/%lu",
                 r->hw, (unsigned long)r->runs, r->k, (unsigned long)r->oh);
        printf("%-24s %10.1f %10.1f %9.1f\n", cfgname, c_est, r->py_est_ts, diff);
        fflush(stdout);
    }
    free(data);
    printf("-----------------------------------------------------------------\n");
    if (cnt)
        printf("mean |C - Py| estimated-TS difference: %.1f steps over %d configs (mode=%s)\n",
               sum_abs / cnt, cnt, mode);
    printf("Note: deployed API is stateless per-call; Python evaluator is the batch walk.\n");
    return 0;
}
