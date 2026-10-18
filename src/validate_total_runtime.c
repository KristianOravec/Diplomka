/* validate_total_runtime.c -- C engine vs. Jarda's Python on total runtime.
 *
 * For every reference config it runs many simulated tuning sessions through the
 * deployed API (initiate / push / find) and measures how much worse its stop is
 * than the best possible stop:
 *
 *     decline % = total_runtime(library's stop) / total_runtime(oracle's stop) - 1
 *
 * and compares the average with Jarda's published Python value.
 *
 * Usage: ./validate_total_runtime [trials=300] [live|hist|hybrid|all] [outfile] [seed]
 * Run from the repo root (needs raw-data/). Rows are written as they finish.
 */

#include "tuner_api.h"
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIT_START 10
#define HISTORY_TESTS 1000
#define MAX_SESSION_STEPS 2000
#define DEFAULT_SEED 0xDA7A1234C0FFEEULL
#define DATA_DIR "raw-data/raw-autotuning-data/gemm-reduced/"
#define LINE "----------------------------------------------------------------------------------"
#define DOUBLE_LINE "=================================================================================="

typedef struct {
    const char *gpu;
    uint64_t total_runs;
    const char *k;              /* "1.0" live, "0.0" hist, "0.5" hybrid */
    uint64_t overhead_us;
    double python_decline_pct;
    const char *history_gpu;    /* must match the GPU Jarda used for this row */
} PythonReference;

/* Jarda's "Performance decline (%)": 680 rows from benchmarking_gemm-reduced_output.csv
 * (history 680), the rest from his JARDA_benchmarking_..._1070_*.csv (history 1070). */
static const PythonReference REFERENCES[] = {
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
    {"750", 10000ULL, "0.0", 100ULL, 15.057263, "1070"},
    {"750", 10000ULL, "0.0", 1000ULL, 14.825848, "1070"},
    {"750", 10000ULL, "0.0", 10000ULL, 14.003097, "1070"},
    {"750", 10000ULL, "0.0", 100000ULL, 10.616935, "1070"},
    {"750", 10000ULL, "0.0", 1000000ULL, 25.745760, "1070"},
    {"750", 10000ULL, "0.5", 100ULL, 5.505519, "1070"},
    {"750", 10000ULL, "0.5", 1000ULL, 6.213154, "1070"},
    {"750", 10000ULL, "0.5", 10000ULL, 10.027153, "1070"},
    {"750", 10000ULL, "0.5", 100000ULL, 10.276305, "1070"},
    {"750", 10000ULL, "0.5", 1000000ULL, 25.686726, "1070"},
    {"750", 10000ULL, "1.0", 100ULL, 21.500750, "1070"},
    {"750", 10000ULL, "1.0", 1000ULL, 21.064250, "1070"},
    {"750", 10000ULL, "1.0", 10000ULL, 18.621411, "1070"},
    {"750", 10000ULL, "1.0", 100000ULL, 10.459304, "1070"},
    {"750", 10000ULL, "1.0", 1000000ULL, 25.295660, "1070"},
    {"750", 1000000ULL, "0.0", 100000ULL, 19.424744, "1070"},
    {"750", 1000000ULL, "0.0", 1000000ULL, 15.078164, "1070"},
    {"750", 1000000ULL, "0.5", 100000ULL, 6.628098, "1070"},
    {"750", 1000000ULL, "0.5", 1000000ULL, 11.620052, "1070"},
    {"750", 1000000ULL, "1.0", 100000ULL, 28.448460, "1070"},
    {"750", 1000000ULL, "1.0", 1000000ULL, 20.782532, "1070"},
    {"750", 10000000ULL, "0.0", 100ULL, 20.826205, "1070"},
    {"750", 10000000ULL, "0.0", 1000ULL, 20.839105, "1070"},
    {"750", 10000000ULL, "0.0", 10000ULL, 22.303512, "1070"},
    {"750", 10000000ULL, "0.0", 1000000ULL, 19.163148, "1070"},
    {"750", 10000000ULL, "0.5", 100ULL, 1.372768, "1070"},
    {"750", 10000000ULL, "0.5", 1000ULL, 1.374576, "1070"},
    {"750", 10000000ULL, "0.5", 10000ULL, 1.322820, "1070"},
    {"750", 10000000ULL, "0.5", 1000000ULL, 6.959985, "1070"},
    {"750", 10000000ULL, "1.0", 100ULL, 30.350266, "1070"},
    {"750", 10000000ULL, "1.0", 1000ULL, 30.369829, "1070"},
    {"750", 10000000ULL, "1.0", 10000ULL, 31.372473, "1070"},
    {"750", 10000000ULL, "1.0", 1000000ULL, 28.859994, "1070"},
    {"1070", 10000ULL, "0.0", 100ULL, 6.761846, "1070"},
    {"1070", 10000ULL, "0.0", 1000ULL, 7.673422, "1070"},
    {"1070", 10000ULL, "0.0", 10000ULL, 9.120515, "1070"},
    {"1070", 10000ULL, "0.0", 100000ULL, 14.880681, "1070"},
    {"1070", 10000ULL, "0.0", 1000000ULL, 129.639996, "1070"},
    {"1070", 10000ULL, "0.5", 100ULL, 7.891179, "1070"},
    {"1070", 10000ULL, "0.5", 1000ULL, 9.230646, "1070"},
    {"1070", 10000ULL, "0.5", 10000ULL, 11.130658, "1070"},
    {"1070", 10000ULL, "0.5", 100000ULL, 15.099843, "1070"},
    {"1070", 10000ULL, "0.5", 1000000ULL, 125.703712, "1070"},
    {"1070", 10000ULL, "1.0", 100ULL, 30.746368, "1070"},
    {"1070", 10000ULL, "1.0", 1000ULL, 28.175394, "1070"},
    {"1070", 10000ULL, "1.0", 10000ULL, 17.131576, "1070"},
    {"1070", 10000ULL, "1.0", 100000ULL, 15.457193, "1070"},
    {"1070", 10000ULL, "1.0", 1000000ULL, 126.747452, "1070"},
    {"1070", 1000000ULL, "0.0", 100000ULL, 6.862627, "1070"},
    {"1070", 1000000ULL, "0.0", 1000000ULL, 9.436599, "1070"},
    {"1070", 1000000ULL, "0.5", 100000ULL, 6.844971, "1070"},
    {"1070", 1000000ULL, "0.5", 1000000ULL, 10.751286, "1070"},
    {"1070", 1000000ULL, "1.0", 100000ULL, 32.902153, "1070"},
    {"1070", 1000000ULL, "1.0", 1000000ULL, 17.633302, "1070"},
    {"1070", 10000000ULL, "0.0", 100ULL, 3.546976, "1070"},
    {"1070", 10000000ULL, "0.0", 1000ULL, 3.664373, "1070"},
    {"1070", 10000000ULL, "0.0", 10000ULL, 4.156336, "1070"},
    {"1070", 10000000ULL, "0.0", 1000000ULL, 7.070348, "1070"},
    {"1070", 10000000ULL, "0.5", 100ULL, 0.989348, "1070"},
    {"1070", 10000000ULL, "0.5", 1000ULL, 1.041797, "1070"},
    {"1070", 10000000ULL, "0.5", 10000ULL, 1.286002, "1070"},
    {"1070", 10000000ULL, "0.5", 1000000ULL, 7.610359, "1070"},
    {"1070", 10000000ULL, "1.0", 100ULL, 39.420400, "1070"},
    {"1070", 10000000ULL, "1.0", 1000ULL, 39.428022, "1070"},
    {"1070", 10000000ULL, "1.0", 10000ULL, 38.668671, "1070"},
    {"1070", 10000000ULL, "1.0", 1000000ULL, 31.560647, "1070"},
    {"2080", 10000ULL, "0.0", 100ULL, 13.960361, "1070"},
    {"2080", 10000ULL, "0.0", 1000ULL, 14.016053, "1070"},
    {"2080", 10000ULL, "0.0", 10000ULL, 11.235602, "1070"},
    {"2080", 10000ULL, "0.0", 100000ULL, 8.088121, "1070"},
    {"2080", 10000ULL, "0.0", 1000000ULL, 27.999905, "1070"},
    {"2080", 10000ULL, "0.5", 100ULL, 6.293504, "1070"},
    {"2080", 10000ULL, "0.5", 1000ULL, 6.993924, "1070"},
    {"2080", 10000ULL, "0.5", 10000ULL, 9.521710, "1070"},
    {"2080", 10000ULL, "0.5", 100000ULL, 8.349592, "1070"},
    {"2080", 10000ULL, "0.5", 1000000ULL, 27.357280, "1070"},
    {"2080", 10000ULL, "1.0", 100ULL, 20.659765, "1070"},
    {"2080", 10000ULL, "1.0", 1000ULL, 19.876260, "1070"},
    {"2080", 10000ULL, "1.0", 10000ULL, 15.825669, "1070"},
    {"2080", 10000ULL, "1.0", 100000ULL, 8.475940, "1070"},
    {"2080", 10000ULL, "1.0", 1000000ULL, 27.229747, "1070"},
    {"2080", 1000000ULL, "0.0", 100000ULL, 17.225466, "1070"},
    {"2080", 1000000ULL, "0.0", 1000000ULL, 13.028445, "1070"},
    {"2080", 1000000ULL, "0.5", 100000ULL, 7.678625, "1070"},
    {"2080", 1000000ULL, "0.5", 1000000ULL, 10.930987, "1070"},
    {"2080", 1000000ULL, "1.0", 100000ULL, 25.214918, "1070"},
    {"2080", 1000000ULL, "1.0", 1000000ULL, 17.965163, "1070"},
    {"2080", 10000000ULL, "0.0", 100ULL, 18.057677, "1070"},
    {"2080", 10000000ULL, "0.0", 1000ULL, 17.998637, "1070"},
    {"2080", 10000000ULL, "0.0", 10000ULL, 17.278640, "1070"},
    {"2080", 10000000ULL, "0.0", 1000000ULL, 17.819912, "1070"},
    {"2080", 10000000ULL, "0.5", 100ULL, 1.165505, "1070"},
    {"2080", 10000000ULL, "0.5", 1000ULL, 1.253014, "1070"},
    {"2080", 10000000ULL, "0.5", 10000ULL, 1.519993, "1070"},
    {"2080", 10000000ULL, "0.5", 1000000ULL, 8.002066, "1070"},
    {"2080", 10000000ULL, "1.0", 100ULL, 26.869252, "1070"},
    {"2080", 10000000ULL, "1.0", 1000ULL, 27.817720, "1070"},
    {"2080", 10000000ULL, "1.0", 10000ULL, 27.396568, "1070"},
    {"2080", 10000000ULL, "1.0", 1000000ULL, 25.690108, "1070"},
};

#define REFERENCE_COUNT ((int)(sizeof(REFERENCES) / sizeof(REFERENCES[0])))


/* ---- small helpers -------------------------------------------------------- */

static bool equals(const char *a, const char *b) { return strcmp(a, b) == 0; }

/* Print the same text to the terminal and to the results file. */
static void print_both(FILE *file, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    va_start(args, format);
    vfprintf(file, format, args);
    va_end(args);

    fflush(stdout);
    fflush(file);
}

static uint64_t next_random(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static TunerMode mode_from_k(const char *k) {
    if (equals(k, "1.0"))
        return TUNER_MODE_LIVE;
    if (equals(k, "0.0"))
        return TUNER_MODE_HISTORICAL;
    return TUNER_MODE_HYBRID;
}

static bool row_selected(const PythonReference *row, const char *filter) {
    TunerMode mode = mode_from_k(row->k);
    if (equals(filter, "all"))    return true;
    if (equals(filter, "live"))   return mode == TUNER_MODE_LIVE;
    if (equals(filter, "hist"))   return mode == TUNER_MODE_HISTORICAL;
    if (equals(filter, "hybrid")) return mode == TUNER_MODE_HYBRID;
    return false;
}


/* ---- data ----------------------------------------------------------------- */

static int column_index(char *header, const char *name_a, const char *name_b) {
    char *save = NULL;
    int index = 0;
    for (char *name = strtok_r(header, ",\r\n", &save); name; name = strtok_r(NULL, ",\r\n", &save)) {
        if (equals(name, name_a) || equals(name, name_b))
            return index;
        index++;
    }
    return -1;
}

static bool read_column_value(char *line, int column, double *value) {
    char *save = NULL;
    int index = 0;
    for (char *cell = strtok_r(line, ",\r\n", &save); cell; cell = strtok_r(NULL, ",\r\n", &save)) {
        if (index == column) {
            *value = atof(cell);
            return true;
        }
        index++;
    }
    return false;
}

/* Kernel times (us) of every config in a pool CSV. Caller frees. */
static double *load_kernel_times(const char *path, uint64_t *count) {
    *count = 0;
    FILE *file = fopen(path, "r");
    if (!file)
        return NULL;

    char line[8192];
    int column = fgets(line, sizeof line, file)
                     ? column_index(line, "Computation duration (us)", "Kernel duration (us)")
                     : -1;
    if (column < 0) {
        fclose(file);
        return NULL;
    }

    uint64_t capacity = 1024;
    double *times = malloc(capacity * sizeof(double));
    while (fgets(line, sizeof line, file)) {
        double value;
        if (!read_column_value(line, column, &value))
            continue;
        if (*count == capacity) {
            capacity *= 2;
            times = realloc(times, capacity * sizeof(double));
        }
        times[(*count)++] = value;
    }
    fclose(file);
    return times;
}


/* ---- one session ---------------------------------------------------------- */

/* Same cost model as evaluator.c: tuning so far + the remaining runs with the best kernel. */
static double total_runtime(double tuning_so_far, uint64_t step, double best_kernel, uint64_t total_runs) {
    double remaining_runs = (total_runs > step) ? (double)(total_runs - step) : 0.0;
    return tuning_so_far + best_kernel * remaining_runs;
}

/* Feeds one random session to the library and returns how much worse its stop is
 * than the best possible stop (0.05 = 5 % worse). The oracle scans the whole session. */
static double session_extra_runtime(KernelHandle *handle, const double *kernels, uint64_t steps,
                                    const PythonReference *row) {
    double overhead = (double)row->overhead_us;
    double tuning_so_far = 0;
    double best_kernel = kernels[0];
    double oracle_total = INFINITY;
    double library_total = 0;
    uint64_t library_stop = steps - 1;
    bool library_stopped = false;

    reset_kernel(handle);

    for (uint64_t step = 0; step < steps; step++) {
        if (!library_stopped) {
            push_result(handle, kernels[step], kernels[step] + overhead);
            if (find_number_of_steps_that_should_be_tuning(handle) == 0) {
                library_stop = step;
                library_stopped = true;
            }
        }

        if (kernels[step] < best_kernel)
            best_kernel = kernels[step];
        tuning_so_far += kernels[step] + overhead;

        double total = total_runtime(tuning_so_far, step, best_kernel, row->total_runs);
        if (total < oracle_total)
            oracle_total = total;
        if (step == library_stop)
            library_total = total;
    }

    return (oracle_total > 0) ? library_total / oracle_total - 1.0 : 0.0;
}


/* ---- one reference config ------------------------------------------------- */

static KernelHandle *create_handle(const PythonReference *row) {
    KernelConfig config = {0};
    config.total_kernel_runs = row->total_runs;
    config.overhead = row->overhead_us;
    config.fit_start = FIT_START;
    config.mode = mode_from_k(row->k);
    config.k = atof(row->k);
    config.hist_HW = row->history_gpu;
    config.file_name = "gemm-reduced_output.csv";
    config.hist_number_of_tests = HISTORY_TESTS;
    return initiate_kernel(&config, NULL, NULL);
}

/* Average decline % of the library over `trials` random sessions drawn from the pool. */
static double average_decline_pct(KernelHandle *handle, const PythonReference *row,
                                  const double *pool, uint64_t pool_size,
                                  uint64_t trials, uint64_t seed) {
    uint64_t steps = row->total_runs < pool_size ? row->total_runs : pool_size;
    if (steps > MAX_SESSION_STEPS)
        steps = MAX_SESSION_STEPS;

    static double kernels[MAX_SESSION_STEPS];
    double extra_sum = 0;

    for (uint64_t trial = 0; trial < trials; trial++) {
        for (uint64_t step = 0; step < steps; step++)
            kernels[step] = pool[next_random(&seed) % pool_size];
        extra_sum += session_extra_runtime(handle, kernels, steps, row);
    }
    return 100.0 * extra_sum / (double)trials;
}


/* ---- main ----------------------------------------------------------------- */

int main(int argc, char **argv) {
    uint64_t trials = (argc > 1) ? strtoull(argv[1], NULL, 10) : 300;
    const char *filter = (argc > 2) ? argv[2] : "all";
    const char *out_path = (argc > 3) ? argv[3] : "total_runtime_results.txt";
    uint64_t seed_base = (argc > 4) ? strtoull(argv[4], NULL, 10) : DEFAULT_SEED;

    FILE *out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "ERROR: cannot open output file '%s'\n", out_path);
        return 1;
    }

    print_both(out, "DEPLOYED tuner_api TOTAL-RUNTIME test vs Python decline%%  (trials=%lu, mode=%s)\n",
               (unsigned long)trials, filter);
    print_both(out, "%s\n", DOUBLE_LINE);
    print_both(out, "seed_base=%llu\n", (unsigned long long)seed_base);
    print_both(out, "%-24s %12s %12s %9s\n", "HW/runs/k/oh", "C_decl%", "Py_decl%", "diff(pp)");
    print_both(out, "%s\n", LINE);

    char loaded_gpu[8] = "";
    double *pool = NULL;
    uint64_t pool_size = 0;
    double abs_diff_sum = 0;
    int configs_done = 0;

    for (int i = 0; i < REFERENCE_COUNT; i++) {
        const PythonReference *row = &REFERENCES[i];
        if (!row_selected(row, filter))
            continue;

        if (!equals(loaded_gpu, row->gpu)) {
            char path[512];
            snprintf(path, sizeof path, DATA_DIR "%s-gemm-reduced_output.csv", row->gpu);
            free(pool);
            pool = load_kernel_times(path, &pool_size);
            if (!pool) {
                printf("[skip %s: no data]\n", row->gpu);
                loaded_gpu[0] = '\0';
                continue;
            }
            snprintf(loaded_gpu, sizeof loaded_gpu, "%s", row->gpu);
        }

        KernelHandle *handle = create_handle(row);
        if (!handle) {
            printf("[init failed %s/%s]\n", row->gpu, row->k);
            continue;
        }

        uint64_t seed = seed_base ^ ((uint64_t)i << 1);
        double c_decline = average_decline_pct(handle, row, pool, pool_size, trials, seed);
        release_kernel(handle);

        double diff = c_decline - row->python_decline_pct;
        abs_diff_sum += fabs(diff);
        configs_done++;

        char name[40];
        snprintf(name, sizeof name, "%s/%lu/%s/%lu", row->gpu,
                 (unsigned long)row->total_runs, row->k, (unsigned long)row->overhead_us);
        print_both(out, "%-24s %12.3f %12.3f %9.2f\n", name, c_decline, row->python_decline_pct, diff);
    }
    free(pool);

    print_both(out, "%s\n", LINE);
    if (configs_done)
        print_both(out, "mean |C - Py| performance-decline difference: %.2f pp over %d configs (mode=%s)\n",
                   abs_diff_sum / configs_done, configs_done, filter);
    print_both(out, "decline%% = total_runtime(estimator stop)/total_runtime(oracle stop) - 1\n");

    fclose(out);
    printf("\nresults written to: %s\n", out_path);
    return 0;
}
