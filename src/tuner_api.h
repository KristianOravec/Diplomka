/* =============================================================================
 * tuner_api.h  -- the deployed C library interface (links into a C++ tuner)
 *
 * Three operations, matching the agreed workflow:
 *
 *   initiate_kernel(...)                          -> handle
 *   push_result(handle, runtime)                  -> void
 *   find_number_of_steps_that_should_be_tuning(h) -> remaining steps; 0 = STOP
 *
 * Plus lifecycle (release) and an X setter (set_total_runs), per the design
 * discussion: total_kernel_runs (X) may not be known at init, or may change when
 * the input size changes, so it can be set/updated after creation.
 *
 * Intended use (C++ tuner):
 *
 *   KernelHandle* h = initiate_kernel(&cfg, "gemm");
 *   bool tuning = true;
 *   while (application_running()) {
 *       if (tuning) {
 *           double rt = perform_one_tuning_step();   // benchmark a config
 *           push_result(h, rt);
 *           if (find_number_of_steps_that_should_be_tuning(h) == 0)
 *               tuning = false;
 *       } else {
 *           run_fastest_kernel();
 *       }
 *   }
 *   release_kernel(h);
 *
 * QUERY SEMANTICS: stateless per-call. Each call fits the regression curve to the
 * results pushed so far and returns how many MORE tuning steps are worth doing
 * (0 = stop now). It does not assume knowledge of future samples. This matches
 * the `== 0` stop check in the loop above.
 *
 * HISTORICAL DATA: supplied the same way as the existing C evaluator -- a
 * historical hardware id + file name read from disk at init. The library fits
 * the historical curve (get_regression_params) and/or computes the historical
 * optimum (history_run) internally, exactly as evaluator_run does.
 *
 * THREADING: one handle == one kernel == one thread. Handles are not internally
 * synchronized; use one per concurrent tuning context.
 * ============================================================================= */
#ifndef TUNER_API_H
#define TUNER_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KernelHandle KernelHandle;

typedef enum {
    TUNER_OK = 0,
    TUNER_ERR_ALLOC = -1,
    TUNER_ERR_INVALID_ARG = -2
} TunerStatus;

/* Sentinel for "no historical curve params" (matches NO_HISTORICAL_DATA). */
#define TUNER_NO_HISTORICAL (-1.0)

/* Estimator mode (the paper's k):
 *   TUNER_MODE_LIVE       (k=1.0) : pure live regression, no history
 *   TUNER_MODE_HISTORICAL (k=0.0) : use historical curve params hist_a, hist_b
 *   TUNER_MODE_HYBRID     (0<k<1) : live regression + historical backstop O_hist */
typedef enum {
    TUNER_MODE_LIVE = 0,
    TUNER_MODE_HISTORICAL = 1,
    TUNER_MODE_HYBRID = 2
} TunerMode;

/* Configuration passed to initiate_kernel.
 * struct_size MUST be set to sizeof(KernelConfig) for ABI safety. */
typedef struct {
    uint64_t struct_size;

    /* core estimator inputs */
    uint64_t total_kernel_runs;  /* X / #E; may be 0 here and set later */
    uint64_t overhead;           /* per-step tuning overhead */
    uint64_t fit_start;          /* warmup before curve fitting (e.g. 10) */

    TunerMode mode;              /* which estimator variant */
    double    k;                 /* exact k; for HYBRID any value in (0,1) */

    /* HISTORICAL DATA SOURCE (read from disk at init, like evaluator.c).
     * Used only when mode is HISTORICAL or HYBRID. Leave hist_file_name NULL for
     * LIVE mode. The library calls get_regression_params / history_run itself. */
    const char* hist_HW;         /* historical hardware id, e.g. "680" */
    const char* file_name;       /* e.g. "gemm-reduced_output.csv" */
    uint64_t    hist_number_of_tests; /* Monte Carlo trials for historical fit */
} KernelConfig;

/* ---- the three workflow functions ---- */

/* Create a kernel tuning context. `debug_name` is for logging only (may be NULL).
 * Reads/fits historical data now if the mode requires it. Returns NULL on error;
 * out_status (optional) receives a TunerStatus. */
KernelHandle* initiate_kernel(const KernelConfig* cfg, const char* debug_name,
                              TunerStatus* out_status);

/* Record one benchmarked configuration's result for this tuning step:
 *   kernel_time_us  -- the configuration's own runtime (what you FOUND); used to
 *                      update best-so-far and to fit the regression curve.
 *   total_time_us   -- the full cost of trying it this step (what HAPPENED) =
 *                      kernel runtime + measured overhead (compilation, searcher,
 *                      etc.). Accumulates into the tuning cost.
 * Pushing both lets overhead be MEASURED per step rather than assumed constant.
 * (total_time_us should be >= kernel_time_us; the difference is this step's
 * overhead.) */
void push_result(KernelHandle* h, double kernel_time_us, double total_time_us);

/* How many MORE tuning steps are worth performing, given everything pushed so
 * far. Returns 0 to mean "stop tuning now". Stateless: recomputed each call. */
uint64_t find_number_of_steps_that_should_be_tuning(KernelHandle* h);

/* ---- lifecycle / configuration ---- */

/* Set or update X (total_kernel_runs) after creation -- e.g. once the
 * application establishes the expected number of executions, or when the input
 * size changes. See note below about hybrid mode. */
void set_total_runs(KernelHandle* h, uint64_t total_kernel_runs);

/* Reset accumulated results (keeps config + historical data) to reuse the handle
 * for a fresh tuning run. */
void reset_kernel(KernelHandle* h);

/* Destroy a handle and free its resources. Safe on NULL. */
void release_kernel(KernelHandle* h);

/* ---- introspection (optional, for logging/debug) ---- */
uint64_t tuner_steps_taken(const KernelHandle* h);   /* samples pushed so far */
double   tuner_best_so_far(const KernelHandle* h);   /* best runtime seen */
const char* tuner_debug_name(const KernelHandle* h); /* the debug label, or "" */

/* NOTE ON set_total_runs + HYBRID MODE:
 * The historical backstop O_hist is itself X-dependent (the paper's 4000x4000 ->
 * 2000x2000 example). For LIVE and HISTORICAL modes, changing X is fully handled
 * -- the next query simply re-minimizes T(x) with the new X. For HYBRID mode, a
 * changed X leaves O_hist stale; if X changes substantially in hybrid mode,
 * re-create the handle (or call reset) so O_hist is recomputed for the new X. */

#ifdef __cplusplus
}
#endif
#endif /* TUNER_API_H */
