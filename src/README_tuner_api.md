# Deployed Tuner API (C, for linking into a C++ tuner)

The production library: three functions matching the agreed workflow, backed by
the validated C estimator math.

## The API

```c
KernelHandle* initiate_kernel(const KernelConfig* cfg, const char* debug_name,
                              TunerStatus* out_status);
void          push_result(KernelHandle* h, double runtime_us);
uint64_t      find_number_of_steps_that_should_be_tuning(KernelHandle* h); /* 0 = stop */

void          set_total_runs(KernelHandle* h, uint64_t total_kernel_runs);
void          reset_kernel(KernelHandle* h);
void          release_kernel(KernelHandle* h);
```

Workflow it is designed for:

```c
KernelHandle* h = initiate_kernel(&cfg, "gemm", NULL);
bool tuning = true;
while (application_running()) {
    if (tuning) {
        double rt = perform_one_tuning_step();
        push_result(h, rt);
        if (find_number_of_steps_that_should_be_tuning(h) == 0)
            tuning = false;
    } else {
        run_fastest_kernel();
    }
}
release_kernel(h);
```

## Decisions baked in (from the design discussion)

- **Handle-based** (Decision 1): `initiate_kernel` returns an opaque handle; the
  `debug_name` is a log/debug label only, not a lookup key.
- **Target is C++** (Decision 2): this C library links directly into the C++
  tuner. The Python store from earlier drops to reference/test only.
- **Query is stateless per-call**: each `find_number_of_steps...` call fits the
  curve to the results pushed so far and returns the remaining budget (0 = stop).
  This is what the `== 0` loop check requires.
- **X (total_kernel_runs) = Option 3**: set in the config at init AND updatable
  via `set_total_runs` (X may be established late, or change with input size).
- **Historical data**: same source as the existing C evaluator -- a historical HW
  id + CSV file name; the library calls `get_regression_params` (HISTORICAL/k=0)
  or `history_run` (HYBRID/0<k<1) internally at init.

## Build

```sh
gcc -O2 -fPIC -shared -fopenmp -o libtuner.so \
    tuner_api.c evaluator.c curvefit.c csv.c random.c minicsv.c \
    -lgsl -lgslcblas -llbfgs -lm
```

Link `libtuner.so` (or the .a) plus `tuner_api.h` into the C++ tuner.

## Correctness: what is verified, and the one honest caveat

Two different equivalences, and it matters which one you claim:

1. **Per-call primitive (the one that matters): IDENTICAL.** Each post-warmup
   `find_number_of_steps...` call returns exactly what the paper's per-step
   decision `local_budget_estimation` returns for the same accumulated history.
   Verified: 384/384 calls identical. So the API computes the *correct per-step
   decision*, faithfully reusing the validated math.

2. **Whole-run stop point vs the batch `recommend_tuning_length`: DIFFERS BY
   DESIGN.** Driving the API step-by-step (stop when query==0) matched the batch
   function in 12/16 cases, with gaps up to ~72 steps in the rest. This is
   expected, not a bug: the batch function carries a *running budget* across steps
   (it can coast for several steps after an estimate before stopping), whereas the
   stateless query re-decides fresh each step and stops the instant continuing
   isn't worth it. The agreed workflow's `== 0` polling is inherently stateless,
   so the stateless semantics is the intended one here.

**How to describe this to your tutor:** the library implements the paper's
per-step budget decision exactly; it intentionally does not reproduce the batch
evaluator's cross-step budget carry, because the live polling loop calls for a
fresh per-step decision. If instead you want the live loop to reproduce the batch
walk bit-for-bit, use the stateful variant (`budget_estimator_live` /
`be_observe`), which was separately verified 2000/2000 against the batch function.

## Mode reference

- `TUNER_MODE_LIVE` (k=1): pure live regression. No historical fields needed.
- `TUNER_MODE_HISTORICAL` (k=0): set `hist_HW`, `file_name`,
  `hist_number_of_tests`; library fits historical a,b at init.
- `TUNER_MODE_HYBRID` (0<k<1): same historical source; library computes O_hist.
  Note: O_hist is X-dependent, so if X changes substantially via
  `set_total_runs` in hybrid mode, re-create the handle (see header note).
```
