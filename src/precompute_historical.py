"""
precompute_historical.py -- compute the historical data ONCE, offline.

WHY
  The historical inputs a tuner needs for k=0 and 0<k<1 do NOT depend on the
  live tuning run at all. They depend only on:

    k=0   (frozen curve)   : hist_HW, file_name, fit_start,
                             number_of_tests           -> (a, b)
    0<k<1 (hybrid backstop): hist_HW, file_name, total_kernel_runs, overhead,
                             number_of_tests           -> O_hist

  Note the two differ: the regression fit ignores `overhead`, and O_hist ignores
  `fit_start`. They are cached under separate keys for that reason.

  The frozen curve ALSO ignores total_kernel_runs. The fit uses
      curve_limit = min(total_kernel_runs, pool size, CURVE_LIMIT_MAX=2000)
  so every runs >= CURVE_LIMIT_MAX is the identical computation. REG rows are
  therefore emitted ONCE per (hist_HW, fit_start) and stored with
  total_kernel_runs = -1, meaning "any runs >= CURVE_LIMIT_MAX";
  historical_cache.c matches that row for any such request. Keying REG on runs
  instead would make the cache MISS for every runs value not literally in the
  grid below -- a tuner configured with runs = 50000 would recompute a value
  the table already holds.

  Recomputing these at every initiate_kernel() is pure waste -- the result is
  identical every time. The values now come from the C engine
  (libevaluator.so), so the cached numbers EXACTLY match what tuner_api's
  compute-fallback produces at runtime. The C engine is self-deterministic
  (each Monte-Carlo trial is salted from its own test index), so no Python-side
  seeding is needed.
  For a DEPLOYED tuner it is worse than waste: a kernel would stall for seconds
  or minutes on startup. So we compute them here, write a small table, and let
  the C library load it.

OUTPUT
  historical_params.csv, consumed by historical_cache.c in the tuner library.

USAGE
  Runs from ANY directory -- no PYTHONPATH tricks. The C engine resolves its
  data paths relative to the repo root, so the script chdirs there on startup.

  python3 src/precompute_historical.py              # default grid
  python3 src/precompute_historical.py --out other.csv

REQUIRES
  libevaluator.so in the repo root: run 'make' in the repo root first.
"""
import argparse
import ctypes
import csv
import itertools
import os
import sys
import time

# ---- C engine (libevaluator.so) -------------------------------------------
# Signatures match src/evaluator.h and src/curvefit.h.
#
# Find the repo root by walking UP from this file until we see libevaluator.so
# next to raw-data/. A fixed ".." only works while this script sits in src/;
# the search keeps working wherever it is moved (src/, testing/<subdir>/, ...).
def _find_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if (os.path.isfile(os.path.join(d, "libevaluator.so"))
                and os.path.isdir(os.path.join(d, "raw-data"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:                     # reached the filesystem root
            return None
        d = parent


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    sys.exit("libevaluator.so not found next to raw-data/ in any parent "
             "directory — run 'make' in the repo root first")
_LIB_PATH = os.path.join(_REPO_ROOT, "libevaluator.so")
lib = ctypes.CDLL(_LIB_PATH)


class CurveParams(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double),
                ("b", ctypes.c_double),
                ("c", ctypes.c_double)]


lib.history_run.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                            ctypes.c_uint64, ctypes.c_uint64,
                            ctypes.c_uint64]
lib.history_run.restype = ctypes.c_uint64
lib.get_regression_params.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_uint64, ctypes.c_uint64,
                                      ctypes.c_uint64,
                                      ctypes.POINTER(CurveParams)]
lib.get_regression_params.restype = None

# ---- the grid to precompute ------------------------------------------------
# Must COVER every (hist_HW, runs, fit_start/overhead, tests) combination the
# tuner will ask for. Anything missing falls back to computing at runtime, so a
# gap costs performance but not correctness.
HIST_HW_LIST        = ["680", "1070"]          # historical hardware ids in use
FILE_NAME           = "gemm-reduced_output.csv"
TOTAL_RUNS_LIST     = [10000, 1000000, 10000000]   # OPT rows only
OVERHEAD_LIST       = [100, 1000, 10000, 100000, 1000000]
FIT_START_LIST      = [10, 15]                 # 10 = tuner, 15 = evaluator default
NUMBER_OF_TESTS     = 1000

# Any value >= CURVE_LIMIT_MAX (2000) gives the same REG result, so one call
# per (hw, fit_start) suffices. This is only what we PASS to the engine; the
# row is STORED with total_kernel_runs = -1 to mark it runs-independent.
CANONICAL_RUNS      = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="historical_params.csv")
    ap.add_argument("--tests", type=int, default=NUMBER_OF_TESTS)
    args = ap.parse_args()

    # Resolve --out against the INVOCATION cwd BEFORE chdir'ing, so "run from
    # anywhere, write where I pointed you" still holds.
    out_path = args.out if os.path.isabs(args.out) else \
        os.path.join(os.getcwd(), args.out)

    # C engine paths ("raw-data/raw-autotuning-data/...") are relative to the
    # repo root, so run from there regardless of invocation directory.
    os.chdir(_REPO_ROOT)

    rows = []
    t_start = time.time()

    # ---- REG rows: the frozen curve parameters (a, b) for k=0 ----
    # Depends on fit_start; NOT on overhead, and NOT on total_kernel_runs
    # (see the module docstring). One row per (hw, fit_start), stored with
    # total_kernel_runs = -1 = "any runs >= CURVE_LIMIT_MAX".
    for hw, fs in itertools.product(HIST_HW_LIST, FIT_START_LIST):
        t0 = time.time()
        try:
            params = CurveParams()
            lib.get_regression_params(hw.encode(), FILE_NAME.encode(),
                                      CANONICAL_RUNS, fs, args.tests,
                                      ctypes.byref(params))
            a, b = params.a, params.b
        except Exception as e:
            print(f"  SKIP REG {hw}/fs={fs}: {e}")
            continue
        rows.append(["REG", hw, FILE_NAME, -1, fs, -1, args.tests,
                     f"{a:.10f}", f"{b:.10f}", -1])
        print(f"  REG {hw}/fs={fs} (any runs): a={a:.4f} b={b:.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- OPT rows: the historical optimum O_hist for hybrid ----
    # depends on overhead, NOT on fit_start
    for hw, runs, oh in itertools.product(HIST_HW_LIST, TOTAL_RUNS_LIST,
                                          OVERHEAD_LIST):
        t0 = time.time()
        try:
            o_hist = lib.history_run(hw.encode(), FILE_NAME.encode(), runs,
                                     oh, args.tests)
        except Exception as e:
            print(f"  SKIP OPT {hw}/{runs}/oh={oh}: {e}")
            continue
        rows.append(["OPT", hw, FILE_NAME, runs, -1, oh, args.tests,
                     -1, -1, int(o_hist)])
        print(f"  OPT {hw}/{runs}/oh={oh}: O_hist={o_hist} "
              f"({time.time()-t0:.1f}s)", flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "hist_hw", "file_name", "total_kernel_runs",
                    "fit_start", "overhead", "number_of_tests",
                    "a", "b", "optimal_steps"])
        w.writerows(rows)

    print()
    print(f"wrote {len(rows)} rows to {out_path} "
          f"in {(time.time()-t_start)/60:.1f} min")
    print("Copy it next to the tuner binary (or point TUNER_HISTORICAL_PARAMS")
    print("at it) and the library will load instead of recomputing.")


if __name__ == "__main__":
    main()