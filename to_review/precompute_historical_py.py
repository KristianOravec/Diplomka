"""
precompute_historical_py.py -- Python-engine twin of src/precompute_historical.py

Computes exactly the same historical_params table as the C precompute script
(same grid, same 34 rows, same header/row formats, same CLI), but evaluates
the values with the PYTHON reference engine (python/evaluator.py:

    history(HW, file_name, total_kernel_runs, overhead, number_of_tests)
    get_history_regression_parameters(HW, file_name, total_kernel_runs,
                                      fit_start, number_of_tests)

instead of the C engine (libevaluator.so). It exists for A/B comparison of the
two engines on the precomputed cache values -- see compare_cache_engines.py.

IMPORTANT DIFFERENCES VS THE C SCRIPT
  * The C engine is self-deterministic (each Monte-Carlo trial is seeded from
    its own test index + salt inside the C code). The Python engine draws from
    numpy's global RNG instead, so this twin re-seeds numpy BEFORE EACH row
    with np.random.seed(zlib.crc32(key.encode())) where
        key = f"{kind}|{hw}|{FILE_NAME}|{runs}|{third}|{tests}"
    (third = fit_start for REG rows, overhead for OPT rows). That replicates
    the row-seed mechanism the C script used to have, so every row is
    reproducible on re-runs: same key -> same seed -> same Monte-Carlo draws.
    For REG rows `runs` is left OUT of the key -- see the next bullet.
  * REG rows do NOT depend on total_kernel_runs. The fit uses
        curve_limit = min(runs, pool size, CURVE_LIMIT_MAX=2000)
    so every runs >= CURVE_LIMIT_MAX is the same computation. ONE row is
    emitted per (hist_hw, fit_start), stored with total_kernel_runs = -1
    meaning "any runs >= CURVE_LIMIT_MAX". Keying on runs instead would make
    historical_cache.c MISS for any runs value not literally in the grid.
  * evaluator.py resolves its data as "../raw-data/raw-autotuning-data/..."
    RELATIVE TO CWD, so this script chdirs to <repo-root>/python before every
    engine call (NOT the repo root, the C script's chdir target).
  * --out is resolved relative to the directory the script was INVOKED from
    (the C script resolves it relative to the repo root after its chdir; the
    Python engine forces a chdir to python/ instead, so the invocation cwd is
    remembered and used for the output path to keep the same practical
    behaviour of "run from anywhere, write where I pointed you").

USAGE
  python3 testing/precompute_historic/precompute_historical_py.py                     # default grid
  python3 testing/precompute_historic/precompute_historical_py.py --out <path> --tests 300
"""
import argparse
import csv
import itertools
import os
import sys
import time
import zlib

import numpy as np

# ---- the grid to precompute ------------------------------------------------
# Must mirror src/precompute_historical.py EXACTLY so both tables cover the
# same (kind, hist_hw, file_name, total_kernel_runs, fit_start, overhead,
# number_of_tests) keys and can be joined by the comparison script.
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
    ap.add_argument("--out", default="historical_params_py.csv")
    ap.add_argument("--tests", type=int, default=NUMBER_OF_TESTS)
    args = ap.parse_args()

    # Resolve --out against the INVOCATION cwd: the engine needs us to chdir
    # into python/ (DATA_PATH is CWD-relative), so remember where we started.
    out_path = args.out if os.path.isabs(args.out) else \
        os.path.join(os.getcwd(), args.out)

    # ---- locate the Python reference engine -------------------------------
    # Walk UP from this file until we find the directory containing
    # python/evaluator.py. Unlike a fixed number of dirname() calls, this keeps
    # working however deeply the script is nested (testing/,
    # testing/precompute_historic/, ...). evaluator.py imports budget_estimator
    # from the same directory, so that directory must be importable.
    _PY_DIR = None
    _d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(_d, "python")
        if os.path.isfile(os.path.join(cand, "evaluator.py")):
            _PY_DIR = cand
            break
        parent = os.path.dirname(_d)
        if parent == _d:            # reached filesystem root
            break
        _d = parent
    if _PY_DIR is None:
        sys.exit("could not find python/evaluator.py in any parent directory "
                 f"of {os.path.dirname(os.path.abspath(__file__))}")
    if _PY_DIR not in sys.path:
        sys.path.insert(0, _PY_DIR)
    try:
        import evaluator
    except ImportError as e:
        sys.exit(f"cannot import python/evaluator.py: {e}")

    def seed_for_row(kind, hw, runs, third, tests):
        """Seed numpy so each row is reproducible across re-runs.

        The key must contain ONLY parameters that actually affect the result,
        or rows that are mathematically identical get different seeds and
        appear to disagree when they don't.

        REG: the fit uses curve_limit = min(runs, pool, 2000), so any
          runs >= 2000 is the SAME computation -- runs is EXCLUDED here. All
          three runs values then share a seed and produce identical a,b,
          matching what the C engine does.
        OPT: O_hist genuinely depends on runs, so it stays in the key.
        """
        if kind == "REG":
            key = f"{kind}|{hw}|{FILE_NAME}|{third}|{tests}"   # third=fit_start
        else:
            key = f"{kind}|{hw}|{FILE_NAME}|{runs}|{third}|{tests}"
        np.random.seed(zlib.crc32(key.encode()))

    rows = []
    t_start = time.time()

    # ---- REG rows: the frozen curve parameters (a, b) for k=0 ----
    # Depends on fit_start; NOT on overhead, and NOT on total_kernel_runs
    # (see the module docstring). One row per (hw, fit_start), stored with
    # total_kernel_runs = -1 = "any runs >= CURVE_LIMIT_MAX".
    for hw, fs in itertools.product(HIST_HW_LIST, FIT_START_LIST):
        t0 = time.time()
        try:
            seed_for_row("REG", hw, CANONICAL_RUNS, fs, args.tests)
            os.chdir(_PY_DIR)  # evaluator DATA_PATH is CWD-relative
            a, b = evaluator.get_history_regression_parameters(
                hw, FILE_NAME, CANONICAL_RUNS, fs, args.tests)
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
            seed_for_row("OPT", hw, runs, oh, args.tests)
            os.chdir(_PY_DIR)
            o_hist = evaluator.history(hw, FILE_NAME, runs, oh, args.tests)
        except Exception as e:
            print(f"  SKIP OPT {hw}/{runs}/oh={oh}: {e}")
            continue
        rows.append(["OPT", hw, FILE_NAME, runs, -1, oh, args.tests,
                     -1, -1, int(o_hist)])
        print(f"  OPT {hw}/{runs}/oh={oh}: O_hist={o_hist} "
              f"({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "hist_hw", "file_name", "total_kernel_runs",
                    "fit_start", "overhead", "number_of_tests",
                    "a", "b", "optimal_steps"])
        w.writerows(rows)

    print()
    print(f"wrote {len(rows)} rows to {out_path} "
          f"in {(time.time()-t_start)/60:.1f} min")
    print("Python-engine twin of src/precompute_historical.py -- values were")
    print("computed with python/evaluator.py (history /")
    print("get_history_regression_parameters), one crc32-numpy seed per row.")


if __name__ == "__main__":
    main()
