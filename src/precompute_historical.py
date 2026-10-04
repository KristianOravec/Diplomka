"""
precompute_historical.py -- write historical_params.csv for historical_cache.c.

The historical values do not depend on the live run, so they are computed once
here with the C engine (libevaluator.so) instead of at every initiate_kernel():

  REG rows : frozen a, b for hist mode   key: hw, fit_start, tests
             (any X >= CURVE_LIMIT_MAX gives the same a, b -> stored as runs = -1)
  OPT rows : O_hist for hybrid mode      key: hw, X, overhead, tests

Rerun after any change to curvefit.c / evaluator.c (and after 'make').

Usage: python3 src/precompute_historical.py [--out FILE] [--tests N]
"""
import argparse
import csv
import ctypes
import os
import sys


HIST_HW_LIST = ["680", "1070"]
FILE_NAME = "gemm-reduced_output.csv"
NUMBER_OF_TESTS = 1000

# REG rows only
FIT_START_LIST = [10, 15]

# OPT rows only
TOTAL_RUNS_LIST = [10000, 1000000, 10000000]
OVERHEAD_LIST = [100, 1000, 10000, 100000, 1000000]

# X passed to the engine for REG rows. Any X >= 2000 gives the same a, b,
# so the row is stored with ANY_RUNS instead.
REG_RUNS = 10000

# Special CSV values (must match historical_cache.c)
NOT_USED = -1   # column not used by this row kind
ANY_RUNS = -1   # REG row valid for any X >= 2000

CSV_COLUMNS = ["kind", "hist_hw", "file_name", "total_kernel_runs",
               "fit_start", "overhead", "number_of_tests",
               "a", "b", "optimal_steps"]


# C engine (libevaluator.so)
class CurveParams(ctypes.Structure):
    """Mirror of the C struct CurveParams { double a, b, c; }."""
    _fields_ = [("a", ctypes.c_double),
                ("b", ctypes.c_double),
                ("c", ctypes.c_double)]


def find_repo_root():
    """Walk up from this file until a folder has libevaluator.so and raw-data/."""
    folder = os.path.dirname(os.path.abspath(__file__))

    while True:
        has_lib = os.path.isfile(os.path.join(folder, "libevaluator.so"))
        has_data = os.path.isdir(os.path.join(folder, "raw-data"))
        if has_lib and has_data:
            return folder

        parent = os.path.dirname(folder)
        if parent == folder:  # reached "/"
            sys.exit("libevaluator.so not found -- run 'make' in the repo root")
        folder = parent


def load_engine(repo_root):
    """Load libevaluator.so and declare the C functions we call."""
    lib = ctypes.CDLL(os.path.join(repo_root, "libevaluator.so"))

    # uint64_t history_run(const char *hw, const char *file,
    #                      uint64_t runs, uint64_t overhead, uint64_t tests);
    lib.history_run.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
    ]
    lib.history_run.restype = ctypes.c_uint64

    # void get_regression_params(const char *hw, const char *file,
    #                            uint64_t runs, uint64_t fit_start,
    #                            uint64_t tests, CurveParams *out);
    lib.get_regression_params.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(CurveParams),
    ]
    lib.get_regression_params.restype = None

    return lib


def compute_frozen_curve(lib, hw, fit_start, tests):
    """Call C get_regression_params(). Returns (a, b); c is not needed."""
    params = CurveParams()
    lib.get_regression_params(hw.encode(), FILE_NAME.encode(),
                              REG_RUNS, fit_start, tests,
                              ctypes.byref(params))
    return params.a, params.b


def compute_o_hist(lib, hw, runs, overhead, tests):
    """Call C history_run(). Returns O_hist (average oracle stop step)."""
    return lib.history_run(hw.encode(), FILE_NAME.encode(),
                           runs, overhead, tests)


# Rows in output
def build_reg_rows(lib, tests):
    """One REG row per (hw, fit_start)."""
    rows = []

    for hw in HIST_HW_LIST:
        for fit_start in FIT_START_LIST:
            a, b = compute_frozen_curve(lib, hw, fit_start, tests)
            print(f"  REG  hw={hw:<5} fit_start={fit_start:<3}  "
                  f"a={a:.4f}  b={b:.2f}", flush=True)

            rows.append({
                "kind": "REG",
                "hist_hw": hw,
                "file_name": FILE_NAME,
                "total_kernel_runs": ANY_RUNS,
                "fit_start": fit_start,
                "overhead": NOT_USED,
                "number_of_tests": tests,
                "a": f"{a:.10f}",
                "b": f"{b:.10f}",
                "optimal_steps": NOT_USED,
            })

    return rows


def build_opt_rows(lib, tests):
    """One OPT row per (hw, X, overhead)."""
    rows = []

    for hw in HIST_HW_LIST:
        for runs in TOTAL_RUNS_LIST:
            for overhead in OVERHEAD_LIST:
                o_hist = compute_o_hist(lib, hw, runs, overhead, tests)
                print(f"  OPT  hw={hw:<5} X={runs:<9} overhead={overhead:<8}"
                      f"  O_hist={o_hist}", flush=True)

                rows.append({
                    "kind": "OPT",
                    "hist_hw": hw,
                    "file_name": FILE_NAME,
                    "total_kernel_runs": runs,
                    "fit_start": NOT_USED,
                    "overhead": overhead,
                    "number_of_tests": tests,
                    "a": NOT_USED,
                    "b": NOT_USED,
                    "optimal_steps": o_hist,
                })

    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="write historical_params.csv")
    parser.add_argument("--out", default="historical_params.csv")
    parser.add_argument("--tests", type=int, default=NUMBER_OF_TESTS)
    args = parser.parse_args()

    # Resolve --out BEFORE chdir, so the file lands where the script was run.
    out_path = os.path.abspath(args.out)

    # The C engine opens "raw-data/..." relative to the current folder.
    repo_root = find_repo_root()
    os.chdir(repo_root)
    lib = load_engine(repo_root)

    print("REG rows (frozen a, b):")
    reg_rows = build_reg_rows(lib, args.tests)

    print("OPT rows (O_hist):")
    opt_rows = build_opt_rows(lib, args.tests)

    write_csv(out_path, reg_rows + opt_rows)
    print(f"\nwrote {len(reg_rows)} REG + {len(opt_rows)} OPT rows to {out_path}")


if __name__ == "__main__":
    main()
