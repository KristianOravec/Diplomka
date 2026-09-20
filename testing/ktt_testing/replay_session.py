#!/usr/bin/env python3
"""
replay_session.py -- prove the ClTuneGemm integration is wired correctly by
replaying a REAL session through the OFFLINE estimator.

WHY THIS IS THE STRONGEST INTEGRATION TEST
  Every other check compares distributions: the simulation predicts a median
  stop of 19, the real run stopped at 18, close enough. That is evidence, but
  it is statistical -- a broken integration could still land in the right
  ballpark by luck.

  This test is deterministic. tuning_steps_*.csv records the EXACT sequence of
  kernel times the deployed API saw, step by step. Feeding that same sequence
  into the batch estimator asks a precise question:

      given identical input, does the library stop at the same step
      inside ClTuneGemm as it does outside it?

  If yes, the integration is correct: the C++ side is feeding the right values
  in the right order and reading the stop decision correctly. If no, the gap
  localises the bug -- wrong column pushed, wrong units, an off-by-one in the
  loop, or a config field not reaching the handle.

WHAT IT CANNOT PROVE
  The deployed API receives a MEASURED total per step (kernel + the real
  overhead for that step), while the batch estimator takes ONE constant
  overhead. So the two cost models differ slightly by construction. Expect a
  match within a step or two rather than always exact; a large gap is the
  signal worth chasing. Pass --overhead to control which constant is used
  (default: the session's own mean, excluding step 1).

USAGE
  python3 replay_session.py tuning_steps_10000_hybrid_hist1070_rand_oh31742.csv \\
      --x 10000 --mode hybrid --hist-hw 1070

  # override the constant overhead, or pin O_hist directly
  python3 replay_session.py <csv> --x 10000 --mode hybrid --hist-hw 1070 \\
      --overhead 31742
  python3 replay_session.py <csv> --x 10000 --mode hybrid --o-hist 12

Run from the directory containing python/ (for the engine) and raw-data/.
"""

import argparse
import csv
import ctypes
import os
import statistics
import sys


# --------------------------------------------------------------------------
# locate the C engine: walk UP until we find libevaluator.so beside raw-data/
# --------------------------------------------------------------------------
def find_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if (os.path.isfile(os.path.join(d, "libevaluator.so"))
                and os.path.isdir(os.path.join(d, "raw-data"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def load_engine(root):
    lib = ctypes.CDLL(os.path.join(root, "libevaluator.so"))
    lib.recommend_tuning_length.restype = ctypes.c_uint64
    lib.recommend_tuning_length.argtypes = [
        ctypes.c_uint64,                    # default_tuning_steps (O_hist)
        ctypes.POINTER(ctypes.c_double),    # tuning_run
        ctypes.c_uint64,                    # tuning_run_len
        ctypes.c_uint64,                    # total_kernel_runs (X)
        ctypes.c_double,                    # regression_weight (k)
        ctypes.c_uint64,                    # fit_start
        ctypes.c_uint64,                    # overhead
        ctypes.c_double,                    # hist_a
        ctypes.c_double,                    # hist_b
    ]
    lib.history_run.restype = ctypes.c_uint64
    lib.history_run.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_uint64, ctypes.c_uint64,
                                ctypes.c_uint64]

    class CurveParams(ctypes.Structure):
        _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double),
                    ("c", ctypes.c_double)]

    lib.get_regression_params.restype = None
    lib.get_regression_params.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                          ctypes.c_uint64, ctypes.c_uint64,
                                          ctypes.c_uint64,
                                          ctypes.POINTER(CurveParams)]
    return lib, CurveParams


def session_cost(kernel, totals, stop_idx, X):
    """Total application runtime if we stop after (stop_idx + 1) steps.

    Uses the session's MEASURED per-step totals, so this is the real cost the
    application paid -- not a model of it.
    """
    n = stop_idx + 1
    return sum(totals[:n]) + (X - n) * min(kernel[:n]) if X > n else sum(totals[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="a tuning_steps_*.csv written by ClTuneGemm")
    ap.add_argument("--x", type=int, default=10000, help="total_kernel_runs")
    ap.add_argument("--mode", default="hybrid",
                    choices=["live", "hybrid", "hist"])
    ap.add_argument("--hist-hw", default="1070")
    ap.add_argument("--file-name", default="gemm-reduced_output.csv")
    ap.add_argument("--fit-start", type=int, default=10)
    ap.add_argument("--hist-tests", type=int, default=1000)
    ap.add_argument("--overhead", type=int, default=None,
                    help="constant overhead for the batch model "
                         "(default: session mean, excluding step 1)")
    ap.add_argument("--o-hist", type=int, default=None,
                    help="pin O_hist instead of recomputing it")
    args = ap.parse_args()

    root = find_repo_root()
    if root is None:
        sys.exit("libevaluator.so not found beside raw-data/ in any parent dir")
    lib, CurveParams = load_engine(root)

    rows = list(csv.DictReader(open(args.csv)))
    kernel = [float(r["kernel_us"]) for r in rows]
    ovh = [float(r["overhead_us"]) for r in rows]
    totals = [float(r["total_us"]) for r in rows]
    live_stop = len(rows)                     # 1-based: the session's last step

    # sanity: the file must not carry the overhead=0 bug
    bad = [i + 1 for i, (k, o, t) in enumerate(zip(kernel, ovh, totals))
           if abs(t - (k + o)) > 1.0]
    if bad:
        print(f"WARNING: total != kernel + overhead at steps {bad[:5]} -- this "
              f"session predates the overhead fix and its costs are wrong.\n")

    # Step 1 carries a one-off data-movement cost, so it is excluded from the
    # mean: it is not representative of a steady-state step.
    oh_const = args.overhead if args.overhead is not None \
        else int(round(statistics.mean(ovh[1:]) if len(ovh) > 1 else ovh[0]))

    os.chdir(root)      # engine data paths are relative to the repo root

    k_val = {"live": 1.0, "hybrid": 0.5, "hist": 0.0}[args.mode]
    dts, hist_a, hist_b = 0, -1.0, -1.0
    if args.mode == "hybrid":
        dts = args.o_hist if args.o_hist is not None else lib.history_run(
            args.hist_hw.encode(), args.file_name.encode(), args.x, oh_const,
            args.hist_tests)
    elif args.mode == "hist":
        cp = CurveParams()
        lib.get_regression_params(args.hist_hw.encode(),
                                  args.file_name.encode(), args.x,
                                  args.fit_start, args.hist_tests,
                                  ctypes.byref(cp))
        hist_a, hist_b = cp.a, cp.b

    # ---- replay: the SAME sequence, through the batch estimator ----
    arr = (ctypes.c_double * len(kernel))(*kernel)
    est_idx = lib.recommend_tuning_length(dts, arr, len(kernel), args.x, k_val,
                                          args.fit_start, oh_const,
                                          hist_a, hist_b)
    batch_stop = est_idx + 1                  # 0-based index -> step count

    # ---- oracle over the same stream, using the MEASURED totals ----
    best, cum, orc, ostep = float("inf"), 0.0, float("inf"), 0
    for i, (kk, tt) in enumerate(zip(kernel, totals)):
        best = min(best, kk)
        cum += tt
        rem = args.x - (i + 1) if args.x > (i + 1) else 0
        T = cum + best * rem
        if T < orc:
            orc, ostep = T, i + 1

    W = 74
    def rule(ch="="):
        print("+" + ch * (W - 2) + "+")
    def row(t=""):
        print("| " + t.ljust(W - 4) + " |")

    rule()
    row(f"SESSION REPLAY -- {os.path.basename(args.csv)}")
    print("+" + "-" * (W - 2) + "+")
    row(f"X={args.x}   mode={args.mode}   hist_hw={args.hist_hw}")
    row(f"steps recorded={len(rows)}   overhead used={oh_const} us"
        + (f"   O_hist={dts}" if args.mode == "hybrid" else ""))
    rule()
    row()
    row("[1] INTEGRATION CHECK  (same input, both sides of the library)")
    row(f"      live  (deployed API, inside ClTuneGemm) stopped at {live_stop}")
    row(f"      batch (offline estimator, same stream)  stopped at {batch_stop}")
    gap = batch_stop - live_stop
    row(f"      gap = {gap:+d} steps")
    row()
    if abs(gap) <= 2:
        row("      VERDICT: MATCH -- the integration feeds the library the")
        row("      right values in the right order. A gap of 0-2 steps is")
        row("      expected: the deployed API uses the MEASURED per-step")
        row("      total, the batch model one CONSTANT overhead.")
    else:
        row("      VERDICT: MISMATCH -- larger than the constant-overhead")
        row("      approximation explains. Check: is kernel_us the right")
        row("      column, are units microseconds, is X the same, and does")
        row("      the C++ loop push every step exactly once?")
    row()
    row("[2] WHAT THE SESSION ACTUALLY COST")
    live_cost = session_cost(kernel, totals, live_stop - 1, args.x)
    row(f"      stopped at {live_stop}: {live_cost/1e6:.3f} s")
    row(f"      oracle  at {ostep}: {orc/1e6:.3f} s")
    row(f"      decline = {100*(live_cost/orc - 1):+.2f} %")
    row()
    rule()


if __name__ == "__main__":
    main()
