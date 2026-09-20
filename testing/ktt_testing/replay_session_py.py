#!/usr/bin/env python3
"""
replay_session_py.py -- Python-engine twin of replay_session.py.

Replays a REAL ClTuneGemm session CSV through the OFFLINE estimator and
compares stop steps -- but the stop decision comes from the PYTHON reference
engine (python/evaluator.py + python/budget_estimator.py) instead of the C
library (libevaluator.so). Same CLI, same report structure, same purpose;
see replay_session.py's docstring for the integration-test rationale.

WHAT DIFFERS FROM THE C REPLAY
  * No ctypes, no libevaluator.so, no repo-root-by-libevaluator walk-up.
    Instead the script walks UP from its own directory until it finds the
    directory containing python/evaluator.py, inserts that directory into
    sys.path, and chdirs into it before every historical engine call.
  * hybrid: O_hist = history(hist_hw, file_name, X, overhead, tests),
    recomputed per invocation unless --o-hist pins it. numpy is seeded with
    zlib.crc32(key) before the call, key = OPT|hw|file|X|overhead|tests.
  * hist: (a, b) = get_history_regression_parameters(hist_hw, file_name, X,
    fit_start, tests), seeded with key = REG|hw|file|fit_start|tests
    (runs excluded -- the fit is runs-independent beyond CURVE_LIMIT_MAX).
  * live: dts = -1. The PYTHON sentinel for "no history" is -1, NOT 0 like
    the C engine; blending engages only when dts >= 0.
  * Batch call: est = tuning_length_recommendation(dts, kernel_us, hist_hw,
    file_name, X, k, fit_start, overhead, a, b); batch_stop = est + 1.
    The engine itself is deterministic (no RNG), so no seeding there.

USAGE
  python3 replay_session_py.py tuning_steps_10000_hybrid_hist1070_rand_oh31742.csv \\
      --x 10000 --mode hybrid --hist-hw 1070

  # override the constant overhead, or pin O_hist directly
  python3 replay_session_py.py <csv> --x 10000 --mode hybrid --hist-hw 1070 \\
      --overhead 31742
  python3 replay_session_py.py <csv> --x 10000 --mode hybrid --o-hist 12

Run from anywhere: the python/ engine directory is located by walking UP
from this script's location (evaluator.py resolves raw-data/ relative to
its own CWD, which is why we chdir there -- never to the repo root).
"""

import argparse
import csv
import os
import statistics
import sys
import zlib

import numpy as np


# --------------------------------------------------------------------------
# locate the Python engine: walk UP until we find python/evaluator.py
# --------------------------------------------------------------------------
def find_python_dir():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "python", "evaluator.py")):
            return os.path.join(d, "python")
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def seed_for(kind, hw, file_name, runs, third, tests):
    """Seed numpy so the historical call is reproducible across re-runs.

    Mirrors precompute_historical_py.py: REG excludes runs (the fit is only
    runs-independent beyond CURVE_LIMIT_MAX=2000, and the key must contain
    ONLY parameters that actually affect the result), OPT includes runs
    (O_hist genuinely depends on runs).
    """
    if kind == "REG":
        key = f"{kind}|{hw}|{file_name}|{third}|{tests}"   # third=fit_start
    else:
        key = f"{kind}|{hw}|{file_name}|{runs}|{third}|{tests}"
    np.random.seed(zlib.crc32(key.encode()))


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

    py_dir = find_python_dir()
    if py_dir is None:
        sys.exit("python/evaluator.py not found in any parent directory of "
                 f"{os.path.dirname(os.path.abspath(__file__))}")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    import evaluator
    import budget_estimator

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

    k_val = {"live": 1.0, "hybrid": 0.5, "hist": 0.0}[args.mode]
    dts = -1                                  # PYTHON sentinel: no history
    hist_a, hist_b = -1.0, -1.0
    if args.mode == "hybrid":
        if args.o_hist is not None:
            dts = args.o_hist
        else:
            seed_for("OPT", args.hist_hw, args.file_name, args.x, oh_const,
                     args.hist_tests)
            os.chdir(py_dir)    # evaluator DATA_PATH is CWD-relative
            dts = int(evaluator.history(args.hist_hw, args.file_name, args.x,
                                        oh_const, args.hist_tests))
    elif args.mode == "hist":
        seed_for("REG", args.hist_hw, args.file_name, args.x, args.fit_start,
                 args.hist_tests)
        os.chdir(py_dir)        # evaluator DATA_PATH is CWD-relative
        hist_a, hist_b = evaluator.get_history_regression_parameters(
            args.hist_hw, args.file_name, args.x, args.fit_start,
            args.hist_tests)

    # ---- replay: the SAME sequence, through the batch estimator ----
    est_idx = budget_estimator.tuning_length_recommendation(
        dts, kernel, args.hist_hw, args.file_name, args.x, k_val,
        args.fit_start, oh_const, hist_a, hist_b)
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
    row(f"SESSION REPLAY (PYTHON ENGINE) -- {os.path.basename(args.csv)}")
    print("+" + "-" * (W - 2) + "+")
    row(f"X={args.x}   mode={args.mode}   hist_hw={args.hist_hw}")
    row(f"steps recorded={len(rows)}   overhead used={oh_const} us"
        + (f"   O_hist={dts}" if args.mode == "hybrid" else ""))
    rule()
    row()
    row("[1] INTEGRATION CHECK  (same input, both sides of the engine)")
    row(f"      live  (deployed API, inside ClTuneGemm) stopped at {live_stop}")
    row(f"      batch (offline estimator, same stream)  stopped at {batch_stop}")
    gap = batch_stop - live_stop
    row(f"      gap = {gap:+d} steps")
    row()
    if abs(gap) <= 2:
        row("      VERDICT: MATCH -- the integration feeds the engine the")
        row("      right values in the right order. A gap of 0-2 steps is")
        row("      expected: the deployed API uses the MEASURED per-step")
        row("      total, the batch model one CONSTANT overhead.")
    else:
        row("      VERDICT: MISMATCH -- larger than the constant-overhead")
        row("      approximation explains. Check: is kernel_us the right")
        row("      column, are units microseconds, is X the same, and does")
        row("      the session push every step exactly once?")
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