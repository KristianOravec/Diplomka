#!/usr/bin/env python3
"""
compare_ktt_vs_work.py -- compare real KTT/GEMM tuning sessions (ClTuneGemm.cpp
integration) against the work_*/seeds_* harness results.

INPUTS
  estimator CSV   : tuning_steps.csv     (ClTuneGemm integration, mode live/hybrid/hist)
  reference CSV   : reference_steps.csv  (ClTuneGemm integration, mode ref)
  work results    : work_<mode>_<seed>/result.txt (or seeds_<mode>.txt) tables
                    rows: HW/runs/k/oh   C_decl%   Py_decl%   diff(pp)

METRIC (real-hardware twin of the harness's decline%):
  T_est(s)  = sum(total_us[:s]) + (X - s) * best_kernel_us[s]     -- cost of the
              actual CI-tuner session that stopped at step s
  T_star(X) = min over s of sum(total_us[:s]) + (X-s)*best_us[s]  (reference log)
              -- the best achievable session on this hardware (oracle proxy)
  decline_real = T_est / T_star - 1
Compared against the work_* row for the same (mode, runs~X, overhead bucket).

USAGE
  python3 compare_ktt_vs_work.py --X 200 --mode live \
      --estimator tuning_steps.csv --reference reference_steps.csv \
      --work seeds_live.txt
"""

import argparse
import csv
import sys


def load_session(path, has_budget=True):
    """Read a steps CSV -> (totals, kernels, stop_index)."""
    totals, kernels, stop = [], [], None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kernels.append(float(row["kernel_us"]))
            totals.append(float(row["total_us"]))
            if has_budget and row.get("budget") and int(row["budget"]) == 0 and stop is None:
                stop = len(totals)
    if stop is None:
        stop = len(totals)
    return totals, kernels, stop


def session_cost(totals, kernels, stop, X):
    """Session cost if tuning stopped after `stop` steps, X executions total."""
    best = min(kernels[:stop]) if stop else 0
    return sum(totals[:stop]) + (X - stop) * best


def oracle_from_reference(totals, kernels, X):
    """Best achievable session cost over all stop points of the reference log."""
    best_cost, best_stop = float("inf"), 1
    cum = 0.0
    best = float("inf")
    for s in range(1, len(totals) + 1):
        cum += totals[s - 1]
        best = min(best, kernels[s - 1])
        cost = cum + (X - s) * best
        if cost < best_cost:
            best_cost, best_stop = cost, s
    return best_cost, best_stop

import math


def oracle_statistical(kernels, X, c_step):
    """Oracle for a RANDOM-stream tuner, from an EXHAUSTIVE pool diary.

    The stream oracle (oracle_from_reference) scores one fixed stream; a
    random-stream session draws from the whole pool and can legitimately beat
    it. The correct referee is the EXPECTED-optimal play over the pool: at s
    draws the expected best-so-far is the order statistic
        E[min of s uniform draws without replacement] = k_sorted[(N+1)/(s+1)]
    and the session cost is s*c_step + (X-s)*E[min_s]. Minimize over s.
    Requires the reference log to be EXHAUSTIVE (whole pool; e.g. the full
    5788-config sweep). c_step = expected per-draw cost (pool mean kernel +
    mean measured overhead)."""
    k = sorted(kernels)
    N = len(k)
    T, sstar = float("inf"), 1
    for s in range(1, X + 1):
        pos = max(0, min(N - 1, math.floor((N + 1) / (s + 1)) - 1))
        cost = s * c_step + (X - s) * k[pos]
        if cost < T:
            T, sstar = cost, s
    return T, sstar



def parse_work_rows(path):
    """Parse a result.txt/seeds table into (cfg, C_decl, Py_decl) triples."""
    rows = []
    for line in open(path):
        parts = line.split()
        if len(parts) >= 4 and "/" in parts[0]:
            try:
                rows.append((parts[0], float(parts[1]), float(parts[2])))
            except ValueError:
                pass
    return rows


def nearest_work_row(rows, X, mode_k, mean_overhead):
    """Pick the work row closest in (runs~X, overhead~measured, k=mode)."""
    best, best_d = None, float("inf")
    for cfg, c, py in rows:
        parts = cfg.split("/")
        if len(parts) != 4 or parts[2] != mode_k:
            continue
        runs, oh = int(parts[1]), int(parts[3])
        d = abs(runs - X) / max(X, 1) + abs(oh - mean_overhead) / max(mean_overhead, 1)
        if d < best_d:
            best, best_d = (cfg, c, py), d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="tuning_steps.csv")
    ap.add_argument("--reference", default="reference_steps.csv")
    ap.add_argument("--work", default=None, help="seeds_<mode>.txt or work_*/result.txt")
    ap.add_argument("--mode", default="live", choices=["live", "hybrid", "hist"])
    ap.add_argument("--X", type=int, default=None,
                    help="execution budget X used in the estimator run (recommended)")
    ap.add_argument("--oracle", default="stream", choices=["stream", "statistical"],
                    help="stream: best stop on the reference log (paired det streams). "
                         "statistical: expected-optimal play over the pool -- use for "
                         "rand-mode sessions against an EXHAUSTIVE reference.")
    args = ap.parse_args()

    est_totals, est_kernels, stop = load_session(args.estimator, has_budget=True)
    X = args.X
    if X is None:
        # budget starts at min(X, 2000) and ticks down each step; recovering X
        # exactly is unreliable -- recommend passing --X explicitly.
        sys.exit("error: pass --X explicitly (the execution budget used in the run)")

    t_est = session_cost(est_totals, est_kernels, stop, X)
    ref_totals, ref_kernels, _ = load_session(args.reference, has_budget=False)
    mean_overhead = sum(est_totals[i] - est_kernels[i] for i in range(stop)) / stop
    if args.oracle == "statistical":
        c_step = sum(ref_kernels) / len(ref_kernels) + mean_overhead
        t_star, oracle_stop = oracle_statistical(ref_kernels, X, c_step)
    else:
        t_star, oracle_stop = oracle_from_reference(ref_totals, ref_kernels, X)
    decline = t_est / t_star - 1.0

    best_est = min(est_kernels[:stop])
    best_ref = min(ref_kernels)
    mean_overhead = sum(est_totals[i] - est_kernels[i] for i in range(stop)) / stop

    print(f"estimator session : {stop} tuning steps of X={X}")
    print(f"  T_est             : {t_est/1e6:.3f} s")
    print(f"  best kernel       : {best_est/1e3:.3f} ms  (reference best: {best_ref/1e3:.3f} ms)")
    print(f"  mean step overhead: {mean_overhead/1e3:.1f} ms")
    print(f"oracle ({args.oracle})  : stop {oracle_stop}, T* = {t_star/1e6:.3f} s")
    print(f"REAL decline        : {100*decline:.2f} %")
    print()

    if args.work:
        rows = parse_work_rows(args.work)
        if not rows:
            print(f"no parsable rows in {args.work}")
            return
        row = nearest_work_row(rows, X, {"live": "1.0", "hist": "0.0", "hybrid": "0.5"}[args.mode], mean_overhead)
        if row:
            print(f"nearest work_* row: {row[0]}")
            print(f"  C_decl% = {row[1]:.2f}   Py_decl% = {row[2]:.2f}")
            print(f"  real decline {100*decline:.2f} vs C {row[1]:.2f} vs Py {row[2]:.2f}")


if __name__ == "__main__":
    main()