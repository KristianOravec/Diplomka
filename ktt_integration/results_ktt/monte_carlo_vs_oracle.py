#!/usr/bin/env python3
"""
monte_carlo_vs_oracle.py -- offline Monte Carlo test of the tuner's stop
decision against two oracles, using the exhaustive pools.

REUSES existing code only:
  - the decision engine: budget_estimator.tuning_length_recommendation
    (the SAME commit-and-countdown logic the deployed C API implements)
  - the data: exhaustive CSVs (raw-data/raw-autotuning-data/...)
  - the statistical-oracle referee: order statistics over the full pool

Per trial: draw a random tuning run (with replacement, like the C engine),
run the estimator, get its stop index, then score two ways:
  per-sample oracle : optimal stop on the SAME drawn run (the harness's
                      crystal-ball definition; reproduces work_* C_decl)
  statistical oracle: expected-optimal play over the whole pool (the referee
                      the real-GPU rand sessions were scored against)

--------------------------------------------------------------------------
NOTE ON `k` AND WHAT `--mode` ACTUALLY SELECTS
--------------------------------------------------------------------------
`k` chooses the MODE, not a blend weight:
    k = 1      -> live      : fit a,b,c from the live run; no history used
    k = 0      -> hist      : freeze a,b from historical hardware, refit c
    0 < k < 1  -> hybrid    : live estimate blended with a historical backstop

In hybrid, the k value itself is NEVER USED. tuning_length_recommendation
overwrites it on the first blended step:

    regression_weight = min(1, i / default_tuning_steps)

so the blend weight is time-varying (paper Eq. 4): early on the HISTORICAL
budget dominates, and by i = O_hist the LIVE estimate takes over completely.
--mode hybrid with k=0.5, 0.3 or 0.99 all behave identically.

What actually drives hybrid quality is WHICH GPU supplies O_hist, i.e.
--hist-hw. A mismatched historical GPU pulls the stop far from the optimum.
--------------------------------------------------------------------------

RUN (from python/):
  python3 monte_carlo_vs_oracle.py --hw 5080 --hist-hw 1070 \
      --mode hybrid --overhead 16600 --trials 1000

  # repeat the WHOLE experiment under several seeds to see how much the
  # averaged result itself moves:
  python3 monte_carlo_vs_oracle.py ... --seeds 1 2 3 4 5
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from budget_estimator import tuning_length_recommendation   # noqa: E402
from evaluator import history, curve_limit_max              # noqa: E402


# ---------------------------------------------------------------------------
# OPTIONAL C ENGINE (ctypes)
#
# The C library implements the SAME commit-and-countdown rule as
# budget_estimator.tuning_length_recommendation. Loading it here lets a single
# run push the IDENTICAL drawn run through both engines and compare their stop
# steps directly -- a far stronger check than comparing two averages produced
# from different random streams.
#
# Build libevaluator.so first with the repo Makefile (from the project root):
#   make
#   (target libevaluator.so, SRCS = src/csv.c src/curvefit.c
#    src/evaluator.c src/historical_cache.c src/minicsv.c)
# ---------------------------------------------------------------------------
_C_LIB = None


def load_c_engine(path):
    """Bind recommend_tuning_length from the shared library. None if absent."""
    import ctypes
    import os
    global _C_LIB
    if _C_LIB is not None:
        return _C_LIB
    for cand in ([path] if path else []) + ["../libevaluator.so",
                                            "./libevaluator.so"]:
        if cand and os.path.exists(cand):
            lib = ctypes.CDLL(os.path.abspath(cand))
            lib.recommend_tuning_length.restype = ctypes.c_uint64
            lib.recommend_tuning_length.argtypes = [
                ctypes.c_uint64,                    # default_tuning_steps
                ctypes.POINTER(ctypes.c_double),    # tuning_run
                ctypes.c_uint64,                    # tuning_run_len
                ctypes.c_uint64,                    # total_kernel_runs
                ctypes.c_double,                    # regression_weight (k)
                ctypes.c_uint64,                    # fit_start
                ctypes.c_uint64,                    # overhead
                ctypes.c_double,                    # hist_a
                ctypes.c_double,                    # hist_b
            ]
            _C_LIB = lib
            return lib
    return None


def c_estimate(lib, dts, run, X, k, fit_start, overhead, a, b):
    """Call the C estimator on one drawn run. Returns a 0-based stop index.

    NOTE the sentinel difference: Python signals "no history" with dts = -1,
    but the C gate is `default_tuning_steps > 0`, so it expects 0. Passing -1
    into a uint64 would wrap to a huge number and silently enable the hybrid
    blend.
    """
    import ctypes
    arr = (ctypes.c_double * len(run))(*run)
    c_dts = 0 if dts is None or dts < 0 else int(dts)
    return int(lib.recommend_tuning_length(
        c_dts, arr, len(run), X, k, fit_start, overhead, a, b))


def load_pool(hw):
    """Exhaustive pool: column 'Computation duration (us)' of the CSV."""
    import pandas as pd
    path = f"../raw-data/raw-autotuning-data/gemm-reduced/{hw}-gemm-reduced_output.csv"
    df = pd.read_csv(path)
    col = ("Computation duration (us)" if "Computation duration (us)" in df
           else "Kernel duration (us)")
    return df[col].values.astype(float)


def statistical_oracle(pool, X, overhead):
    """Expected-optimal play over the whole pool.

    E[min of s draws] is computed EXACTLY for sampling WITH replacement, to
    match how the trials below draw their runs:

        E[min of s draws] = sum_i k_i * [ ((n-i)/n)^s - ((n-i-1)/n)^s ]

    (the earlier order-statistic approximation (n+1)/(s+1) assumes sampling
    WITHOUT replacement, which is not what the estimator sees).
    """
    k = np.sort(pool)
    n = len(k)
    c_step = k.mean() + overhead

    idx = np.arange(n)
    T, sstar = float("inf"), 1
    for s in range(1, X + 1):
        # probability that the minimum of s draws equals k[i]
        p = ((n - idx) / n) ** s - ((n - idx - 1) / n) ** s
        exp_best = float(np.dot(k, p))
        cost = s * c_step + (X - s) * exp_best
        if cost < T:
            T, sstar = cost, s
        elif s > sstar + 50 and cost > T * 1.05:
            break   # past the minimum and rising; no need to scan to X
    return T, sstar


def session_cost(tuning_run, est_idx, X, overhead):
    """Session cost if the estimator stopped at index `est_idx`.

    CONVENTION: tuning_length_recommendation returns a 0-BASED INDEX, and the
    reference evaluator.py indexes total_runtimes[estimate] where entry i
    corresponds to i+1 samples consumed. So `est_idx` means (est_idx + 1)
    steps were actually taken -- hence the +1 below. Getting this wrong drops
    the final step's cost and computes `best` over one sample too few.
    """
    n = est_idx + 1
    taken = tuning_run[:n]
    return taken.sum() + n * overhead + (X - n) * taken.min()


def per_sample_oracle(tuning_run, X, overhead):
    """Optimal stop on THIS drawn run (the harness's crystal ball).

    Uses the same "s steps taken" convention as session_cost above.
    """
    best, cum, T = float("inf"), 0.0, float("inf")
    for s in range(1, len(tuning_run) + 1):
        cum += tuning_run[s - 1] + overhead
        best = min(best, tuning_run[s - 1])
        cost = cum + (X - s) * best
        if cost < T:
            T = cost
    return T


def run_experiment(args, seed, pool, curve_limit, t_star, dts, a, b, verbose,
                   engine="python", lib=None, agree=None):
    """One full experiment: `args.trials` draws under a single seed.

    engine: "python" -> budget_estimator.tuning_length_recommendation
            "c"      -> the same rule from the shared library, via ctypes
    If `agree` is a list, each trial ALSO runs the other engine on the SAME
    drawn run and appends (py_stop, c_stop) so the two can be compared exactly.
    """
    np.random.seed(seed)
    k = {"live": 1.0, "hybrid": 0.5, "hist": 0.0}[args.mode]

    stat_decl, ps_decl, stops, bests = [], [], [], []
    for t in range(args.trials):
        run = np.random.choice(pool, size=curve_limit, replace=True)

        if engine == "c":
            est = c_estimate(lib, dts, run, args.x, k, args.fit_start,
                             args.overhead, a, b)
        else:
            est = tuning_length_recommendation(
                dts, run, args.hw, "gemm-reduced_output.csv", args.x,
                k, args.fit_start, args.overhead, a, b)

        if agree is not None:
            # same run through BOTH engines -- the exact comparison
            py = tuning_length_recommendation(
                dts, run, args.hw, "gemm-reduced_output.csv", args.x,
                k, args.fit_start, args.overhead, a, b)
            ce = c_estimate(lib, dts, run, args.x, k, args.fit_start,
                            args.overhead, a, b)
            agree.append((py, ce))

        t_est = session_cost(run, est, args.x, args.overhead)
        t_ps = per_sample_oracle(run, args.x, args.overhead)
        stat_decl.append(t_est / t_star - 1.0)
        ps_decl.append(t_est / t_ps - 1.0)
        stops.append(est + 1)                    # report as a STEP COUNT
        bests.append(run[:est + 1].min())
        if verbose and ((t + 1) % 5 == 0 or t == 0):
            print(f"  trial {t+1}: stop {est+1}, best {bests[-1]/1e3:.3f} ms, "
                  f"stat-decl {100*stat_decl[-1]:+.2f}%, "
                  f"ps-decl {100*ps_decl[-1]:+.2f}%", flush=True)
    return np.array(stat_decl), np.array(ps_decl), stops, bests


W = 74   # box width, matching the RNG self-check boxes in evaluator.c


def _rule(ch="-"):
    return "+" + ch * (W - 2) + "+"


def _row(text=""):
    return "| " + text.ljust(W - 4) + " |"


def summarise(tag, d, p, stops, bests, pool, ctx=None):
    """Print the results in the project's boxed report style."""
    print()
    print(_rule("="))
    print(_row(f"MONTE CARLO vs ORACLE -- {tag}"))
    if ctx:
        print(_rule())
        for line in ctx:
            print(_row(line))
    print(_rule("="))

    print(_row())
    print(_row("[1] STOP STEP  (where the estimator decided to stop)"))
    print(_row(f"      mean {np.mean(stops):.0f}   median {np.median(stops):.0f}"
               f"   range [{min(stops)}, {max(stops)}]"))

    print(_row())
    print(_row("[2] DECLINE vs PER-SAMPLE ORACLE   <- compare to C_decl%"))
    print(_row("      best possible stop on the SAME drawn run;"))
    print(_row("      always >= 0, this is the harness's definition"))
    print(_row(f"      mean {100*p.mean():+7.2f}%   median {100*np.median(p):+7.2f}%"))
    print(_row(f"      p10  {100*np.percentile(p,10):+7.2f}%   p90    "
               f"{100*np.percentile(p,90):+7.2f}%"))

    print(_row())
    print(_row("[3] DECLINE vs STATISTICAL ORACLE  (expectation over the pool)"))
    print(_row("      a lucky draw CAN beat it, so negative values are normal"))
    print(_row("      and it is NOT comparable to the C harness numbers"))
    print(_row(f"      mean {100*d.mean():+7.2f}%   median {100*np.median(d):+7.2f}%"))
    print(_row(f"      p10  {100*np.percentile(d,10):+7.2f}%   p90    "
               f"{100*np.percentile(d,90):+7.2f}%"))

    print(_row())
    print(_row("[4] CONFIG QUALITY"))
    gap = 100 * (np.mean(bests) / pool.min() - 1.0)
    print(_row(f"      best found  mean {np.mean(bests)/1e3:.3f} ms"))
    print(_row(f"      pool best        {pool.min()/1e3:.3f} ms   "
               f"({gap:+.1f}% off optimal)"))
    print(_row())
    print(_rule("="))


def engine_agreement(pairs):
    """Compare the two engines on the SAME drawn runs.

    This is the strong form of the C-vs-Python check: identical input, so any
    difference is real divergence, not sampling noise.
    """
    py = np.array([a for a, _ in pairs], dtype=float)
    ce = np.array([b for _, b in pairs], dtype=float)
    diff = ce - py
    same = int((diff == 0).sum())
    n = len(pairs)

    print()
    print(_rule("="))
    print(_row("ENGINE AGREEMENT  (same drawn runs, both estimators)"))
    print(_rule("="))
    print(_row())
    print(_row(f"      trials              {n}"))
    print(_row(f"      identical stop      {same} / {n}  ({100.0*same/n:.1f}%)"))
    print(_row(f"      mean |C - Python|   {np.abs(diff).mean():.2f} steps"))
    print(_row(f"      max  |C - Python|   {np.abs(diff).max():.0f} steps"))
    print(_row(f"      signed mean         {diff.mean():+.2f} steps"))
    print(_row())
    if same == n:
        verdict = "IDENTICAL -- the C port reproduces the reference exactly"
    elif np.abs(diff).mean() < 1.0:
        verdict = "EQUIVALENT within rounding (mean gap < 1 step)"
    else:
        verdict = "DIVERGENT -- investigate, this is not sampling noise"
    print(_row(f"      verdict: {verdict}"))
    print(_row())
    print(_rule("="))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="1070")
    ap.add_argument("--x", type=int, default=10000, dest="x")
    ap.add_argument("--mode", default="hybrid", choices=["live", "hybrid", "hist"])
    ap.add_argument("--overhead", type=int, default=100000)
    ap.add_argument("--fit-start", type=int, default=10)
    ap.add_argument("--hist-hw", default=None,
                    help="historic GPU supplying O_hist (hybrid) or the frozen "
                         "curve (hist); default: same as --hw. "
                         "e.g. --hw 5080 --hist-hw 1070")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="repeat the WHOLE experiment under each seed and "
                         "report the spread of the averaged results")
    ap.add_argument("--engine", default="python",
                    choices=["python", "c", "both"],
                    help="which estimator to score: the Python reference, the "
                         "C library via ctypes, or BOTH on the same drawn runs "
                         "(compares their stop steps exactly)")
    ap.add_argument("--lib", default=None,
                    help="path to the shared library (default: "
                         "../libevaluator.so, ./libevaluator.so)")
    args = ap.parse_args()

    lib = None
    if args.engine in ("c", "both"):
        lib = load_c_engine(args.lib)
        if lib is None:
            print("ERROR: could not load libevaluator.so. Build it with:")
            print("  make   (in the repo root; target libevaluator.so, "
                  "SRCS = src/csv.c src/curvefit.c src/evaluator.c "
                  "src/historical_cache.c src/minicsv.c)")
            print("  (or point --lib at an existing library)")
            sys.exit(1)

    pool = load_pool(args.hw)
    curve_limit = min(args.x, len(pool), curve_limit_max)

    t_star, o_stop = statistical_oracle(pool, args.x, args.overhead)
    print(f"pool {args.hw}: {len(pool)} configs, mean {pool.mean()/1e3:.2f} ms, "
          f"best {pool.min()/1e3:.3f} ms")
    print(f"X={args.x}, mode={args.mode}, overhead={args.overhead} us, "
          f"trials={args.trials}")
    print(f"statistical oracle: stop {o_stop}, T* = {t_star/1e6:.3f} s "
          f"(c_step = {pool.mean()/1e3:.2f} + {args.overhead/1e3:.2f} ms)")
    print()

    # sentinels: -1 means "no history" / "fit all curve params live"
    dts, a, b = -1, -1, -1
    hist_hw = args.hist_hw or args.hw
    if args.mode == "hybrid":
        dts = history(hist_hw, "gemm-reduced_output.csv", args.x, args.overhead)
        print(f"O_hist (historic gpu: {hist_hw}) = {dts}")
        print("  (note: k is unused in hybrid -- the blend weight is "
              "min(1, i/O_hist); --hist-hw is what matters)")
    elif args.mode == "hist":
        from evaluator import get_history_regression_parameters
        # FIX: was passing args.hw, which silently ignored --hist-hw and fitted
        # the curve from the TARGET gpu -- defeating the point of k=0, which is
        # to borrow a curve shape from DIFFERENT hardware.
        a, b = get_history_regression_parameters(
            hist_hw, "gemm-reduced_output.csv", args.x, args.fit_start)
        print(f"frozen curve (historic gpu: {hist_hw}): a={a:.4f} b={b:.2f}")
    print()

    if not args.seeds:
        agree = [] if args.engine == "both" else None
        scored = "c" if args.engine == "c" else "python"
        d, p, stops, bests = run_experiment(
            args, args.seed, pool, curve_limit, t_star, dts, a, b, verbose=True,
            engine=scored, lib=lib, agree=agree)
        ctx = [f"pool {args.hw}   X={args.x}   mode={args.mode}   "
               f"overhead={args.overhead} us",
               f"hist_hw={hist_hw}   trials={args.trials}   seed={args.seed}",
               f"engine={args.engine}   statistical oracle stop = {o_stop}"]
        summarise(f"{args.trials} trials", d, p, stops, bests, pool, ctx)
        if agree is not None:
            engine_agreement(agree)
        return

    # multi-seed: repeat the whole experiment, then report how much the
    # AVERAGED result itself moves between seeds
    means_stat, means_ps, mean_stops = [], [], []
    for s in args.seeds:
        d, p, stops, bests = run_experiment(
            args, s, pool, curve_limit, t_star, dts, a, b, verbose=False,
            engine=("c" if args.engine == "c" else "python"), lib=lib)
        means_stat.append(100 * d.mean())
        means_ps.append(100 * p.mean())
        mean_stops.append(np.mean(stops))
        print(f"  seed {s:>4}: stop {np.mean(stops):6.1f}   "
              f"stat-decl {100*d.mean():+7.2f}%   ps-decl {100*p.mean():+7.2f}%",
              flush=True)

    sp_ps = max(means_ps) - min(means_ps)
    print()
    print(_rule("="))
    print(_row(f"ACROSS {len(args.seeds)} SEEDS  ({args.trials} trials each)"))
    print(_rule("="))
    print(_row())
    print(_row(f"      mean stop   {np.mean(mean_stops):8.1f}      "
               f"spread {max(mean_stops)-min(mean_stops):.1f}"))
    print(_row(f"      ps-decl     {np.mean(means_ps):+8.2f}%     "
               f"spread {sp_ps:.2f} pp"))
    print(_row(f"      stat-decl   {np.mean(means_stat):+8.2f}%     "
               f"spread {max(means_stat)-min(means_stat):.2f} pp"))
    print(_row())
    verdict = ("TRIAL COUNT SUFFICIENT (averaged result is stable)"
               if sp_ps < 2.0 else
               "TRIAL COUNT TOO LOW (averaged result still moves)")
    print(_row(f"      verdict: {verdict}"))
    print(_row())
    print(_rule("="))


if __name__ == "__main__":
    main()
