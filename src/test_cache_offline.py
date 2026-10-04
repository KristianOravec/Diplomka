#!/usr/bin/env python3
"""
test_cache_offline.py -- check that historical_params.csv is an exact drop-in
for computing O_hist and frozen a, b. Offline harness, no GPU needed.

  1. HITS : a debug build logs "from CACHE" / "COMPUTED" per config;
            every hist and hybrid config must read from the table.
  2. SAME : run twice with the same seed, with the table and without it
            (forced to compute); all result rows must be identical.
            A difference means the table is stale
            (or there is some bug :()): rerun precompute_historical.py.

Usage: python3 test_cache_offline.py [--table FILE] [--trials N] [--mode all|live|hist|hybrid]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

LIB_SOURCES = ["tuner_api", "evaluator", "curvefit", "csv",
               "historical_cache", "minicsv"]
LINK_FLAGS = ["-lgsl", "-lgslcblas", "-llbfgs", "-lm"]
SEED = "1"


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def find_repo_root():
    """Walk up from this script until it finds src/tuner_api.c, it awkwardly works"""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if (os.path.isfile(os.path.join(d, "src", "tuner_api.c"))
                and os.path.isdir(os.path.join(d, "raw-data"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            sys.exit("ERROR: repo root (src/tuner_api.c + raw-data/) not found")
        d = parent


def build(root, out_path, debug=False):
    """Compile validate_total_runtime + the library into one binary. """
    src = os.path.join(root, "src")
    files = [os.path.join(src, "validate_total_runtime.c")]
    files += [os.path.join(src, f + ".c") for f in LIB_SOURCES]
    cmd = ["gcc", "-O2", "-fopenmp", "-I" + src, "-o", out_path] + files + LINK_FLAGS
    if debug:
        cmd.insert(1, "-DTUNER_DEBUG=1")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("ERROR: build failed:\n" + r.stderr)

def run_harness(binary, table, trials, mode, out_file, cwd):
    """Run validate_total_runtime (built by build()) with the given table.
    Returns (stderr text, seconds taken)."""
    # the C library reads the table path from this environment variable
    env = dict(os.environ)
    env["TUNER_HISTORICAL_PARAMS"] = table

    # validate_total_runtime <trials> <mode> <out_file> <seed>
    command = [binary, str(trials), mode, out_file, SEED]

    start = time.time()
    result = subprocess.run(command, cwd=cwd, env=env,
                            capture_output=True, text=True)
    seconds = time.time() - start

    return result.stderr, seconds


def result_rows(path):
    """The per-config result lines, e.g. '1070/10000/0.5/100  7.891 ...'."""
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln[:1].isdigit() and "/" in ln]


def count_configs(rows, k):
    """How many result rows use k ('0.0' = hist, '0.5' = hybrid)."""
    return sum(1 for r in rows if r.split()[0].split("/")[2] == k)


def status(ok):
    return "PASS" if ok else "FAIL"



# the two checks

def check_hits(debug_bin, table, root, workdir):
    """CHECK 1: every historical value must come from the table."""
    out_file = os.path.join(workdir, "hits.txt")
    log, _ = run_harness(debug_bin, table, 1, "all", out_file, root)
    rows = result_rows(out_file)

    # how many configs of each mode ran
    hist_configs = count_configs(rows, "0.0")
    hybrid_configs = count_configs(rows, "0.5")

    # how many of them read the table (CACHE) vs computed (COMPUTED)
    hist_hits = log.count("a,b from CACHE")
    hist_misses = log.count("a,b COMPUTED")
    hybrid_hits = log.count("O_hist from CACHE")
    hybrid_misses = log.count("O_hist COMPUTED")

    ok = (hist_hits == hist_configs and hist_misses == 0 and
          hybrid_hits == hybrid_configs and hybrid_misses == 0)

    print("CHECK 1  HITS -- is every historical value read from the table?")
    print(f"   hist  : {hist_hits}/{hist_configs} from table, {hist_misses} computed")
    print(f"   hybrid: {hybrid_hits}/{hybrid_configs} from table, {hybrid_misses} computed")
    if not ok:
        print("   -> some configs are missing in the table; "
              "add them to precompute_historical.py")
    print(f"   {status(ok)}\n")
    return ok


def check_same(bin_path, table, trials, mode, root, workdir, out_dir, timestamp):
    """CHECK 2: results with and without the table must be identical."""
    with_file = os.path.join(out_dir, f"cache_test_with_{timestamp}.txt")
    without_file = os.path.join(out_dir, f"cache_test_without_{timestamp}.txt")
    missing_table = os.path.join(workdir, "does_not_exist.csv")

    print(f"CHECK 2  SAME -- results WITH vs WITHOUT the table ({trials} trials/config)")

    # run 1: read values from the table
    _, time_with = run_harness(bin_path, table, trials, mode, with_file, root)
    # run 2: table missing -> every value is computed
    _, time_without = run_harness(bin_path, missing_table, trials, mode, without_file, root)

    rows_with = result_rows(with_file)
    rows_without = result_rows(without_file)

    ok = len(rows_with) > 0 and rows_with == rows_without

    print(f"   with table   : {time_with:.1f} s, {len(rows_with)} rows")
    print(f"   without table: {time_without:.1f} s, {len(rows_without)} rows")
    if ok:
        print("   all rows identical")
    else:
        print("   rows differ -> table is stale, rerun precompute_historical.py")
        print(f"   compare: diff {with_file} {without_file}")
    print(f"   {status(ok)}\n")
    return ok


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--table", default=None,
                    help="historical table (default: <repo>/historical_params.csv)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--mode", default="all",
                    choices=["all", "live", "hist", "hybrid"])
    args = ap.parse_args()

    root = find_repo_root()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(root, f"test_cache_offline_{timestamp}")
    os.makedirs(out_dir)
    table = os.path.abspath(args.table or os.path.join(root, "historical_params.csv"))
    if not os.path.isfile(table):
        sys.exit(f"ERROR: table not found: {table}\n"
                 "       generate it: python3 src/precompute_historical.py "
                 "--out historical_params.csv")
    with open(table) as f:
        n_rows = sum(1 for _ in f) - 1

    print(f"repo  : {root}")
    print(f"table : {table}  ({n_rows} rows)\n")

    with tempfile.TemporaryDirectory() as workdir:
        normal_bin = os.path.join(workdir, "vtr")
        # Debug build contains debug prints when cache is hit, line 170 and 191 in tuner_api.c
        debug_bin = os.path.join(workdir, "vtr_debug")
        print("building the harness (normal + debug) ...\n", flush=True)
        build(root, normal_bin)
        build(root, debug_bin, debug=True)

        hits_ok = check_hits(debug_bin, table, root, workdir)
        same_ok = check_same(normal_bin, table, args.trials, args.mode,
                             root, workdir, out_dir, timestamp)

    print("=" * 64)
    print(f"  table covers every harness config : {status(hits_ok)}")
    print(f"  identical results with / without  : {status(same_ok)}")
    if hits_ok and same_ok:
        print("  -> the precomputed table is an EXACT drop-in replacement")
    print("=" * 64)
    print(f"full outputs:{out_dir}")
    sys.exit(0 if hits_ok and same_ok else 1)


if __name__ == "__main__":
    main()
