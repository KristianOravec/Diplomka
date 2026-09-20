"""
compare_cache_engines.py -- compare the C-generated and Python-generated
historical_params tables row by row.

Both tables were produced by the SAME grid in two engines:
  * C table:      src/precompute_historical.py      (libevaluator.so)
  * Python table: testing/precompute_historical_py.py (python/evaluator.py)

Rows are joined on the full key
    (kind, hist_hw, file_name, total_kernel_runs, fit_start, overhead,
     number_of_tests)
and printed one line per measurement:

  * REG rows (frozen curve, k=0): two lines -- relative % difference of `a`
    and of `b`, computed as (py - c)/|c|*100.
  * OPT rows (hybrid backstop):   one line -- absolute step difference of the
    historical optimum O_hist, computed as (py - c).

The summary block reports, for a / b / O_hist, the mean, median and max of the
|differences| across all compared keys, the number of keys compared, and how
many keys were present in one table but missing from the other.

USAGE
  python3 testing/compare_cache_engines.py [C_TABLE] [PY_TABLE] [--out PATH]

  With no positional arguments the script uses
  results_ktt/historical_params_c.csv and
  results_ktt/historical_params_py.csv -- but only if BOTH exist; otherwise it
  prints an error and requires the two paths explicitly.

  The full report is printed to stdout AND written to a text file:
    <dir of C table>/comparison_<stem1>_vs_<stem2>.txt
  where each stem is the input file name with a leading
  "historical_params_" prefix (if present) stripped -- e.g.
  historical_params_c.csv / historical_params_py.csv ->
  comparison_c_vs_py.txt. --out PATH overrides the derived file name.
"""
import argparse
import datetime
import os
import statistics
import sys

import pandas as pd

_HEADER = ["kind", "hist_hw", "file_name", "total_kernel_runs", "fit_start",
           "overhead", "number_of_tests", "a", "b", "optimal_steps"]


def load_table(path):
    df = pd.read_csv(path)
    missing = [c for c in _HEADER if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing}; expected "
                 f"{_HEADER}")
    # Normalise key columns so the join keys compare equal between tables
    # regardless of how the CSV writer typed them.
    df["hist_hw"] = df["hist_hw"].astype(str)
    df["file_name"] = df["file_name"].astype(str)
    for c in ("total_kernel_runs", "fit_start", "overhead",
              "number_of_tests"):
        df[c] = df[c].astype("int64")
    df["a"] = pd.to_numeric(df["a"], errors="coerce")
    df["b"] = pd.to_numeric(df["b"], errors="coerce")
    df["optimal_steps"] = pd.to_numeric(df["optimal_steps"],
                                        errors="coerce")
    return df


def sort_key(row):
    # (kind, hist_hw, runs, fit_start, overhead) -- fit_start and overhead
    # are mutually exclusive per kind but sort stably either way.
    return (row["kind"], row["hist_hw"], row["total_kernel_runs"],
            row["fit_start"], row["overhead"])


def derive_output_path(c_path, py_path):
    """Derive the report file name from the two input table names.

    Both input stems have a leading "historical_params_" prefix (if present)
    stripped, then the file is named "comparison_<stem1>_vs_<stem2>.txt" and
    placed in the same directory as the FIRST input CSV.
    """
    def stem(path):
        s = os.path.splitext(os.path.basename(path))[0]
        if s.startswith("historical_params_"):
            s = s[len("historical_params_"):]
        return s

    name = f"comparison_{stem(c_path)}_vs_{stem(py_path)}.txt"
    return os.path.join(os.path.dirname(os.path.abspath(c_path)), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("c_path", nargs="?", default=None,
                    help="C-engine table CSV (default: "
                         "results_ktt/historical_params_c.csv if it exists)")
    ap.add_argument("py_path", nargs="?", default=None,
                    help="Python-engine table CSV (default: "
                         "results_ktt/historical_params_py.csv if it exists)")
    ap.add_argument("--out", default=None,
                    help="report output file (default: derived from the two "
                         "input table names, next to the first CSV)")
    args = ap.parse_args()

    c_path = args.c_path
    py_path = args.py_path
    default_c = "results_ktt/historical_params_c.csv"
    default_py = "results_ktt/historical_params_py.csv"
    if c_path is None and py_path is None:
        if os.path.exists(default_c) and os.path.exists(default_py):
            c_path, py_path = default_c, default_py
        else:
            sys.exit("default tables missing from results_ktt/ (need BOTH "
                     f"{default_c} and {default_py}); pass the two table "
                     "paths explicitly")
    elif c_path is None or py_path is None:
        sys.exit("pass BOTH table paths (C table and Python table)")

    c_tbl = load_table(c_path)
    py_tbl = load_table(py_path)

    def keyed(df):
        return {tuple(r[c] for c in _HEADER[:7]): r for _, r in df.iterrows()}

    c_by_key = keyed(c_tbl)
    py_by_key = keyed(py_tbl)

    keys_c = set(c_by_key)
    keys_py = set(py_by_key)
    common = sorted(keys_c & keys_py, key=lambda k: sort_key(c_by_key[k]))
    missing_from_py = sorted(keys_c - keys_py,
                             key=lambda k: sort_key(c_by_key[k]))
    missing_from_c = sorted(keys_py - keys_c,
                            key=lambda k: sort_key(py_by_key[k]))

    rel_a, rel_b, step_diff = [], [], []      # absolute magnitudes
    sgn_a, sgn_b, sgn_step = [], [], []        # SIGNED, to expose bias
    out_rows = []

    for k in common:
        c = c_by_key[k]
        p = py_by_key[k]
        kind, hw, runs = c["kind"], c["hist_hw"], c["total_kernel_runs"]
        fs, oh, tests = c["fit_start"], c["overhead"], c["number_of_tests"]
        if kind == "REG":
            c_a, p_a = float(c["a"]), float(p["a"])
            c_b, p_b = float(c["b"]), float(p["b"])
            d_a = (p_a - c_a) / abs(c_a) * 100.0
            d_b = (p_b - c_b) / abs(c_b) * 100.0
            rel_a.append(abs(d_a))
            rel_b.append(abs(d_b))
            sgn_a.append(d_a)
            sgn_b.append(d_b)
            out_rows.append([kind, hw, runs, f"fs={fs} a",
                             f"{c_a:.10f}", f"{p_a:.10f}", f"{d_a:+.6f}%"])
            out_rows.append([kind, hw, runs, f"fs={fs} b",
                             f"{c_b:.10f}", f"{p_b:.10f}", f"{d_b:+.6f}%"])
        elif kind == "OPT":
            c_s = int(c["optimal_steps"])
            p_s = int(p["optimal_steps"])
            d = p_s - c_s
            step_diff.append(abs(d))
            sgn_step.append(d)
            out_rows.append([kind, hw, runs, f"oh={oh}",
                             str(c_s), str(p_s), f"{d:+d}"])
        else:
            sys.exit(f"unexpected kind in table: {kind!r}")

    # ---- per-key table -----------------------------------------------------
    hdr = ["kind", "hw", "runs", "discriminator", "C value", "Py value",
           "difference"]
    lines = []
    lines.append(f"comparing {c_path} (C) vs {py_path} (Py), "
                 f"{len(common)} common keys")
    lines.append(f"{hdr[0]:<6} {hdr[1]:<9} {hdr[2]:>10} {hdr[3]:<16} "
                 f"{hdr[4]:>24} {hdr[5]:>24} {hdr[6]:>24}")
    lines.append("-" * 110)
    for r in sorted(out_rows, key=lambda r: (r[0], r[1], r[2], r[3])):
        # re-sort by printed fields (kind, hw, runs, discriminator)
        lines.append(f"{r[0]:<6} {r[1]:<9} {str(r[2]):>10} {r[3]:<16} "
                     f"{r[4]:>24} {r[5]:>24} {r[6]:>24}")
    lines.append("-" * 110)

    # ---- missing keys ------------------------------------------------------
    if missing_from_py:
        lines.append(f"keys in C table but MISSING from Python table "
                     f"({len(missing_from_py)}):")
        for k in missing_from_py:
            lines.append("   " + str(k))
    if missing_from_c:
        lines.append(f"keys in Python table but MISSING from C table "
                     f"({len(missing_from_c)}):")
        for k in missing_from_c:
            lines.append("   " + str(k))

    # ---- summary stats -----------------------------------------------------
    if rel_a:
        lines.append(f"  a   |rel %|  mean {statistics.mean(rel_a):10.4f}"
                     f"  median {statistics.median(rel_a):10.4f}"
                     f"  max {max(rel_a):10.4f}"
                     f"   SIGNED mean {statistics.mean(sgn_a):+9.4f}")
    if rel_b:
        lines.append(f"  b   |rel %|  mean {statistics.mean(rel_b):10.4f}"
                     f"  median {statistics.median(rel_b):10.4f}"
                     f"  max {max(rel_b):10.4f}"
                     f"   SIGNED mean {statistics.mean(sgn_b):+9.4f}")
    if step_diff:
        pos = sum(1 for x in sgn_step if x > 0)
        lines.append(f"  O_hist |steps| mean {statistics.mean(step_diff):10.4f}"
                     f"  median {statistics.median(step_diff):10.4f}"
                     f"  max {max(step_diff):10.4f}"
                     f"   SIGNED mean {statistics.mean(sgn_step):+9.4f}")
        lines.append(f"  O_hist  Py > C in {pos}/{len(sgn_step)} rows")

    # ---- how to read the numbers above -------------------------------------
    # NOTE: deliberately OUTSIDE `if step_diff:` -- the interpretation must be
    # printed even for a comparison that happens to contain no OPT rows.
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("  Use the SIGNED means, not just the magnitudes: a signed mean")
    lines.append("  near zero means the engines disagree only by sampling noise,")
    lines.append("  while a consistent sign means a real systematic difference.")
    lines.append("")
    lines.append("  REG rows (frozen curve a, b)")
    lines.append("    total_kernel_runs is NOT part of this key. The fit uses")
    lines.append("    curve_limit = min(runs, pool size, CURVE_LIMIT_MAX), so every")
    lines.append("    runs >= 2000 is the SAME computation. Both precompute scripts")
    lines.append("    therefore emit ONE row per (hw, fit_start) with")
    lines.append("    total_kernel_runs = -1 (= any runs >= CURVE_LIMIT_MAX), so you")
    lines.append("    should see 4 REG rows here, not 12.")
    lines.append("    If a table still carries one row per runs value it predates")
    lines.append("    that change: its repeated a,b are correct but redundant, and")
    lines.append("    any VARIATION across them is a seeding artifact, not an engine")
    lines.append("    difference.")
    lines.append("")
    lines.append("  OPT rows (historical optimum O_hist)")
    lines.append("    A systematic Py > C bias is EXPECTED. Python samples WITHOUT")
    lines.append("    replacement (pandas .sample defaults to replace=False, giving")
    lines.append("    curve_limit distinct configs); the C samples WITH replacement")
    lines.append("    (~1690 distinct of 2000 drawn from a 5788 pool). Exploring")
    lines.append("    more distinct configs finds a better minimum, pushing the")
    lines.append("    cost-model optimum LATER -- hence larger O_hist in Python.")
    lines.append("    Documented in evaluator.c (SAMPLING NOTE).")

    # ---- emit: stdout unchanged + text file --------------------------------
    for ln in lines:
        print(ln)

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else \
            os.path.join(os.getcwd(), args.out)
    else:
        out_path = derive_output_path(c_path, py_path)
    header = [
        f"C table:     {c_path}",
        f"Python table: {py_path}",
        f"generated:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(header + lines) + "\n")
    print(f"report written to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Downstream consumer (e.g. `| head`) closed the pipe: exit quietly
        # instead of dumping a traceback.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)
