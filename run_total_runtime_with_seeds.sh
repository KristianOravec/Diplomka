#!/usr/bin/env bash
# =============================================================================
# run_total_runtime.sh -- build and run the TOTAL-RUNTIME validation harness.
#
# Compares the deployed tuner API's performance decline against the Python
# reference values, on the metric that matters (total runtime, not stop step).
#
# USAGE:
#   ./run_total_runtime.sh                    # 300 trials, all modes
#   ./run_total_runtime.sh 300 live           # 300 trials, live mode only
#   ./run_total_runtime.sh 1000 all           # full run (slow)
#   ./run_total_runtime.sh 2 live debug       # tiny run, full per-step traces
#   ./run_total_runtime.sh 20 live rngcheck   # prove the randomness is real
#   ./run_total_runtime.sh 300 live seeds     # SAME configs across 5 seeds,
#                                             # side by side -> shows how much
#                                             # of each row is sampling noise
#   ./run_total_runtime.sh 20 live "" 1       # seed 1  (different sample)
#   ./run_total_runtime.sh 20 live "" 2       # seed 2  -> different numbers
#
# ARGS:
#   $1  trials  (default 300)
#   $2  mode    live | hist | hybrid | all   (default all)
#   $3  "debug"    -> TUNER_DEBUG + CURVEFIT_DEBUG (verbose per-step trace)
#       "rngcheck" -> EVALUATOR_RNG_CHECK + TUNER_RNG_CHECK (short RNG report)
#       "seeds"    -> run the grid once per seed and tabulate the spread
#   $4  seed base (optional). Omit for the reproducible default; pass 1, 2, 3
#       ... to draw DIFFERENT random samples. Running several seeds and
#       comparing shows how much of each row is sampling noise -- and is the
#       clearest way to demonstrate the randomness is real.
#
# Must be run from the directory containing raw-data/.
# =============================================================================
set -e

TRIALS="${1:-300}"
MODE="${2:-all}"
EXTRA="${3:-}"
SEED="${4:-}"        # optional seed base: omit = reproducible default

# ---- locate the sources (flat src/, or current dir) ----
if [ -f "src/tuner_api.h" ]; then
    SRC="src"
elif [ -f "tuner_api.h" ]; then
    SRC="."
else
    echo "ERROR: cannot find tuner_api.h (looked in ./src and .)" >&2
    exit 1
fi
echo "sources: $SRC/"

# ---- check the data is reachable from here ----
if [ ! -d "raw-data/raw-autotuning-data/gemm-reduced" ]; then
    echo "WARNING: raw-data/raw-autotuning-data/gemm-reduced not found here." >&2
    echo "         The harness reads data relative to the CURRENT directory," >&2
    echo "         so run this script from the dir containing raw-data/." >&2
fi

# ---- extra build flags ----
DBGFLAGS=""
BIN="validate_total_runtime"
RNGMODE=0
SEEDMODE=0
case "$EXTRA" in
    debug)
        DBGFLAGS="-DTUNER_DEBUG=1 -DCURVEFIT_DEBUG=1"
        BIN="validate_total_runtime_dbg"
        echo "debug traces: ON (use a small trial count!)"
        ;;
    rngcheck)
        DBGFLAGS="-DEVALUATOR_RNG_CHECK=1 -DTUNER_RNG_CHECK=1"
        BIN="validate_total_runtime_rng"
        RNGMODE=1
        echo "RNG self-check: ON"
        # the flags only exist in the updated sources -- fail early if missing
        if ! grep -q RNG_CHECK "$SRC/evaluator.c" 2>/dev/null; then
            echo "ERROR: $SRC/evaluator.c has no RNG_CHECK support." >&2
            echo "       Copy the updated evaluator.c and tuner_api.c first." >&2
            exit 1
        fi
        if [ "$TRIALS" -gt 100 ] 2>/dev/null; then
            echo "NOTE: the duplicate-seed check is O(n^2); 20-50 trials is plenty."
        fi
        ;;
    seeds)
        SEEDMODE=1
        echo "multi-seed comparison: ON"
        ;;
    "") ;;
    *)
        echo "ERROR: unknown 3rd argument '$EXTRA'" >&2
        echo "       (use 'debug', 'rngcheck' or 'seeds')" >&2
        exit 1
        ;;
esac

# ---- build ----
echo "building $BIN ..."
gcc -O2 -fopenmp -I"$SRC" $DBGFLAGS -o "$BIN" \
    "$SRC/validate_total_runtime.c" \
    "$SRC/tuner_api.c" "$SRC/evaluator.c" "$SRC/curvefit.c" \
    "$SRC/csv.c" "$SRC/random.c" "$SRC/minicsv.c" \
    -lgsl -lgslcblas -llbfgs -lm

echo "running: ./$BIN $TRIALS $MODE"
echo

if [ "$SEEDMODE" = "1" ]; then
    # ---- run the whole grid once per seed, then tabulate side by side ----
    # Different seeds draw DIFFERENT random samples, so the spread across
    # columns is the sampling noise for that row. A row whose columns agree
    # closely is a solid measurement; one that swings is noise-dominated.
    SEEDS="1 2 3 4 5"
    for s in $SEEDS; do
        echo "  seed $s ..."
        ./"$BIN" "$TRIALS" "$MODE" "seedrun_$s.txt" "$s" > /dev/null 2>&1
    done
    echo
    # join the per-seed files on the config label (column 1)
    awk -v seeds="$SEEDS" '
      FNR==1 { nf++ }                      # count files as they open
      /^[0-9]/ && NF>=4 {
          cfg=$1; val[cfg","nf]=$2; py[cfg]=$3;
          if (!(cfg in seen)) { seen[cfg]=1; order[++n]=cfg }
      }
      END {
          printf "%-26s", "config";
          for (f=1; f<=nf; f++) printf "%9s%d", "seed", f;
          printf "%10s%9s%9s\n", "Python", "spread", "sd";
          printf "%s\n", "-------------------------------------------------------------------------------------------";
          totspread=0;
          for (i=1; i<=n; i++) {
              cfg=order[i]; mn=1e18; mx=-1e18; sum=0; sum2=0; cnt=0;
              printf "%-26s", cfg;
              for (f=1; f<=nf; f++) {
                  v=val[cfg","f]; printf "%10.2f", v;
                  if (v<mn) mn=v; if (v>mx) mx=v; sum+=v; sum2+=v*v; cnt++;
              }
              m=sum/cnt; var=sum2/cnt-m*m;
              if (var < 0) var=0;   # cancellation -> tiny negative
              sd=sqrt(var);
              printf "%10.2f%9.2f%9.2f\n", py[cfg], mx-mn, sd;
              totspread+=mx-mn;
          }
          printf "%s\n", "-------------------------------------------------------------------------------------------";
          printf "mean spread across seeds: %.2f pp over %d configs\n", totspread/n, n;
          if (totspread == 0) {
              printf "\n*** WARNING: every seed gave IDENTICAL results ***\n";
              printf "The binary is ignoring the seed argument.\n";
              printf "Check:  grep -c seed_base <src>/validate_total_runtime.c\n";
              printf "If 0, copy the updated source and rebuild.\n";
          }
          printf "A row whose seeds agree closely is a solid measurement;\n";
          printf "a row that swings is dominated by sampling noise, not by the implementation.\n";
      }' seedrun_1.txt seedrun_2.txt seedrun_3.txt seedrun_4.txt seedrun_5.txt \
      | tee seed_comparison.txt
    echo
    echo "saved: seed_comparison.txt (per-seed files: seedrun_1..5.txt)"
elif [ "$RNGMODE" = "1" ]; then
    # The RNG report goes to stderr, results to stdout. Show ONLY the report
    # (clean to read or screenshot), and save the full run alongside it.
    ./"$BIN" "$TRIALS" "$MODE" rng_results.txt $SEED > /dev/null 2> rng_proof.txt || true
    cat rng_proof.txt
    echo
    echo "-----------------------------------------------------------------"
    echo "RNG evidence above (also in rng_proof.txt)."
    echo "Normal harness results went to rng_results.txt."
else
    ./"$BIN" "$TRIALS" "$MODE" total_runtime_results.txt $SEED
fi
