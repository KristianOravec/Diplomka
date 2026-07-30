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
#
# ARGS:
#   $1  trials  (default 300)
#   $2  mode    live | hist | hybrid | all   (default all)
#   $3  "debug"    -> TUNER_DEBUG + CURVEFIT_DEBUG (verbose per-step trace)
#       "rngcheck" -> EVALUATOR_RNG_CHECK + TUNER_RNG_CHECK (short RNG report)
#
# Must be run from the directory containing raw-data/.
# =============================================================================
set -e

TRIALS="${1:-300}"
MODE="${2:-all}"
EXTRA="${3:-}"

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
    "") ;;
    *)
        echo "ERROR: unknown 3rd argument '$EXTRA' (use 'debug' or 'rngcheck')" >&2
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

if [ "$RNGMODE" = "1" ]; then
    # The RNG report goes to stderr, results to stdout. Show ONLY the report
    # (clean to read or screenshot), and save the full run alongside it.
    ./"$BIN" "$TRIALS" "$MODE" > rng_results.txt 2> rng_proof.txt || true
    cat rng_proof.txt
    echo
    echo "-----------------------------------------------------------------"
    echo "RNG evidence above (also in rng_proof.txt)."
    echo "Normal harness results went to rng_results.txt."
else
    ./"$BIN" "$TRIALS" "$MODE"
fi