#!/usr/bin/env bash
# Build and run the C-vs-Python validation locally.
#
# Usage:
#   ./run_validation.sh [trials] [mode] [which]
#     trials : MC trials per config (default 300; 1000 matches Python exactly)
#     mode   : fast (default) | all | live
#     which  : extended (default, compares decline% + estTS + GT-TS)
#              basic    (compares decline% + estTS only)
#
# Examples:
#   ./run_validation.sh                  # 300 trials, fast subset, extended
#   ./run_validation.sh 1000 all extended
#   ./run_validation.sh 200 live basic
#
# Requirements: gcc, libgsl-dev, liblbfgs-dev, OpenMP.
#   sudo apt-get install build-essential libgsl-dev liblbfgs-dev
#
# Run from the project root. Expects data at:
#   raw-data/raw-autotuning-data/gemm-reduced/<HW>-gemm-reduced_output.csv
set -euo pipefail

SRC="${SRC:-src}"
TRIALS="${1:-300}"
MODE="${2:-fast}"
WHICH="${3:-extended}"

if   [ -f "$SRC/include/evaluator.h" ]; then INC="$SRC/include";
elif [ -f "$SRC/evaluator.h" ];        then INC="$SRC";
elif [ -f "evaluator.h" ];             then INC="."; SRC=".";
else echo "ERROR: evaluator.h not found (looked in $SRC/include, $SRC, .)"; exit 1; fi
echo "sources: $SRC   headers: $INC"

DATA="raw-data/raw-autotuning-data/gemm-reduced/680-gemm-reduced_output.csv"
[ -f "$DATA" ] || echo "WARNING: $DATA not found; place the per-HW CSVs there."

if [ "$WHICH" = "basic" ]; then VALSRC="validate.c"; OUT="validate"; else VALSRC="validate_extended.c"; OUT="validate_extended"; fi
# find the validator source
if   [ -f "$SRC/$VALSRC" ]; then V="$SRC/$VALSRC";
elif [ -f "$VALSRC" ];      then V="$VALSRC";
else echo "ERROR: $VALSRC not found (in $SRC or .)"; exit 1; fi

echo "Compiling $OUT ..."
gcc -O2 -fopenmp -I"$INC" -o "$OUT" \
    "$V" "$SRC/evaluator.c" "$SRC/curvefit.c" "$SRC/csv.c" "$SRC/random.c" "$SRC/minicsv.c" \
    -lgsl -lgslcblas -llbfgs -lm

echo "Running: ./$OUT $TRIALS $MODE"; echo
"./$OUT" "$TRIALS" "$MODE"
