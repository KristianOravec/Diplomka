#!/usr/bin/env bash
# =============================================================================
# run_all_seeds.sh -- multi-seed comparison for ALL THREE modes, in one go.
#
# Runs run_total_runtime_with_seeds.sh in `seeds` mode for live, hist and
# hybrid, saving each mode's table under its own name so they do not overwrite
# each other.
#
# USAGE:
#   ./run_all_seeds.sh              # 1000 trials (long: hours)
#   ./run_all_seeds.sh 300          # 300 trials  (recommended, still solid)
#   ./run_all_seeds.sh 300 "live hist"   # only the modes you name
#
# Order is deliberate: live first (fastest, fails fast if something is wrong),
# hybrid last (slowest -- it computes O_hist over 1000 simulated runs per
# config, five times over).
#
# Run from the directory containing raw-data/.
# =============================================================================
set -u   # not -e: one failing mode should not abandon the rest

TRIALS="${1:-1000}"
MODES="${2:-live hist hybrid}"
RUNNER="./run_total_runtime_with_seeds.sh"

# ---- preflight: catch the two things that silently waste hours ----
if [ ! -x "$RUNNER" ]; then
    echo "ERROR: $RUNNER not found or not executable." >&2
    echo "       chmod +x $RUNNER" >&2
    exit 1
fi
if ! grep -q SEEDMODE "$RUNNER"; then
    echo "ERROR: $RUNNER has no 'seeds' mode (old copy)." >&2
    exit 1
fi
SRC="src"; [ -f "tuner_api.h" ] && SRC="."
if ! grep -q seed_base "$SRC/validate_total_runtime.c" 2>/dev/null; then
    echo "ERROR: $SRC/validate_total_runtime.c ignores the seed argument." >&2
    echo "       Every seed would produce IDENTICAL results. Update it first." >&2
    exit 1
fi
echo "preflight OK (sources in $SRC/)"
echo

START=$(date +%s)
for MODE in $MODES; do
    echo "==============================================================="
    echo " MODE: $MODE   trials=$TRIALS   started $(date +%H:%M:%S)"
    echo "==============================================================="
    T0=$(date +%s)

    if "$RUNNER" "$TRIALS" "$MODE" seeds; then
        mv -f seed_comparison.txt "seeds_${MODE}.txt"
        mkdir -p "seedruns_${MODE}"
        mv -f seedrun_*.txt "seedruns_${MODE}/" 2>/dev/null || true
        echo
        echo "  -> saved seeds_${MODE}.txt  (raw runs in seedruns_${MODE}/)"
    else
        echo "  -> MODE $MODE FAILED, continuing with the rest" >&2
    fi

    echo "  -> $MODE took $(( ($(date +%s) - T0) / 60 )) min"
    echo
done

echo "==============================================================="
echo "ALL DONE in $(( ($(date +%s) - START) / 60 )) min"
ls -la seeds_*.txt 2>/dev/null
echo
echo "Each file holds one row per config with a column per seed:"
echo "  tight columns  = solid measurement"
echo "  wide  columns  = dominated by sampling noise"
