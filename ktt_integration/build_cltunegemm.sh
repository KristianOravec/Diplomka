#!/usr/bin/env bash
# =============================================================================
# build_cltunegemm.sh -- build (and optionally validate) the CI-tuner-driven
#                        ClTuneGemm binary.
#
# Portable: no hardcoded user paths. Locations can be overridden via env:
#   KTT_ROOT      KTT clone containing Source/Ktt.h + Build/.../libktt.so
#                 (default: ~/KTT)
#   CUDA_PATH     CUDA toolkit (default: /usr/local/cuda)
#   JOBS          parallel make jobs (default: nproc)
#
# USAGE
#   ./build_cltunegemm.sh            # build only
#   ./build_cltunegemm.sh --test     # build + smoke-validate on the GPU
#
# WHAT IT BUILDS
#   C library objects (gcc): src/{tuner_api,evaluator,curvefit,csv,
#                              historical_cache,minicsv}.c
#   Integration host (g++):  ClTuneGemm.cpp, linked against libktt.so
#   Output: $ROOT/ClTuneGemm  (next to the repo root, next to raw-data/)
#
# DEPENDENCIES (checked, with the exact fix printed if missing)
#   gcc, g++ (>= C++17), libgsl-dev, liblbfgs-dev, CUDA toolkit,
#   a built KTT (see PREMAKE step below), OpenMP
#
# PREREQUISITE: KTT itself. If ~/KTT/Build/*/libktt.so is missing:
#     cd "$KTT_ROOT" && export CUDA_PATH && premake5 gmake --no-opencl
#     cd Build && make config=release_x86_64 -j$(nproc)
#   (premake5 binary: https://github.com/premake/premake-core/releases)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # Diplomka root
KTT_ROOT="${KTT_ROOT:-$HOME/KTT}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
JOBS="${JOBS:-$(nproc)}"
BUILD_TEST=0
[ "${1:-}" = "--test" ] && BUILD_TEST=1

err() { echo "ERROR: $*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "== 0. locating inputs =============================================="
[ -f "$ROOT/src/tuner_api.c" ] || { err "repo sources not found at $ROOT/src"; exit 1; }
[ -f "$KTT_ROOT/Source/Ktt.h" ] || {
    err "KTT not found at $KTT_ROOT (need Source/Ktt.h)"
    echo "  fix: git clone https://github.com/HiPerCoRe/KTT.git ~/KTT"
    echo "       (or: export KTT_ROOT=/path/to/KTT)"
    exit 1
}
LIBKTT="$(find "$KTT_ROOT/Build" -name libktt.so 2>/dev/null | head -1)"
[ -n "$LIBKTT" ] || {
    err "libktt.so not found under $KTT_ROOT/Build -- KTT is not built yet"
    echo "  fix: cd $KTT_ROOT && export CUDA_PATH=$CUDA_PATH"
    echo "       premake5 gmake --no-opencl && cd Build && make config=release_x86_64 -j$JOBS"
    exit 1
}
[ -d "$CUDA_PATH" ] || {
    err "CUDA toolkit not found at $CUDA_PATH (need nvrtc/cudart)"
    echo "  fix: install cuda-toolkit (Ubuntu/WSL: NVIDIA cuda repo, toolkit only,"
    echo "       NEVER the driver) or: export CUDA_PATH=/path/to/cuda"
    exit 1
}
for d in gcc g++ make; do
    have "$d" || { err "$d not found (apt install build-essential)"; exit 1; }
done
ldconfig -p 2>/dev/null | grep -q libgsl.so || {
    err "GSL not found (apt install libgsl-dev)"; exit 1; }
ldconfig -p 2>/dev/null | grep -q liblbfgs || {
    err "libLBFGS not found (apt install liblbfgs-dev)"; exit 1; }
[ -f "$CUDA_PATH/lib64/libcudart.so" ] || {
    err "libcudart not found under $CUDA_PATH/lib64"; exit 1; }
echo "   repo:    $ROOT"
echo "   KTT:     $LIBKTT"
echo "   CUDA:    $CUDA_PATH"

echo "== 1. compiling the C library objects (gcc) ========================="
OBJDIR="$(mktemp -d /tmp/cltunegemm_objs.XXXXXX)"
trap 'rm -rf "$OBJDIR"' EXIT
for f in tuner_api evaluator curvefit csv historical_cache minicsv; do
    gcc -c -O2 -I"$ROOT/src" "$ROOT/src/$f.c" -o "$OBJDIR/$f.o" \
        || { err "compiling $f.c failed"; exit 1; }
done
echo "   6 objects -> $OBJDIR"

echo "== 2. linking ClTuneGemm ============================================"
OUT="$ROOT/ClTuneGemm"
g++ -O2 -std=c++17 -DKTT_CUDA_EXAMPLE=1 \
    -I"$KTT_ROOT/Source" -I"$ROOT/src" \
    "$ROOT/ktt_integration/ClTuneGemm.cpp" "$OBJDIR"/*.o \
    -o "$OUT" \
    -L"$(dirname "$LIBKTT")" -lktt \
    -L"$CUDA_PATH/lib64" -lcudart \
    -lgsl -lgslcblas -llbfgs -lm -fopenmp \
    || { err "link failed"; exit 1; }
echo "   built: $OUT"

if [ "$BUILD_TEST" = "1" ]; then
    echo "== 3. smoke validation (X=5 must NOT stop: warmup > X) =============="
    export LD_LIBRARY_PATH="$(dirname "$LIBKTT"):$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if pgrep -f "$OUT" >/dev/null 2>&1; then
        err "another ClTuneGemm is running -- kill it first (two GPU runs = garbage timings)"
        exit 1
    fi
    # Run in a THROWAWAY directory. The binary writes tuning_steps_<tag>.csv
    # and GemmOutput_<tag>.xml into its cwd; running in $ROOT would mix smoke
    # output with real session files (and an earlier version of this test
    # deleted files there). live mode reads no historical data, so it does not
    # need raw-data/ to be reachable.
    SMOKEDIR="$(mktemp -d /tmp/cltunegemm_smoke.XXXXXX)"
    trap 'rm -rf "$OBJDIR" "$SMOKEDIR"' EXIT
    OUTFILE="$SMOKEDIR/stdout.txt"
    ( cd "$SMOKEDIR" && timeout 120 "$OUT" 0 0 \
        "$KTT_ROOT/Examples/ClTuneGemm/ClTuneGemm.cu" \
        "$KTT_ROOT/Examples/ClTuneGemm/ClTuneGemmReference.cu" \
        5 live 680 rand ) > "$OUTFILE" 2>&1 \
        || { err "smoke run crashed:"; tail -5 "$OUTFILE"; exit 1; }

    # 1. with X=5 the library cannot stop (first decision is at fit_start+6),
    #    so the budget must run out while still tuning
    grep -q "Execution budget spent while still tuning" "$OUTFILE" \
        || { err "smoke run did NOT produce the expected verdict:"; tail -8 "$OUTFILE"; exit 1; }

    # 2. the run must announce its own conditions
    grep -q "^CONFIG: X=5 mode=live" "$OUTFILE" \
        || { err "CONFIG line missing -- is this the updated ClTuneGemm.cpp?"; exit 1; }

    # 3. live mode must NOT trigger the hybrid/hist overhead warning
    ! grep -q "WARNING: mode" "$OUTFILE" \
        || { err "overhead warning fired in live mode (it should not)"; exit 1; }

    # 4. the step log: the binary names it tuning_steps_<tag>.csv, NOT
    #    tuning_steps.csv -- the previous check looked for the wrong name and
    #    therefore failed on every correct build.
    STEPFILE="$(ls "$SMOKEDIR"/tuning_steps_*.csv 2>/dev/null | head -1)"
    [ -n "$STEPFILE" ] && [ -s "$STEPFILE" ] \
        || { err "no tuning_steps_*.csv produced"; ls -la "$SMOKEDIR"; exit 1; }
    STEPS=$(( $(wc -l < "$STEPFILE") - 1 ))
    [ "$STEPS" -eq 5 ] \
        || { err "expected 5 tuning steps in $(basename "$STEPFILE"), got $STEPS"; exit 1; }

    # 5. total_us must equal kernel_us + overhead_us on every row (catches the
    #    overhead=0 class of bug that invalidated an earlier session)
    awk -F, 'NR>1 { d=$4-($2+$3); if (d>1 || d<-1) bad=1 } END { exit bad }' "$STEPFILE" \
        || { err "total_us != kernel_us + overhead_us in $(basename "$STEPFILE")"; exit 1; }

    echo "   PASS: X=5 ran without stopping, CONFIG line present, no spurious"
    echo "         warning, $(basename "$STEPFILE") has $STEPS consistent rows."
    echo "   GPU stack + KTT + library + build all OK."
fi

echo "== done. run e.g.: ==================================================="
echo "  export LD_LIBRARY_PATH=$(dirname "$LIBKTT"):$CUDA_PATH/lib64"
echo "  cd $ROOT && ./ClTuneGemm 0 0 $KTT_ROOT/Examples/ClTuneGemm/ClTuneGemm.cu \\"
echo "      $KTT_ROOT/Examples/ClTuneGemm/ClTuneGemmReference.cu 10000 hybrid 1070 rand - <overhead_us>"
echo "  (hybrid/hist NEED the measured overhead as argv[10] -- e.g. the median"
echo "   overhead_us of a reference sweep -- or O_hist comes out too deep.)"
