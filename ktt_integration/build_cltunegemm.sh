#!/usr/bin/env bash
# =============================================================================
# build_cltunegemm.sh -- build the CI-tuner-driven
#                        ClTuneGemm binary.
#
# Portable: no hardcoded user paths. Locations can be overridden via env:
#   KTT_ROOT      KTT clone containing Source/Ktt.h + Build/.../libktt.so
#                 (default: ~/KTT)
#   CUDA_PATH     CUDA toolkit (default: /usr/local/cuda)
#   JOBS          parallel make jobs (default: nproc)
#
# USAGE
#   ./build_cltunegemm.sh
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
ldconfig -p 2>/dev/null | grep libgsl.so > /dev/null || {
    err "GSL not found (apt install libgsl-dev)"; exit 1; }
ldconfig -p 2>/dev/null | grep liblbfgs > /dev/null || {
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

echo "== done. run e.g.: ==================================================="
echo "  export LD_LIBRARY_PATH=$(dirname "$LIBKTT"):$CUDA_PATH/lib64"
echo "  cd $ROOT && ./ClTuneGemm 0 0 $KTT_ROOT/Examples/ClTuneGemm/ClTuneGemm.cu \\"
echo "      $KTT_ROOT/Examples/ClTuneGemm/ClTuneGemmReference.cu 10000 hybrid 1070 - <overhead_us>"
echo "  (hybrid/hist NEED the measured overhead as argv[9] -- e.g. the median"
echo "   overhead_us of a reference sweep -- or O_hist comes out too deep.)"
echo "  reference sweep (whole space, 5788 configs):"
echo "      ... ClTuneGemmReference.cu 5788 ref"
