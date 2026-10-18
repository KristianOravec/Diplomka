#!/usr/bin/env bash
# Build everything that uses the C engine (src/*.c), from the same sources:
#   ~/Diplomka/ClTuneGemm        the KTT binary (C engine + ClTuneGemm.cpp + KTT)
#   ~/Diplomka/libevaluator.so   the library the Python scripts load (via 'make')
#
# Override locations with env vars if needed:
#   KTT_ROOT  (default ~/KTT)      CUDA_PATH  (default /usr/local/cuda)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KTT_ROOT="${KTT_ROOT:-$HOME/KTT}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
OUTPUT="$REPO/ClTuneGemm"
C_SOURCES="tuner_api evaluator curvefit csv historical_cache minicsv"

fail() {
    echo "ERROR: $1" >&2
    [ $# -ge 2 ] && echo "  fix: $2" >&2
    exit 1
}

has_library() {
    ldconfig -p 2>/dev/null | grep -q "$1"
}

check_requirements() {
    [ -f "$REPO/src/tuner_api.c" ] || fail "repo sources not found in $REPO/src"

    [ -f "$KTT_ROOT/Source/Ktt.h" ] || fail "KTT not found in $KTT_ROOT" \
        "git clone https://github.com/HiPerCoRe/KTT.git ~/KTT  (or export KTT_ROOT=...)"

    LIBKTT="$(find "$KTT_ROOT/Build" -name libktt.so 2>/dev/null | head -1)"
    [ -n "$LIBKTT" ] || fail "KTT is not built yet (no libktt.so)" \
        "cd $KTT_ROOT && premake5 gmake --no-opencl && cd Build && make config=release_x86_64 -j\$(nproc)"

    [ -f "$CUDA_PATH/lib64/libcudart.so" ] || fail "CUDA not found in $CUDA_PATH" \
        "export CUDA_PATH=/path/to/cuda"

    for tool in gcc g++ make; do
        command -v "$tool" >/dev/null || fail "$tool not found" "sudo apt install build-essential"
    done
    has_library libgsl.so || fail "GSL not found"     "sudo apt install libgsl-dev"
    has_library liblbfgs  || fail "libLBFGS not found" "sudo apt install liblbfgs-dev"

    echo "repo: $REPO"
    echo "KTT:  $LIBKTT"
    echo "CUDA: $CUDA_PATH"
}

compile_c_engine() {
    for name in $C_SOURCES; do
        gcc -c -O2 -I"$REPO/src" "$REPO/src/$name.c" -o "$OBJECTS/$name.o" \
            || fail "compiling $name.c failed"
    done
    echo "compiled the C engine: $C_SOURCES"
}

link_binary() {
    g++ -O2 -std=c++17 -DKTT_CUDA_EXAMPLE=1 \
        -I"$KTT_ROOT/Source" -I"$REPO/src" \
        "$REPO/ktt_integration/ClTuneGemm.cpp" "$OBJECTS"/*.o \
        -o "$OUTPUT" \
        -L"$(dirname "$LIBKTT")" -lktt \
        -L"$CUDA_PATH/lib64" -lcudart \
        -lgsl -lgslcblas -llbfgs -lm -fopenmp \
        || fail "linking failed"
    echo "built: $OUTPUT"
}

build_python_library() {
    make -B -C "$REPO" --no-print-directory > /dev/null || fail "make (libevaluator.so) failed"
    [ -f "$REPO/libevaluator.so" ] || fail "make ran but libevaluator.so is missing"
    echo "built: $REPO/libevaluator.so"
}

print_usage_example() {
    local kernels="$KTT_ROOT/Examples/ClTuneGemm"
    echo
    echo "run e.g.:"
    echo "  $OUTPUT 0 0 $kernels/ClTuneGemm.cu $kernels/ClTuneGemmReference.cu 10000 hybrid 5050 fast 68220"
    echo "  $OUTPUT 0 0 $kernels/ClTuneGemm.cu $kernels/ClTuneGemmReference.cu 5788 ref 5050"
}

OBJECTS="$(mktemp -d)"
trap 'rm -rf "$OBJECTS"' EXIT

check_requirements
compile_c_engine
link_binary
build_python_library
print_usage_example
