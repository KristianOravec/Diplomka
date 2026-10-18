#!/usr/bin/env bash
# Runs the offline C evaluator (validate_total_runtime) for several modes x seeds
# at the same time, one process per CPU core. Each run writes a C-vs-Python
# table (decline % per config) to its own result.txt.
#
# Usage (from the repo root):
#   ./run_all_seeds_parallel.sh                       1000 trials, all modes, seeds 1-5
#   ./run_all_seeds_parallel.sh 1000 "live hist hybrid" 1   all modes, one seed
#   ./run_all_seeds_parallel.sh 300 "live hist"       300 trials, only these modes
#   ./run_all_seeds_parallel.sh 1000 hybrid random    5 fresh random seeds
#   ./run_all_seeds_parallel.sh 1000 hybrid "7 8 9"   explicit seeds
#
# Output: seeds_output/work_<mode>_<seed>/result.txt, plus a one-line summary per run
# Ctrl+C stops every run.
set -u
trap 'echo; echo "stopped, killing all runs"; kill 0' INT TERM

TRIALS="${1:-1000}"
MODES="${2:-live hist hybrid}"
SEED_CHOICE="${3:-1 2 3 4 5}"

SRC="src"
BINARY="$(pwd)/validate_total_runtime"
DATA="$(pwd)/raw-data"
OUTPUT_DIR="$(pwd)/seeds_output"


pick_seeds() {
    if [ "$SEED_CHOICE" != "random" ]; then
        SEEDS="$SEED_CHOICE"
        return
    fi
    SEEDS=$(python3 -c "import random; print(*random.sample(range(1, 100000), 5))" 2>/dev/null \
            || echo "$RANDOM $RANDOM $RANDOM $RANDOM $RANDOM")
    echo "random seeds: $SEEDS"
    mkdir -p "$OUTPUT_DIR"
    echo "$SEEDS" > "$OUTPUT_DIR/seeds_used.txt"       # written down, so the run can be repeated
}

check_setup() {
    if ! grep -q seed_base "$SRC/validate_total_runtime.c" 2>/dev/null; then
        echo "ERROR: $SRC/validate_total_runtime.c ignores the seed (all seeds would be identical)" >&2
        exit 1
    fi
    if [ ! -d "raw-data/raw-autotuning-data/gemm-reduced" ]; then
        echo "ERROR: raw-data/ not found, run from the repo root" >&2
        exit 1
    fi
}

build_binary() {
    echo "building validate_total_runtime ..."
    gcc -O2 -fopenmp -I"$SRC" -o "$BINARY" \
        "$SRC/validate_total_runtime.c" \
        "$SRC/tuner_api.c" "$SRC/evaluator.c" "$SRC/curvefit.c" \
        "$SRC/csv.c" "$SRC/historical_cache.c" "$SRC/minicsv.c" \
        -lgsl -lgslcblas -llbfgs -lm || exit 1
}

format_duration() {
    printf '%dm%02ds' $(($1 / 60)) $(($1 % 60))
}

# One run in its own folder, so parallel runs never overwrite each other's files.
run_one() {
    local mode=$1 seed=$2
    local folder="$OUTPUT_DIR/work_${mode}_${seed}"

    rm -rf "$folder"
    mkdir -p "$folder"
    ln -s "$DATA" "$folder/raw-data"

    (
        cd "$folder" || exit 1
        local start=$(date +%s)
        "$BINARY" "$TRIALS" "$mode" result.txt "$seed" > /dev/null 2>&1
        local seconds=$(( $(date +%s) - start ))
        printf '  done: %-8s seed %-3s %s\n' "$mode" "$seed" "$(format_duration $seconds)"
    ) &
}

run_all() {
    local cores=$(nproc --all)           # read before OMP_NUM_THREADS=1, nproc honours it
    export OMP_NUM_THREADS=1             # one thread per run; parallelism = many runs

    local jobs=0
    for mode in $MODES; do for seed in $SEEDS; do jobs=$((jobs + 1)); done; done
    echo "launching $jobs concurrent runs ($TRIALS trials each) on $cores cores"
    echo

    local start=$(date +%s)
    for mode in $MODES; do
        for seed in $SEEDS; do
            run_one "$mode" "$seed"
        done
    done

    echo "waiting for all runs to finish ..."
    wait
    echo
    echo "all runs finished in $(( ($(date +%s) - start) / 60 )) min"
}

print_summaries() {
    echo
    for mode in $MODES; do
        for seed in $SEEDS; do
            local result="$OUTPUT_DIR/work_${mode}_${seed}/result.txt"
            if [ -f "$result" ]; then
                echo "$mode seed $seed: $(grep 'mean |C - Py|' "$result")"
            else
                echo "$mode seed $seed: no result"
            fi
        done
    done
}


pick_seeds
check_setup
build_binary
run_all
print_summaries
echo "full tables: seeds_output/work_<mode>_<seed>/result.txt"
