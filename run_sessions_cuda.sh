#!/usr/bin/env bash

SESSIONS_PER_MODE=${1:-1000}
OUTPUT_DIR=${2:-$HOME/sessions_1000}

TUNER=$HOME/Diplomka/ClTuneGemm
KERNELS=$HOME/KTT/Examples/ClTuneGemm
RAW_DATA=$HOME/Diplomka/raw-data

TOTAL_RUNS=10000
HISTORY_GPU=5050
OVERHEAD_US=68220
FIT_START=10

export CUDA_CACHE_DISABLE=1


session_is_done() {
    grep -qs "Summary:" "$1/run.log"
}

prepare_folder() {
    rm -rf "$1"
    mkdir -p "$1"
    ln -s "$RAW_DATA" "$1/raw-data"
}

run_tuner() {
    local folder=$1
    local mode=$2

    cd "$folder" || return
    "$TUNER" 0 0 "$KERNELS/ClTuneGemm.cu" "$KERNELS/ClTuneGemmReference.cu" \
        "$TOTAL_RUNS" "$mode" "$HISTORY_GPU" fast "$OVERHEAD_US" "$FIT_START" \
        > run.log 2>&1
    cd - > /dev/null
}

log_progress() {
    echo "$(date +%H:%M:%S) $1" >> "$OUTPUT_DIR/progress.log"
}


mkdir -p "$OUTPUT_DIR"

for number in $(seq 1 "$SESSIONS_PER_MODE"); do
    for mode in live hybrid hist; do
        folder="$OUTPUT_DIR/${mode}_${number}"

        if session_is_done "$folder"; then
            continue
        fi

        prepare_folder "$folder"
        run_tuner "$folder" "$mode"
        log_progress "$mode $number"
    done
done

log_progress "ALL DONE"
