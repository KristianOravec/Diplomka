#!/usr/bin/env bash
# =============================================================================
# run_all_seeds_parallel.sh -- multi-seed comparison for all modes, run
#                              CONCURRENTLY across CPU cores.
#
# HOW THE CLOBBERING IS AVOIDED
#   Each run gets its own working directory (work_<mode>_<seed>/) with raw-data
#   symlinked in, so the per-run output files cannot overwrite each other.
#
# USAGE
#   ./run_all_seeds_parallel.sh                      # 1000 trials, all modes
#   ./run_all_seeds_parallel.sh 300                  # 300 trials
#   ./run_all_seeds_parallel.sh 300 "live hist"      # subset of modes
#   ./run_all_seeds_parallel.sh 1000 hybrid random   # ONE mode, 5 random seeds
#   ./run_all_seeds_parallel.sh 1000 hybrid "7 8 9"  # explicit seed list
#
# ARGS
#   $1  trials  (default 1000)
#   $2  modes   space-separated: live / hist / hybrid  (default: all three)
#   $3  seeds   "1 2 3 4 5" (default) | "random" | any explicit list
#
# Run from the directory containing raw-data/.
# =============================================================================
set -u

TRIALS="${1:-1000}"
MODES="${2:-live hist hybrid}"
SEEDSPEC="${3:-1 2 3 4 5}"

# SEEDSPEC controls which seeds are used:
#   "1 2 3 4 5"  (default)  -- fixed, reproducible, comparable across runs
#   "random"                -- 5 fresh seeds drawn from OS entropy, so this run
#                              samples DIFFERENTLY from any previous one. Useful
#                              when a mode is re-run separately and you do not
#                              want to reuse the same five draws.
if [ "$SEEDSPEC" = "random" ]; then
    if command -v python3 >/dev/null 2>&1; then
        SEEDS=$(python3 -c "import random; random.seed(); print(' '.join(str(x) for x in random.sample(range(1,100000),5)))")
    else
        SEEDS="$RANDOM $RANDOM $RANDOM $RANDOM $RANDOM"
    fi
    echo "random seeds: $SEEDS"
    # record them -- the run is only reproducible if the seeds are written down
    echo "$SEEDS" > seeds_used.txt
else
    SEEDS="$SEEDSPEC"
fi

SRC="src"; [ -f "tuner_api.h" ] && SRC="."

# ---- preflight ----
if ! grep -q seed_base "$SRC/validate_total_runtime.c" 2>/dev/null; then
    echo "ERROR: $SRC/validate_total_runtime.c ignores the seed argument." >&2
    echo "       Every seed would give IDENTICAL results. Update it first." >&2
    exit 1
fi
if [ ! -d "raw-data/raw-autotuning-data/gemm-reduced" ]; then
    echo "ERROR: raw-data/ not found here. Run from the project root." >&2
    exit 1
fi

# ---- build ONCE, then every process reuses the same binary ----
echo "building validate_total_runtime ..."
gcc -O2 -fopenmp -I"$SRC" -o validate_total_runtime \
    "$SRC/validate_total_runtime.c" \
    "$SRC/tuner_api.c" "$SRC/evaluator.c" "$SRC/curvefit.c" \
    "$SRC/csv.c" "$SRC/historical_cache.c" "$SRC/minicsv.c" \
    -lgsl -lgslcblas -llbfgs -lm || exit 1
BIN="$(pwd)/validate_total_runtime"
DATA="$(pwd)/raw-data"

# Capture the core count BEFORE restricting OpenMP: nproc honours
# OMP_NUM_THREADS, so asking after the export would always report 1.
NCORES=$(nproc --all)

# each worker is single-threaded; parallelism comes from running many of them
export OMP_NUM_THREADS=1

NJOBS=0
for M in $MODES; do for S in $SEEDS; do NJOBS=$((NJOBS+1)); done; done
echo "launching $NJOBS concurrent runs ($TRIALS trials each) on $NCORES cores"
echo

START=$(date +%s)
for M in $MODES; do
    for S in $SEEDS; do
        D="work_${M}_${S}"
        rm -rf "$D"; mkdir -p "$D"
        ln -s "$DATA" "$D/raw-data"          # share the data, don't copy it
        (
            cd "$D" || exit 1
            T0=$(date +%s)
            "$BIN" "$TRIALS" "$M" "result.txt" "$S" > /dev/null 2>&1
            ELAPSED=$(( $(date +%s) - T0 ))
            printf '%d\n' "$ELAPSED" > time.txt
            printf '  done: %-8s seed %-3s %dm %02ds\n' "$M" "$S" \
                $((ELAPSED / 60)) $((ELAPSED % 60))
        ) &
    done
done

echo "waiting for all runs to finish ..."
wait
echo
echo "all runs finished in $(( ($(date +%s) - START) / 60 )) min"
for M in $MODES; do
    FILES=""
    TIMES=""
    for S in $SEEDS; do
        [ -f "work_${M}_${S}/result.txt" ] && FILES="$FILES work_${M}_${S}/result.txt"
        T="work_${M}_${S}/time.txt"
        [ -f "$T" ] && TIMES="$TIMES $(printf '%dm%02ds' $(($(cat "$T") / 60)) $(($(cat "$T") % 60)))"
    done
    [ -z "$FILES" ] && { echo "no results for $M"; continue; }

    awk '
      FNR==1 { nf++ }
      /^[0-9]/ && NF>=4 {
          cfg=$1; val[cfg","nf]=$2; py[cfg]=$3;
          if (!(cfg in seen)) { seen[cfg]=1; order[++n]=cfg }
      }
      END {
          printf "%-26s", "config";
          for (f=1; f<=nf; f++) printf "%9s%d", "seed", f;
          printf "%10s%9s%9s\n", "Python", "spread", "sd";
          printf "%s\n", "-------------------------------------------------------------------------------------------";
          tot=0;
          for (i=1; i<=n; i++) {
              cfg=order[i]; mn=1e18; mx=-1e18; sum=0; sum2=0; cnt=0;
              printf "%-26s", cfg;
              for (f=1; f<=nf; f++) {
                  v=val[cfg","f]; printf "%10.2f", v;
                  if (v<mn) mn=v; if (v>mx) mx=v; sum+=v; sum2+=v*v; cnt++;
              }
              m=sum/cnt; var=sum2/cnt-m*m; if (var<0) var=0;
              printf "%10.2f%9.2f%9.2f\n", py[cfg], mx-mn, sqrt(var);
              tot+=mx-mn;
          }
          printf "%s\n", "-------------------------------------------------------------------------------------------";
          printf "mean spread across seeds: %.2f pp over %d configs\n", tot/n, n;
          if (tot==0) printf "\n*** WARNING: all seeds identical -- binary ignores the seed ***\n";
      }' $FILES > "seeds_${M}.txt"
    echo "run times (seed order $SEEDS): $TIMES" >> "seeds_${M}.txt"
    echo "=== $M ==="
    cat "seeds_${M}.txt"
    echo
done

echo "saved: seeds_<mode>.txt   (raw runs kept in work_<mode>_<seed>/)"
