#!/usr/bin/env bash
# overhead per step in 10 ms buckets: CUDA cache ON vs OFF

echo "== cache ON"
awk -F, 'NR>2 {b=int($3/10000)*10; c[b]++} END {for (k in c) print k" ms:", c[k]}' \
	~/Diplomka/results_5050/cache_on/reference_steps_5788_ref_*.csv | sort -n

echo
echo "== cache OFF"
awk -F, 'NR>2 {b=int($3/10000)*10; c[b]++} END {for (k in c) print k" ms:", c[k]}' \
    ./ref_nocache_full/reference_steps_5788_ref_*.csv | sort -n
