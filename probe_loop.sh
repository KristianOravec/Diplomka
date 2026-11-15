#!/usr/bin/env bash

OUT=~/Diplomka/results_5050/sessions_probe_v1
KERNELS=~/KTT/Examples/ClTuneGemm
SESSIONS=1000

for mode in live hybrid hist; do
  for i in $(seq 1 $SESSIONS); do
    dir=$OUT/${mode}_$i
    if ls $dir/tuning_steps_*_probe.csv >/dev/null 2>&1 && grep -q "probe done" $dir/run.log 2>/dev/null; then
      continue                              # this session is already done, if some catastrophic failure or restart so we can continue
    fi
    mkdir -p $dir && ln -sfn ~/Diplomka/raw-data $dir/raw-data
    (cd $dir && ~/Diplomka/ClTuneGemm 0 0 $KERNELS/ClTuneGemm.cu $KERNELS/ClTuneGemmReference.cu \
        10000 $mode 5050 - 68220 10 probe > run.log 2>&1)
    echo "$(date +%H:%M) $mode $i/$SESSIONS"
  done
done
echo "all done"
