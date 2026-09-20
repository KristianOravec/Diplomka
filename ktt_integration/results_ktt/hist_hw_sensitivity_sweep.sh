cd ~/Diplomka/python
for h in 680 750 1070 2080 5080; do
  echo "########## hist_hw = $h ##########"
  python3 ../results_ktt/monte_carlo_vs_oracle.py \
      --hw 5080 --hist-hw $h --mode hybrid --overhead 16600 \
      --trials 1000 --engine c --lib ../libevaluator.so 2>&1 | tail -26
done | tee ~/Diplomka/results_ktt/hist_hw_sweep.txt
