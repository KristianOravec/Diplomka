# BUILD & TEST GUIDE — tuning-budget estimator

Everything: C library, KTT integration, harness, offline tests. Commands are
copy-pasteable from a fresh clone; paths auto-detect or are overridable.

## Structure

The project is organized into the following directories:

- `src/`: C library source code
- `ktt_integration/`: ClTuneGemm.cpp with build_cltunegemm.sh -> core, some *.xml and *.csv tuning steps and references, python scripts which should test something but I already forgot
- `python/`: jarda's implementation (the "OG" variant is totally vanilla version)
- `testing/`: ktt_testing, tried to check whether KTT integration works as expected, testing for precompute_historical, ignore, and seed_testing results from roots `run_all_seeds_parallel.sh`
- `raw-data/`: Jarda's data
- `obsolete/`: ignore, just some old scripts that are no longer needed but kept just in case
- `output`: my output from `src/` engine
- `benchmark_c.py`: runs `src/` engine 

## 0. Prerequisites

```bash
sudo apt install build-essential libgsl-dev liblbfgs-dev
# premake5 is NOT in Ubuntu's repos; install the GitHub binary release:
curl -L https://github.com/premake/premake-core/releases/download/v5.0.0-beta8/premake-5.0.0-beta8-linux.tar.gz | tar -xz -C ~/.local/bin && chmod +x ~/.local/bin/premake5
export PATH=$PATH:~/.local/bin        # add ~/.local/bin to PATH (per-session)
# CUDA toolkit (WSL: toolkit ONLY, never the driver, I used WSL):
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install cuda-toolkit-12-9        # RTX 50xx needs >= 12.8 (sm_120)
# Python (3.12+): numpy, pandas, scipy
# PEP 668: Ubuntu 24+ refuses plain pip (externally-managed); --user --break-system-packages installs to the user site
python3 -m pip install --user --break-system-packages numpy pandas scipy
# KTT source:
git clone https://github.com/HiPerCoRe/KTT.git ~/KTT
```

## 1. Build KTT (once)

KTT master uses **premake5** (there is no CMakeLists.txt).

```bash
cd ~/KTT
export CUDA_PATH=/usr/local/cuda
premake5 gmake --no-opencl          # CUDA backend only; --no-opencl avoids CL/cl.h errors
cd Build && make config=release_x86_64 -j$(nproc)
# produces: Build/x86_64_Release/libktt.so
```

Sanity check: `~/KTT/Build/x86_64_Release/02KernelRunningCuda` should print
"Kernel run completed successfully".

## 2. Build the C library (libevaluator.so)

```bash
cd ~/Diplomka && make          # -> libevaluator.so (tuner_api is built into the KTT
    # integration separately; this lib serves the
    # harness + Python ctypes paths + src/precompute_historical.py)
```

## 3. Build the KTT integration binary (auto-detects paths, validates)

```bash
cd ~/Diplomka
KTT_ROOT=~/KTT ./ktt_integration/build_cltunegemm.sh
# append --test to also run the built-in GPU smoke validation
```

## 4. Run everything

Set up the environment once per shell:

```bash
cd ~/Diplomka
export LD_LIBRARY_PATH=~/KTT/Build/x86_64_Release:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 4.1 Real-hardware tuning sessions (the deployed path)

```bash
# <X> is required = total GEMM executions (tuning steps + production runs)
./ClTuneGemm 0 0 ~/KTT/Examples/ClTuneGemm/ClTuneGemm.cu \
    ~/KTT/Examples/ClTuneGemm/ClTuneGemmReference.cu <X> <mode> [histHW] [searcher] [fast] [overhead_us]

# modes:  live (k=1, no history) | hybrid (k=0.5, blend with history) | hist (k=0, frozen curve)
# histHW: which GPU's dataset teaches the estimator (default 680); ignored by live
# searcher: rand (random, deployment-style) [default rand] | det (fixed order, paired-stream; pass explicitly)
# fast: stop right after the tuning decision -- skips production runs (~50 min -> ~3 min)
# overhead_us: per-step cost fed to the HISTORICAL fit at init (default 0 = free steps)
#              set it to your measured overhead (see tuning_steps.csv) for realistic O_hist!

# Examples:
./ClTuneGemm 0 0 ~/KTT/Examples/ClTuneGemm/ClTuneGemm.cu \
    ~/KTT/Examples/ClTuneGemm/ClTuneGemmReference.cu 10000 hybrid 1070 rand "" 16600
./ClTuneGemm ... 500 ref                                # oracle diary (no tuner, N configs)
```

Outputs (auto-named with the full input signature, never overwrite each other):
`tuning_steps_<X>_<mode>_hist<hw>_<searcher>[_oh<oh>][_fast].csv`, `GemmOutput_<...>.xml`.

**Rule**: `pgrep -af ClTuneGemm` must be empty before launching; two GPU runs at
once produce garbage timings.

## After this were my desperate attempts/efforts to get information if it works but still not quite sure rip all wasted water in AI datacentres

### 4.2 Offline Monte-Carlo test of the decision engine (no GPU)

```bash
cd python
python3 ../results_ktt/monte_carlo_vs_oracle.py --hw 5080 --x 10000 \
    --mode hybrid --overhead 16600 --trials 1000 [--hist-hw 1070] [--seed 0]
# trials: draws random runs from the pool, runs the decision engine, scores
# each against the statistical oracle (expected-optimal play) and the
# per-sample oracle (harness definition). 1000 trials ~ 10-15 min.
```

Pool file needed at `raw-data/raw-autotuning-data/gemm-reduced/<hw>-gemm-reduced_output.csv`
(column `Computation duration (us)`; durations must be integers).

### 4.3 Cross-engine replay (Python engine vs C engine on real data)

```bash
cd ~/Diplomka
python3 cross_engine/cross_engine_test.py --x 200 1000 10000
# feeds the REAL measured stream (results_ktt/reference_steps_5788.csv) to both
# engines; 'match' column = port fidelity on real silicon data
```

### 4.4 Monte-Carlo harness sweeps (C deployed engine vs Python references)

```bash
./run_all_seeds_parallel.sh [trials] [modes] [seeds]   # e.g. 1000 "live hist hybrid" random
# 15 concurrent runs -> work_<mode>_<seed>/ + seeds_<mode>.txt tables
# per-run timers land in work_*/time.txt
```

## 5. Analyze results

```bash
python3 results_ktt/show_numbers.py                 # audit trail: every number -> file/row/column
python3 ktt_integration/compare_ktt_vs_work.py --X 10000 --mode hybrid \
    --oracle statistical \
    --estimator results_ktt/tuning_steps_<tag>.csv \
    --reference results_ktt/reference_steps_5788_rand.csv \
    --work results_ktt/mc_5080_hist1070.log         # or a work_*/result.txt
```

`--oracle stream` = paired det-stream referee; `--oracle statistical` =
expected-optimal play over an exhaustive pool (use for rand sessions).

## 6. Known gotchas

| Symptom | Cause / fix |
|---|---|
| `libktt.so: cannot open` | export LD_LIBRARY_PATH (see §4) |
| `CL/cl.h: No such file` | rebuild KTT with `--no-opencl` |
| C files fail under g++ (`void*` casts) | compile with `gcc -c` first (build script does this) |
| `-DKTT_CUDA_EXAMPLE` undefined | build script handles it; manual g++ must pass it |
| `Kernel duration (us)` KeyError in Python | pool must have integer durations under `Computation duration (us)` |
| kernel timings 10-25 s instead of ~1 s | another GPU process is running — kill it |
| Python engine crashes on float durations | known: Python engine assumes int data (np.int64); round the pool |
| Same config: 180 ms in one run, 13 ms in another | GPU power state changed (laptop AC power mode) — never mix eras; log power state per session |
