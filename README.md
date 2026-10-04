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
- `benchmark_c.py`: runs `src/` engine, C mirror to jarda's benchmark_suite_variations.py 

### Historical cache

- `src/historical_cache.c/.h` – part of the C engine; looks up precomputed
  historical values instead of computing them at startup.
- `src/precompute_historical.py` – generates `historical_params.csv`
  (rerun after changing `curvefit.c` / `evaluator.c`).
- Tests:
  - `testing/test_historical_cache.c` – lookup rules (hit/miss edge cases)
  - `testing/test_cache_offline.py` – same results with and without the table

Row kinds in `historical_params.csv`:
- `REG` – frozen curve a, b for hist mode (valid for any X >= 2000)
- `OPT` – O_hist for hybrid mode (exact X and overhead)



## 0. Prerequisites

```bash
sudo apt install build-essential libgsl-dev liblbfgs-dev
# premake5 is NOT in Ubuntu's repos; install the GitHub binary release:
curl -L https://github.com/premake/premake-core/releases/download/v5.0.0-beta8/premake-5.0.0-beta8-linux.tar.gz | tar -xz -C ~/.local/bin && chmod +x ~/.local/bin/premake5

# JUST USE VENV PLS 
python3 -m pip install numpy pandas scipy
# KTT source (use development branch):
git clone https://github.com/HiPerCoRe/KTT.git ~/KTT 
```
Content of .bashrc file on Airacuda:
```bash
export CUDA_PATH=/usr/local/cuda
export CUDA_CACHE_DISABLE=1 # experiment, maybe omit ?? ask Jirka
export LD_LIBRARY_PATH=$HOME/KTT/Build/x86_64_Release:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```
## 1. Build KTT (once)
Check README.md u can do it  https://github.com/HiPerCoRe/KTT


## 2. Build the C library (libevaluator.so)

```bash
cd ~/Diplomka && make          # -> libevaluator.so (tuner_api is built into the KTT
```

## 3. Build the KTT integration binary (auto-detects paths, validates)

KTT_ROOT hardcoded in `ktt_integration/build_cltunegemm.sh`.
```bash
cd ~/Diplomka
./ktt_integration/build_cltunegemm.sh
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
