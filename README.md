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
K=~/KTT/Examples/ClTuneGemm
./ClTuneGemm 0 0 $K/ClTuneGemm.cu $K/ClTuneGemmReference.cu \
    <X> [mode] [histHW] [fast] [overhead_us] [fit_start]
```

| argument | meaning | default |
|---|---|---|
| `X` | total GEMM executions (tuning steps + production runs), required | – |
| `mode` | `live` (no history), `hybrid` (blend with O_hist), `hist` (frozen a, b), `ref` (random search for X steps, no library) | `live` |
| `histHW` | GPU whose dataset provides the history; in `ref` mode the name of the measured GPU (required); ignored by `live` | `680` |
| `fast` | stop right after the tuning decision; production cost is computed from the CSV. Any other value (e.g. `-`) = not fast | not fast |
| `overhead_us` | per-step overhead for the historical values (O_hist, a, b); use the measured median | `0` |
| `fit_start` | warmup steps before the first curve fit | `10` |

The searcher is always random (no repeats). Output: `tuning_steps_<tag>.csv` in the current folder.

Run with the CUDA JIT cache disabled, so every step has a constant compile cost:
```bash
export CUDA_CACHE_DISABLE=1
```

Examples:
```bash


# one hybrid session, as in the experiments (RTX 5050, overhead 68.2 ms)
~/Diplomka/ClTuneGemm 0 0 ~/KTT/Examples/ClTuneGemm/ClTuneGemm.cu  ~/KTT/Examples/ClTuneGemm/ClTuneGemmReference.cu 10000 hybrid 5050 fast 68220

# reference sweep over the whole space (5788 configs)
~/Diplomka/ClTuneGemm 0 0 ~/KTT/Examples/ClTuneGemm/ClTuneGemm.cu ~/KTT/Examples/ClTuneGemm/ClTuneGemmReference.cu 5788 ref 5050
# many sessions overnight (modes interleaved, resumable)
./run_sessions.sh 1000 ~/sessions_1000
```
Outputs (named after the arguments, written to the current folder):

- tuning runs: `tuning_steps_<X>_<mode>_hist<hw>_rand_n4096_fs<fit_start>[_fast][_oh<overhead>].csv`
               and `GemmOutput_<tag>.xml` (KTT's own results)
- ref mode:    `<gpu>-gemm-reduced_output.csv` (pool in Jarda's format),
               `<gpu>-reference_steps.csv` (step log with overhead), `<gpu>-GemmOutput.xml`

Runs with the same arguments in the same folder overwrite each other,
so use one folder per session (as `run_sessions.sh` does).
