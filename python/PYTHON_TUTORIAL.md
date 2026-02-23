# Python Implementation - Tutorial

## Overview

This is the original Python implementation using pandas, numpy, and scipy for autotuning budget estimation.

## Dependencies

Install required Python packages:
```bash
pip install pandas numpy scipy
```

Or use the virtual environment:
```bash
source venv/bin/activate
pip install pandas numpy scipy
```

## Directory Structure

```
Diplomka/
├── benchmark_suite_variations.py    # Main benchmark script
├── evaluator.py                      # Evaluation functions
├── budget_estimator.py              # Budget estimation algorithms
├── evaluator.py                     # Original evaluator
├── benchmark_suite_variations.py     # Original benchmark runner
└── raw-data/raw-autotuning-data/   # CSV data files
```

## Quick Start

### Run benchmark
```bash
cd /home/kristian/Desktop/diplomka/Diplomka
python3 benchmark_suite_variations.py
```

### Use in your own code
```python
from evaluator import evaluator

# Run evaluation
avg_extra, std_extra, avg_gt, std_gt, avg_est, std_est, avg_miss = evaluator(
    HW="1070",
    file_name="gemm-reduced_output.csv",
    total_kernel_runs=10000,
    hist_HW="1070",
    k=1.0,              # 1.0 = regression only, 0.0 = history only, 0.5 = hybrid
    overhead=10000,
    fit_start=10,
    number_of_tests=1000
)

print(f"Performance decline: {avg_extra*100:.2f}%")
print(f"Ground truth TS: {avg_gt:.1f}")
print(f"Estimated TS: {avg_est:.1f}")
```

## Parameters Explained

| Parameter | Type | Description |
|-----------|------|-------------|
| `HW` | str | Hardware name (e.g., "1070", "2080") |
| `file_name` | str | CSV filename with tuning data |
| `total_kernel_runs` | int | Total number of kernel executions |
| `hist_HW` | str | Hardware for historical data (regression) |
| `k` | float | **1.0** = regression only, **0.0** = history only, **0<k<1** = hybrid |
| `overhead` | int | Tuning overhead per configuration (microseconds) |
| `fit_start` | int | Start curve fitting from this step |
| `number_of_tests` | int | Monte Carlo simulation iterations |

### k values:
- **`k = 1.0`**: Pure regression - uses curve fitting only
- **`k = 0.0`**: Pure history - uses historical stopping point
- **`k = 0.5`**: Hybrid - combines both approaches

## Module Functions

### evaluator.py

#### `evaluator(HW, file_name, total_kernel_runs, hist_HW, k, overhead, fit_start, number_of_tests)`

Main evaluation function that simulates autotuning and measures prediction accuracy.

**Returns:** `(avg_extra_runtime, std_extra_runtime, avg_crystal_ball, std_crystal_ball, avg_estimate, std_estimate, avg_miss)`

```python
from evaluator import evaluator

avg_extra, std_extra, avg_gt, std_gt, avg_est, std_est, avg_miss = evaluator(
    HW="1070",
    file_name="gemm-reduced_output.csv",
    total_kernel_runs=10000,
    hist_HW="1070",
    k=1.0,
    overhead=10000,
    fit_start=10,
    number_of_tests=1000
)
```

#### `history(HW, file_name, total_kernel_runs, overhead, number_of_tests)`

Simulates historical tuning to find optimal stopping point.

```python
from evaluator import history

optimal_steps = history(
    HW="1070",
    file_name="gemm-reduced_output.csv",
    total_kernel_runs=10000,
    overhead=10000,
    number_of_tests=1000
)
```

#### `get_history_regression_parameters(HW, file_name, total_kernel_runs, fit_start, number_of_tests)`

Fits power-law curve to historical tuning data.

```python
from evaluator import get_history_regression_parameters

a, b = get_history_regression_parameters(
    HW="1070",
    file_name="gemm-reduced_output.csv",
    total_kernel_runs=10000,
    fit_start=10,
    number_of_tests=1000
)
```

### budget_estimator.py

#### `tuning_length_recommendation(default_tuning_steps, tuning_run, HW, file_name, total_kernel_runs, regression_weight, fit_start, overhead, a, b)`

Recommends optimal tuning length based on curve fitting.

```python
from budget_estimator import tuning_length_recommendation

estimate = tuning_length_recommendation(
    default_tuning_steps=-1,
    tuning_run=sample_data,
    HW="1070",
    file_name="gemm-reduced_output.csv",
    total_kernel_runs=10000,
    regression_weight=1.0,
    fit_start=10,
    overhead=10000,
    a=-1,
    b=-1
)
```

## Running Different Benchmarks

### Modify benchmark_suite_variations.py

```python
# Change benchmark file
experiment("conv-output.csv")      # Convolution
experiment("nbody-output.csv")      # N-body
experiment("coulomb-output.csv")    # Coulomb

# Change hardware
hw_list = ["1070", "2080"]        # Multiple GPUs

# Change kernel runs
kernel_run_number_list = ["10000", "10000000"]

# Change k values
k_values = ["1.0", "0.0", "0.5"]
```

### Full example with custom parameters
```python
from evaluator import evaluator
import csv

# Test multiple configurations
results = []
configs = [
    {"HW": "1070", "k": 1.0, "overhead": 10000, "runs": 10000},
    {"HW": "1070", "k": 0.5, "overhead": 10000, "runs": 10000},
    {"HW": "1070", "k": 0.0, "overhead": 10000, "runs": 10000},
]

for cfg in configs:
    result = evaluator(
        HW=cfg["HW"],
        file_name="gemm-reduced_output.csv",
        total_kernel_runs=cfg["runs"],
        hist_HW="1070",
        k=cfg["k"],
        overhead=cfg["overhead"],
        fit_start=10,
        number_of_tests=100
    )
    results.append((cfg, result))

# Print results
for cfg, r in results:
    print(f"HW={cfg['HW']}, k={cfg['k']}: extra={r[0]*100:.2f}%")
```

## Data Format

CSV files should have columns:
- `Computation duration (us)` or `Kernel duration (us)` - Runtime in microseconds
- Additional configuration parameters

Example:
```csv
Kernel name,Computation duration (us),Global size,Local size,...
gemm_fast,11588,262144,64,...
gemm_fast,1956,262144,64,...
```

## Performance Notes

- **number_of_tests**: More tests = more accurate results but slower
- **fit_start**: Higher = more stable curve fitting but less data
- **overhead**: Lower overhead = shorter optimal tuning time

## Troubleshooting

### ModuleNotFoundError: No module named 'pandas'
```bash
pip install pandas numpy scipy
# or
source venv/bin/activate
```

### FileNotFoundError
- Ensure you're in the correct directory
- Check that `raw-data/raw-autotuning-data/` exists
- Verify CSV file exists: `ls raw-data/raw-autotuning-data/gemm-reduced/`

## Comparison with C Version

| Aspect | Python | C |
|--------|--------|---|
| Dependencies | pandas, numpy, scipy | GSL, libcsv |
| Speed | Slower | ~10-70x faster |
| Ease of use | Easier | More setup |

See `C_imple/` for the optimized C implementation.
