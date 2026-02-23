# Python Auto-tuning Evaluator

Monte Carlo simulation for GPU kernel auto-tuning performance estimation.

## Requirements

```bash
pip install numpy pandas
```

## Usage

```bash
python benchmark_suite_variations.py
```

## Files

- `benchmark_suite_variations.py` - Main benchmark script
- `evaluator.py` - Evaluator module
- `budget_estimator.py` - Budget estimation module
- `PYTHON_TUTORIAL.md` - Detailed documentation

## Configuration

Edit `benchmark_suite_variations.py` to change:
- Hardware list: `hw_list = ["680", "1070", "2080"]`
- Kernel runs: `kernel_run_number_list = ["10000", "10000000"]`
- k values: `k_values = ["1.0", "0.0", "0.5"]`
- Overhead: `overhead_values = ["10000", "1000000"]`

## Algorithm

Uses Monte Carlo simulation to estimate optimal GPU kernel tuning length:

1. **Random sampling** - Sample tuning runtimes from historical data
2. **Best-so-far** - Compute cumulative minimum at each step
3. **Cost model** - Total = Tuning Cost + Kernel Cost
4. **Crystal ball** - Find theoretical optimal stopping point
5. **Prediction** - Compare with curve fitting estimate

See `PYTHON_TUTORIAL.md` for detailed explanation.
