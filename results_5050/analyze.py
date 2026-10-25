import ctypes
import glob
import os
import re

import numpy as np
import pandas as pd

SESSIONS_DIR = "results_5050/sessions_1000_v2"
REFERENCE_CSV = "results_5050/ref_5050_run2/5050-reference_steps.csv"
TUNER_LIBRARY = "libtuner.so"

TOTAL_RUNS = 10000
OVERHEAD_US = 68220
FIT_START = 10
HISTORY_GPU = b"5050"
HISTORY_FILE = b"gemm-reduced_output.csv"
HISTORY_TESTS = 1000

SIMULATED_SESSIONS = 1000
CONFIGS_PER_SESSION = 2000
RANDOM_SEED = 1

MODES = ["live", "hybrid", "hist"]
FIRST_DECISION_STEP = FIT_START + 5 + 2

SESSION_FOLDER = re.compile(r".*/(live|hybrid|hist)_(\d+)/")

US_PER_S = 1e6       # microseconds per second
US_PER_MS = 1e3      # microseconds per millisecond

# comparison table: header and rows share one layout, so the columns line up
TABLE_HEADER = (f"{'mode':7} {'':8} {'TS med':>8} {'TS mean':>9} {'best ms':>8} "
                f"{'cost s':>8} {'decl fixed %':>13} {'perf decline %':>17}")
TABLE_ROW = ("{mode:7} {label:8} {stop_median:8.0f} {stop_mean:9.1f} {best_ms:8.2f} "
             "{cost_s:8.1f} {vs_fixed:12.1f}% {vs_hindsight:16.1f}%")


def total_cost_us(tuning_us, steps_tuned, best_kernel_us):
    production_runs = TOTAL_RUNS - steps_tuned
    return tuning_us + production_runs * best_kernel_us


def read_one_session(path):
    match = SESSION_FOLDER.match(path)
    if match is None:
        raise ValueError(f"not a session folder: {path}")

    mode, run = match.groups()
    steps = pd.read_csv(path)

    steps_tuned = len(steps)
    best_kernel_us = steps.kernel_us.min()
    tuning_us = steps.total_us.sum()
    cost_us = total_cost_us(tuning_us, steps_tuned, best_kernel_us)
    overhead_without_first_step = steps.overhead_us.iloc[1:]
    ended_by_library = steps.budget.iloc[-1] == 0

    return {
        "mode": mode,
        "run": int(run),
        "stop": steps_tuned,
        "best": best_kernel_us,
        "tune_s": tuning_us / US_PER_S,
        "cost_s": cost_us / US_PER_S,
        "oh_med": overhead_without_first_step.median(),
        "bad": not ended_by_library,
    }


def read_real_sessions():
    # it will match on sessions_1000_v2 directory and its content
    paths = glob.glob(SESSIONS_DIR + "/*/tuning_steps_*.csv")
    paths = [path for path in paths if "/test_" not in path]
    return pd.DataFrame([read_one_session(path) for path in paths])


class CurveParams(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double),
                ("b", ctypes.c_double),
                ("c", ctypes.c_double)]


# Python copy of KernelConfig from src/tuner_api.h: same fields, same order, same types.
class KernelConfig(ctypes.Structure):
    _fields_ = [("total_kernel_runs", ctypes.c_uint64),
                ("overhead", ctypes.c_uint64),
                ("fit_start", ctypes.c_uint64),
                ("mode", ctypes.c_int),               # TunerMode enum
                ("k", ctypes.c_double),
                ("hist_HW", ctypes.c_char_p),
                ("file_name", ctypes.c_char_p),
                ("hist_number_of_tests", ctypes.c_uint64)]


# TunerMode values from tuner_api.h, and the k of each mode
TUNER_MODES = {
    "live":   {"mode": 0, "k": 1.0},
    "hybrid": {"mode": 2, "k": 0.5},
    "hist":   {"mode": 1, "k": 0.0},
}


class Tuner:
    # The deployed tuner_api (libtuner.so), used exactly like ClTuneGemm uses it.
    def __init__(self, library_path):
        u64 = ctypes.c_uint64
        double = ctypes.c_double
        text = ctypes.c_char_p
        handle = ctypes.c_void_p

        self.lib = ctypes.CDLL(os.path.abspath(library_path))

        self.lib.initiate_kernel.restype = handle
        self.lib.initiate_kernel.argtypes = [ctypes.POINTER(KernelConfig), text, ctypes.c_void_p]
        self.lib.push_result.argtypes = [handle, double, double]
        self.lib.find_number_of_steps_that_should_be_tuning.restype = u64
        self.lib.find_number_of_steps_that_should_be_tuning.argtypes = [handle]
        self.lib.reset_kernel.argtypes = [handle]
        self.lib.release_kernel.argtypes = [handle]

        # only for printing O_hist and a, b
        self.lib.history_run.restype = u64
        self.lib.history_run.argtypes = [text, text, u64, u64, u64]
        self.lib.get_regression_params.argtypes = [
            text, text, u64, u64, u64, ctypes.POINTER(CurveParams)]

    def create_handle(self, mode_name):
        settings = TUNER_MODES[mode_name]
        config = KernelConfig(
            total_kernel_runs=TOTAL_RUNS,
            overhead=OVERHEAD_US,
            fit_start=FIT_START,
            mode=settings["mode"],
            k=settings["k"],
            hist_HW=HISTORY_GPU,
            file_name=HISTORY_FILE,
            hist_number_of_tests=HISTORY_TESTS,
        )
        return self.lib.initiate_kernel(ctypes.byref(config), mode_name.encode(), None)

    def stop_step(self, handle, kernels_us, totals_us):
        # Push one step at a time and ask after every step, like ClTuneGemm.
        # Returns the 1-based step where the library says stop (or the last step).
        self.lib.reset_kernel(handle)
        for step, (kernel, total) in enumerate(zip(kernels_us, totals_us), start=1):
            self.lib.push_result(handle, kernel, total)
            if self.lib.find_number_of_steps_that_should_be_tuning(handle) == 0:
                return step
        return len(kernels_us)

    def o_hist(self):
        return self.lib.history_run(
            HISTORY_GPU, HISTORY_FILE, TOTAL_RUNS, OVERHEAD_US, HISTORY_TESTS)

    def frozen_curve(self):
        curve = CurveParams()
        self.lib.get_regression_params(
            HISTORY_GPU, HISTORY_FILE, TOTAL_RUNS, FIT_START, HISTORY_TESTS,
            ctypes.byref(curve))
        return curve


def print_history(tuner):
    o_hist = tuner.o_hist()
    curve = tuner.frozen_curve()
    print(f"history (5050, OH={OVERHEAD_US}): O_hist={o_hist}  "
          f"a={curve.a:.3f} b={curve.b:.0f}")


def read_reference_sweep():
    # Every config measured once on the GPU: its kernel time and its real per-step total.
    # Step 1 is dropped: its overhead includes one-off setup.
    sweep = pd.read_csv(REFERENCE_CSV).iloc[1:]
    return sweep.kernel_us.values, sweep.total_us.values


def cost_for_every_stop(kernels, totals):
    # Total cost (us) of stopping after each step 1..N: tuning so far + (X - s) * best kernel so far.
    # Tuning so far uses the measured per-step totals (kernel + real overhead).
    steps = np.arange(1, len(kernels) + 1)
    tuning_us = np.cumsum(totals)
    best_so_far = np.minimum.accumulate(kernels)
    return total_cost_us(tuning_us, steps, best_so_far), best_so_far


def simulate(tuner, sweep_kernels, sweep_totals):
    print_history(tuner)
    handles = {name: tuner.create_handle(name) for name in MODES}
    rng = np.random.default_rng(RANDOM_SEED)

    offline_rows = [] # one row per session and mode, 3000 in total.
    oracle_rows = [] # one (best cost, best stop) pair per session, 1000 in total.
    summed_cost_curve = np.zeros(CONFIGS_PER_SESSION) # 2000 zeros. Each session’s cost curve gets added in here, for the average later.

    for _ in range(SIMULATED_SESSIONS):
        # one fake session: random configs without repeats, like KTT's random searcher
        order = rng.choice(len(sweep_kernels), CONFIGS_PER_SESSION, replace=False)
        kernels = sweep_kernels[order]
        totals = sweep_totals[order]
        cost, best_so_far = cost_for_every_stop(kernels, totals)
        summed_cost_curve += cost

        # oracle: best possible stop for this session, with hindsight
        oracle_cost = cost.min()
        oracle_stop = cost.argmin() + 1
        oracle_rows.append((oracle_cost, oracle_stop))

        # Ask the deployed tuner_api, step by step, where each mode (live, hybrid, hist) stops,
        # then store that stop, the best kernel found by then, its total cost and the oracle's cost.
        for name, handle in handles.items():
            stop = tuner.stop_step(handle, kernels, totals)
            offline_rows.append({
                "mode": name,
                "stop": stop,
                "best": best_so_far[stop - 1],
                "cost_s": cost[stop - 1] / US_PER_S,
                "orc_s": oracle_cost / US_PER_S,
            })

    for handle in handles.values():
        tuner.lib.release_kernel(handle)

    # average curve over all sessions; its minimum is the best fixed stop
    mean_cost_curve = summed_cost_curve / SIMULATED_SESSIONS
    oracle = pd.DataFrame(oracle_rows, columns=["cost_us", "stop"])
    return pd.DataFrame(offline_rows), oracle, mean_cost_curve


def percent_worse(cost, reference):
    return (cost / reference - 1) * 100


def print_comparison(real, offline, oracle, mean_cost_curve):
    hindsight_cost = oracle.cost_us.mean() / US_PER_S
    hindsight_stop = oracle.stop.mean()
    fixed_cost = mean_cost_curve.min() / US_PER_S
    fixed_stop = mean_cost_curve.argmin() + 1

    print(f"oracle (GT): mean cost {hindsight_cost:.1f} s "
          f"(average GT TS {hindsight_stop:.0f}) | "
          f"best fixed TS {fixed_stop} -> {fixed_cost:.1f} s")
    print()

    print(TABLE_HEADER)

    for mode in MODES:
        is_this_mode = real["mode"] == mode      # True/False for each row
        real_sessions = real[is_this_mode]       # keep only the True rows

        is_this_mode = offline["mode"] == mode    # True/False for each row
        offline_sessions = offline[is_this_mode]  # keep only the True rows

        for label, sessions in [("REAL", real_sessions), ("offline", offline_sessions)]:
            mean_cost = sessions.cost_s.mean()

            # % more expensive than always stopping at the best fixed step
            vs_fixed = percent_worse(mean_cost, fixed_cost)

            # % more expensive than a perfect stop
            if label == "offline":
                # each simulated session vs. its own oracle, then averaged
                vs_hindsight = percent_worse(sessions.cost_s, sessions.orc_s).mean()
            else:
                # real sessions have no oracle of their own: average vs. average oracle
                vs_hindsight = percent_worse(mean_cost, hindsight_cost)

            print(TABLE_ROW.format(
                mode=mode,
                label=label,
                stop_median=sessions.stop.median(),
                stop_mean=sessions.stop.mean(),
                best_ms=sessions.best.mean() / US_PER_MS,
                cost_s=mean_cost,
                vs_fixed=vs_fixed,
                vs_hindsight=vs_hindsight,
            ))
    print()


def stop_percentiles(sessions):
    return np.percentile(sessions.stop, [10, 50, 90]).round(0)


def share_stopped_at_first_decision(sessions):
    return (sessions.stop == FIRST_DECISION_STEP).mean().round(2)


def print_spread(real, offline):

    for mode in MODES:
        sessions = real[real["mode"] == mode]
        cost_sd = sessions.cost_s.std()
        cost_se = cost_sd / np.sqrt(len(sessions))
        print(mode, "REAL stop p10/p50/p90:", stop_percentiles(sessions),
              f" cost sd {cost_sd:.1f} s, se {cost_se:.2f} s",
              f" stopped at first decision ({FIRST_DECISION_STEP}):",
              share_stopped_at_first_decision(sessions))

    for mode in MODES:
        sessions = offline[offline["mode"] == mode]
        print(mode, "OFFL stop p10/p50/p90:", stop_percentiles(sessions),
              f" at {FIRST_DECISION_STEP}:", share_stopped_at_first_decision(sessions))


def print_session_overview(real):
    # Sanity check: sessions per mode, sessions not stopped by the library, median step overhead.
    sessions_per_mode = real.groupby("mode").size().to_dict()
    print("sessions per mode:", sessions_per_mode,
          " not ended by budget 0:", real.bad.sum())
    print(f"per-step overhead median over sessions: {real.oh_med.median():.0f} us")


def main():
    real = read_real_sessions()
    # merge them together into single .csv file (all 3000 of them if sessions_1000_v2)
    real.to_csv(os.path.join(SESSIONS_DIR, "..", "real_sessions.csv"), index=False)
    print_session_overview(real)

    # Kernel time and real per-step total (us) of every config; offline sessions are drawn from it.
    sweep_kernels, sweep_totals = read_reference_sweep()
    tuner = Tuner(TUNER_LIBRARY)
    offline, oracle, mean_cost_curve = simulate(tuner, sweep_kernels, sweep_totals)

    print_comparison(real, offline, oracle, mean_cost_curve)
    print_spread(real, offline)


if __name__ == "__main__":
    main()
