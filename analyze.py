import ctypes
import glob
import os
import re

import numpy as np
import pandas as pd

SESSIONS_DIR = "sessions_1000"
POOL_CSV = "raw-data/raw-autotuning-data/gemm-reduced/5050-gemm-reduced_output.csv"
ENGINE_LIBRARY = "libevaluator.so"

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


def total_cost_us(tuning_us, steps_tuned, best_kernel_us):
    production_runs = TOTAL_RUNS - steps_tuned
    return tuning_us + production_runs * best_kernel_us


def read_one_session(path):
    mode, run = re.match(r".*/(live|hybrid|hist)_(\d+)/", path).groups()
    steps = pd.read_csv(path)

    steps_tuned = len(steps)
    best_kernel_us = steps.kernel_us.min()
    tuning_us = steps.total_us.sum()
    cost_us = total_cost_us(tuning_us, steps_tuned, best_kernel_us)
    overhead_without_first_step = steps.overhead_us.iloc[1:]
    ended_by_library = steps.budget.iloc[-1] == 0

    return dict(
        mode=mode,
        run=int(run),
        stop=steps_tuned,
        best=best_kernel_us,
        tune_s=tuning_us / 1e6,
        cost_s=cost_us / 1e6,
        oh_med=overhead_without_first_step.median(),
        bad=not ended_by_library,
    )


def read_real_sessions():
    paths = glob.glob(SESSIONS_DIR + "/*/tuning_steps_*.csv")
    paths = [path for path in paths if "/test_" not in path]
    return pd.DataFrame([read_one_session(path) for path in paths])


class CurveParams(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double),
                ("b", ctypes.c_double),
                ("c", ctypes.c_double)]


class Engine:
    def __init__(self, library_path):
        u64 = ctypes.c_uint64
        double = ctypes.c_double
        text = ctypes.c_char_p

        self.lib = ctypes.CDLL(os.path.abspath(library_path))

        self.lib.recommend_tuning_length.restype = u64
        self.lib.recommend_tuning_length.argtypes = [
            u64, ctypes.POINTER(double), u64, u64, double, u64, u64, double, double]

        self.lib.history_run.restype = u64
        self.lib.history_run.argtypes = [text, text, u64, u64, u64]

        self.lib.get_regression_params.argtypes = [
            text, text, u64, u64, u64, ctypes.POINTER(CurveParams)]

    def o_hist(self):
        return self.lib.history_run(
            HISTORY_GPU, HISTORY_FILE, TOTAL_RUNS, OVERHEAD_US, HISTORY_TESTS)

    def frozen_curve(self):
        curve = CurveParams()
        self.lib.get_regression_params(
            HISTORY_GPU, HISTORY_FILE, TOTAL_RUNS, FIT_START, HISTORY_TESTS,
            ctypes.byref(curve))
        return curve

    def stop_step(self, kernels, mode):
        c_kernels = (ctypes.c_double * len(kernels))(*kernels)
        last_index = self.lib.recommend_tuning_length(
            mode["o_hist"], c_kernels, len(kernels), TOTAL_RUNS, mode["k"],
            FIT_START, OVERHEAD_US, mode["hist_a"], mode["hist_b"])
        return int(last_index) + 1


def mode_settings(engine):
    o_hist = engine.o_hist()
    curve = engine.frozen_curve()
    print(f"history (5050, OH={OVERHEAD_US}): O_hist={o_hist}  "
          f"a={curve.a:.3f} b={curve.b:.0f}")

    no_history = -1.0
    return {
        "live":   dict(k=1.0, o_hist=0,      hist_a=no_history, hist_b=no_history),
        "hybrid": dict(k=0.5, o_hist=o_hist, hist_a=no_history, hist_b=no_history),
        "hist":   dict(k=0.0, o_hist=0,      hist_a=curve.a,    hist_b=curve.b),
    }


def cost_for_every_stop(kernels):
    steps = np.arange(1, len(kernels) + 1)
    tuning_us = np.cumsum(kernels + OVERHEAD_US)
    best_so_far = np.minimum.accumulate(kernels)
    return total_cost_us(tuning_us, steps, best_so_far), best_so_far


def simulate(engine, pool):
    modes = mode_settings(engine)
    rng = np.random.default_rng(RANDOM_SEED)

    offline_rows = []
    oracle_rows = []
    summed_cost_curve = np.zeros(CONFIGS_PER_SESSION)

    for _ in range(SIMULATED_SESSIONS):
        kernels = rng.choice(pool, CONFIGS_PER_SESSION, replace=False)
        cost, best_so_far = cost_for_every_stop(kernels)
        summed_cost_curve += cost

        oracle_cost = cost.min()
        oracle_stop = cost.argmin() + 1
        oracle_rows.append((oracle_cost, oracle_stop))

        for name, settings in modes.items():
            stop = engine.stop_step(kernels, settings)
            offline_rows.append(dict(
                mode=name,
                stop=stop,
                best=best_so_far[stop - 1],
                cost_s=cost[stop - 1] / 1e6,
                orc_s=oracle_cost / 1e6,
            ))

    mean_cost_curve = summed_cost_curve / SIMULATED_SESSIONS
    return pd.DataFrame(offline_rows), np.array(oracle_rows), mean_cost_curve


def percent_worse(cost, reference):
    return (cost / reference - 1) * 100


def print_comparison(real, offline, oracle, mean_cost_curve):
    oracle_costs = oracle[:, 0]
    oracle_stops = oracle[:, 1]
    hindsight_cost = oracle_costs.mean() / 1e6
    fixed_cost = mean_cost_curve.min() / 1e6
    fixed_stop = mean_cost_curve.argmin() + 1

    print(f"oracle: hindsight mean cost {hindsight_cost:.1f} s "
          f"(mean stop {oracle_stops.mean():.0f}) | "
          f"best FIXED stop {fixed_stop} -> {fixed_cost:.1f} s")
    print()

    print(f"{'mode':7} {'':8} {'stop med':>8} {'stop mean':>9} {'best ms':>8} "
          f"{'cost s':>8} {'decl vs fixed':>13} {'decl vs hindsight':>17}")

    for mode in MODES:
        real_sessions = real[real["mode"] == mode]
        offline_sessions = offline[offline["mode"] == mode]

        for label, sessions in [("REAL", real_sessions), ("offline", offline_sessions)]:
            mean_cost = sessions.cost_s.mean()
            vs_fixed = percent_worse(mean_cost, fixed_cost)

            if label == "offline":
                vs_hindsight = percent_worse(sessions.cost_s, sessions.orc_s).mean()
            else:
                vs_hindsight = percent_worse(mean_cost, hindsight_cost)

            print(f"{mode:7} {label:8} {sessions.stop.median():8.0f} "
                  f"{sessions.stop.mean():9.1f} {sessions.best.mean() / 1e3:8.2f} "
                  f"{mean_cost:8.1f} {vs_fixed:12.1f}% {vs_hindsight:16.1f}%")
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
              " cost sd %.1f s, se %.2f s" % (cost_sd, cost_se),
              f" stopped at first decision ({FIRST_DECISION_STEP}):",
              share_stopped_at_first_decision(sessions))

    for mode in MODES:
        sessions = offline[offline["mode"] == mode]
        print(mode, "OFFL stop p10/p50/p90:", stop_percentiles(sessions),
              f" at {FIRST_DECISION_STEP}:", share_stopped_at_first_decision(sessions))


def print_session_overview(real):
    sessions_per_mode = real.groupby("mode").size().to_dict()
    print("sessions per mode:", sessions_per_mode,
          " not ended by budget 0:", real.bad.sum())
    print("per-step overhead median over sessions: %.0f us" % real.oh_med.median())


def main():
    real = read_real_sessions()
    real.to_csv(os.path.join(SESSIONS_DIR, "..", "real_sessions.csv"), index=False)
    print_session_overview(real)

    pool = pd.read_csv(POOL_CSV)["Computation duration (us)"].values
    engine = Engine(ENGINE_LIBRARY)
    offline, oracle, mean_cost_curve = simulate(engine, pool)

    print_comparison(real, offline, oracle, mean_cost_curve)
    print_spread(real, offline)


if __name__ == "__main__":
    main()
