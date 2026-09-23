# Parallelized version of the benchmark suite.
#
# ONLY CHANGE vs the serial original: the (HW x runs x k x overhead) configs are
# dispatched across CPU cores instead of running one after another. The configs
# are fully independent, so results are identical -- same evaluator, same
# budget_estimator, same maxfev, same math, same CSV output.

# Keep BLAS single-threaded inside each worker: otherwise numpy/scipy spawn their
# own threads in every process and oversubscribe the CPU, which can end up SLOWER.
# Must be set before numpy is imported, so this stays at the very top.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from evaluator import evaluator
import csv
import time
from os.path import exists
from multiprocessing import Pool

# Ryzen 7 5800HS: 8 physical cores. Use 8, not 16 -- SMT gives little on tight
# numeric loops and can hurt via cache contention.
# FIX for i7 14650hx
WORKERS = 16

DATA_PATH = "../raw-data/raw-autotuning-data/"


def hw_id(hw):
    """'GPU-680' -> '680'. The HW names keep the GPU- prefix as labels (as in the
    original), but the data files are named <id>-<file_name>, so strip it when
    building paths."""
    return hw.replace("GPU-", "")


def data_file(hw, file_name):
    """Full path to one hardware's data file."""
    benchmark = file_name.replace("_output.csv", "")
    return DATA_PATH + benchmark + "/" + hw_id(hw) + "-" + file_name


def run_one(job):
    """Run ONE configuration. Executed in a worker process (no shared state)."""
    hw_kernel_combination, file_name, hist_HW, fit_start = job
    HW, total_kernel_runs_str, k_str, overhead_str = hw_kernel_combination
    total_kernel_runs = int(total_kernel_runs_str)
    k = float(k_str)
    overhead = int(overhead_str)

    # evaluator() takes the bare id (it builds the same <id>-<file> path itself)
    results = evaluator(hw_id(HW), file_name, total_kernel_runs, hw_id(hist_HW),
                        k, overhead, fit_start)
    return hw_kernel_combination, results


def experiment(file_name):
    result_file_name = "benchmarking_" + file_name + "_1070_100-1000"
    extra_result_file_name = "extra_results_" + file_name + "_1070_100-1000"

    write_file = True

    if write_file:
        with open(result_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["HW / # of runs / method / overhead", "Performance decline (%)"])
        with open(extra_result_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["HW / # of runs / method / overhead", "Average extra runtime", "Extra runtime deviation", "Average GT TS", "GT TS deviation", "Average estimated TS", "Estimated TS deviation", "Average miss (tuning steps)"])

    #Comparing various matrix dimensions in GEMM
    #hist_HW = "gemm-reduced"
    #hw_list = ["gemm-reduced"]
    #file_name_list = ["1070-gemm-16-4096-4096_output.csv", "1070-gemm-128-128-128_output.csv", "1070-gemm-4096-16-4096_output.csv", "1070-gemm-4096-4096-16_output.csv", "1070-gemm-reduced_output.csv"]

    hist_HW = "GPU-1070" # the HW used to set the estimate
    hw_list = ["GPU-750", "GPU-1070", "GPU-2080"]
    #file_name_list = ["gemm_batch_output.csv","gemm_output.csv","conv_output.csv","reduction_output.csv"]

    kernel_run_number_list = ["10000","10000000"]

    k_values = ["1.0","0.0","0.5"]
    overhead_values = ["10000","1000000"]
    hw_kernel_combinations = []

    fit_start = 10

    for a in hw_list:
#        for b in file_name_list:
        for b in kernel_run_number_list:
            for c in k_values:
                for d in overhead_values:
                    if exists(data_file(a, file_name)):
                        hw_kernel_combinations.append([a,b,c,d])

    if not hw_kernel_combinations:
        print("No configurations (no data files found).")
        return

    workers = min(WORKERS, len(hw_kernel_combinations))
    print("Running %d configs on %d workers...\n" % (len(hw_kernel_combinations), workers))

    jobs = [(combo, file_name, hist_HW, fit_start) for combo in hw_kernel_combinations]

    start_time = time.time()
    done = 0

    # Configs run in parallel; imap_unordered yields each as it completes, so
    # progress prints appear promptly instead of all at the end.
    with Pool(processes=workers) as pool:
        for hw_kernel_combination, results in pool.imap_unordered(run_one, jobs):
            (average_extra_runtime, std_extra_runtime, avg_crystal_ball,
             std_crystal_ball, avg_estimate, std_estimate, avg_miss) = results

            done += 1

            print("***************************")
            print(hw_kernel_combination)
            print("***************************")
            print("Prediction:")
            print(str(round(100 * average_extra_runtime,2))+"% performance decline (compared with a crystal ball prediction)")
            print("Standard deviation: "+str(std_extra_runtime))
            print("Average ground truth TS: "+str(avg_crystal_ball))
            print("Standard deviation: "+str(std_crystal_ball))
            print("Average estimate: "+str(avg_estimate))
            print("Standard deviation: "+str(std_estimate))
            print("Average miss: "+str(avg_miss))

            print("\nTime:")
            print(time.time()-start_time)
            print("(%d/%d done)" % (done, len(hw_kernel_combinations)))
            print("***************************\n\n\n")

            # Writes happen in the PARENT process only -> no file races.
            if write_file:
                row = []
                row.append(hw_kernel_combination)
                row.append(100 * average_extra_runtime)

                with open(result_file_name,'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                extrarow = []
                extrarow.append(hw_kernel_combination)
                extrarow.append(average_extra_runtime)
                extrarow.append(std_extra_runtime)
                extrarow.append(avg_crystal_ball)
                extrarow.append(std_crystal_ball)
                extrarow.append(avg_estimate)
                extrarow.append(std_estimate)
                extrarow.append(avg_miss)

                with open(extra_result_file_name,'a',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(extrarow)

    print("Done: %d configs in %.1fs on %d workers" %
          (len(hw_kernel_combinations), time.time()-start_time, workers))


if __name__ == "__main__":
    # Required for multiprocessing: without this guard each worker would
    # re-execute the module top level and spawn workers recursively.
    experiment("gemm-reduced_output.csv")
