#!/usr/bin/env python3
"""
Benchmark script for C Evaluator Library
Similar to benchmark_suite_variations.py but uses C library
"""

import ctypes
import csv
import time
import os

# Load the C library
DATA_PATH = "raw-data/raw-autotuning-data/"
OUTPUT_DIR = "output/"

# Define C structures matching evaluator.h
class EvaluatorParams(ctypes.Structure):
    _fields_ = [
        ("HW", ctypes.c_char_p),
        ("file_name", ctypes.c_char_p),
        ("total_kernel_runs", ctypes.c_uint64),
        ("hist_HW", ctypes.c_char_p),
        ("k", ctypes.c_double),
        ("overhead", ctypes.c_uint64),
        ("fit_start", ctypes.c_uint64),
        ("number_of_tests", ctypes.c_uint64),
    ]

class EvaluatorResult(ctypes.Structure):
    _fields_ = [
        ("avg_extra_runtime", ctypes.c_double),
        ("std_extra_runtime", ctypes.c_double),
        ("avg_crystal_ball", ctypes.c_double),
        ("std_crystal_ball", ctypes.c_double),
        ("avg_estimate", ctypes.c_double),
        ("std_estimate", ctypes.c_double),
        ("avg_miss", ctypes.c_double),
    ]

# Load library
lib = ctypes.CDLL("./libevaluator.so")
lib.evaluator_run.argtypes = [ctypes.POINTER(EvaluatorParams), ctypes.POINTER(EvaluatorResult)]

def run_evaluator(HW, file_name, total_kernel_runs, hist_HW, k, overhead, fit_start, number_of_tests):
    """Call the C evaluator function"""
    params = EvaluatorParams(
        HW=HW.encode(),
        file_name=file_name.encode(),
        total_kernel_runs=total_kernel_runs,
        hist_HW=hist_HW.encode(),
        k=k,
        overhead=overhead,
        fit_start=fit_start,
        number_of_tests=number_of_tests
    )
    result = EvaluatorResult()
    lib.evaluator_run(ctypes.byref(params), ctypes.byref(result))
    return result

def experiment(file_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_file_name = OUTPUT_DIR + "benchmarking_c_" + file_name
    extra_result_file_name = OUTPUT_DIR + "extra_results_c_" + file_name
    
    write_file = True

    if write_file:
        with open(result_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["HW / # of runs / method / overhead", "Performance decline (%)"])
        with open(extra_result_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["HW / # of runs / method / overhead", "Average extra runtime", "Extra runtime deviation", "Average GT TS", "GT TS deviation", "Average estimated TS", "Estimated TS deviation", "Average miss (tuning steps)"])

    hist_HW = "680"
    # Full configuration (can be slow)
    hw_list = ["680", "750", "1070", "2080"]
    # hw_list = ["1070"]
    kernel_run_number_list = ["10000", "10000000"]
    k_values = ["1.0", "0.0", "0.5"]
    overhead_values = ["10000", "1000000"]
    hw_kernel_combinations = []

    for a in hw_list:
        for b in kernel_run_number_list:
            for c in k_values:
                for d in overhead_values:
                    benchmark = file_name.replace("_output.csv", "")
                    if os.path.exists(DATA_PATH + benchmark + "/" + a + "-" + file_name):
                        hw_kernel_combinations.append([a, b, c, d])

    start_time = time.time()

    for hw_kernel_combination in hw_kernel_combinations:
        HW, total_kernel_runs_str, k_str, overhead_str = hw_kernel_combination
        total_kernel_runs = int(total_kernel_runs_str)
        k = float(k_str)
        overhead = int(overhead_str)
        
        fit_start = 10
        number_of_tests = 1000

        result = run_evaluator(HW, file_name, total_kernel_runs, hist_HW, k, overhead, fit_start, number_of_tests)

        print("***************************")
        print(hw_kernel_combination)
        print("***************************")
        print("Prediction:")
        print(str(round(100 * result.avg_extra_runtime, 2)) + "% performance decline")
        print("Standard deviation: " + str(result.std_extra_runtime))
        print("Average ground truth TS: " + str(result.avg_crystal_ball))
        print("Standard deviation: " + str(result.std_crystal_ball))
        print("Average estimate: " + str(result.avg_estimate))
        print("Standard deviation: " + str(result.std_estimate))
        print("Average miss: " + str(result.avg_miss))

        print("\nTime:")
        print(time.time() - start_time)
        print("***************************\n\n\n")

        if write_file:
            row = []
            row.append(hw_kernel_combination)
            row.append(100 * result.avg_extra_runtime)
            
            with open(result_file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            extrarow = []
            extrarow.append(hw_kernel_combination)
            extrarow.append(result.avg_extra_runtime)
            extrarow.append(result.std_extra_runtime)
            extrarow.append(result.avg_crystal_ball)
            extrarow.append(result.std_crystal_ball)
            extrarow.append(result.avg_estimate)
            extrarow.append(result.std_estimate)
            extrarow.append(result.avg_miss)

            with open(extra_result_file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(extrarow)

experiment("gemm-reduced_output.csv")
