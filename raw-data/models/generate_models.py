#!/usr/bin/env python

"""Generate models (least squares regression nonlinear and decision tree) to predict performance counters from training data (does not need KTT)

Usage:
  ./generate_models.py --benchmark <benchmark>


Options:
  -h Show this screen.
  --benchmark   Benchmark type: GPU, GEMM
"""

from docopt import docopt
import os

def generateModel(problFrom, gpuFrom) :
    command = "mkdir generated-models"
    os.system(command)
    if problFrom[0].startswith("gemm") :
        directory = "gemm-reduced"
    else:
        directory = problFrom[0]
    command = "Rscript ./create_least_squares_models.R ../raw-autotuning-data/" + directory + "/" + gpuFrom[0] + "-" + problFrom[0] + "_output.csv "
    command = command + "generated-models/" + gpuFrom[0] + "-" + problFrom[0]
    command = command + " 4:" + str(4+problFrom[1]) + " 2,3," + str(4+problFrom[1]) + ":" + str(4+problFrom[1]+gpuFrom[4])
    print ("Executing " + command)
    os.system(command)

    command = "python3 ./generate_decision_tree_model.py -i ../raw-autotuning-data/" + directory + "/" + gpuFrom[0] + "-" + problFrom[0] + "_output.csv "
    command = command + "-t 4:" + str(4+problFrom[1]) + " -c 2,3," + str(4+problFrom[1]) + ":" + str(4+problFrom[1]+gpuFrom[4])
    print ("Executing " + command)
    os.system(command)
    command = "mv ../raw-autotuning-data/" + directory + "/" + gpuFrom[0] + "-" + problFrom[0] + "_output_DT.sav* generated-models/"
    print ("Executing " + command)
    os.system(command)

arguments = docopt(__doc__)

print("Creating models for all examples for different GPUs")

# (processor_name, computing_capability, profiling_counters)
processors = [["680", 3.0, 8, 1536, 35], ["750", 5.0, 4, 512, 41], ["1070", 6.1, 15, 1920, 43], ["2080", 7.5, 46, 2944, 38]]
# (problem_name, tuning_parameters, boundary)
problemsGeneral = [["coulomb", 8, '--compute_bound'], ["mtran", 9, '--memory_bound'], ["gemm-reduced", 15, '--compute_bound'], ["nbody", 8, '--compute_bound'], ["bicg", 11, '--memory_bound'], ["conv", 10, "--compute_bound"]]
problemsGemm = [["gemm-reduced", 15, '--compute_bound'], ["gemm-128-128-128", 15, '--compute_bound'], ["gemm-16-4096-4096", 15, '--memory_bound'], ["gemm-4096-16-4096", 15, '--memory_bound'], ["gemm-4096-4096-16", 15, '--compute_bound']]
problemsFrom = [["gemm-reduced", 15]]

if arguments['<benchmark>'] == "GPU" :
    for probl in problemsGeneral :
        for gpuFrom in processors :
            generateModel(probl, gpuFrom)
elif arguments['<benchmark>'] == "GEMM" :
    for problFrom in problemsGemm :
        for gpu in [processors[2]] :
            generateModel(problFrom, gpu)
else :
    print("Unknown benchmark, exiting.")
    exit()

