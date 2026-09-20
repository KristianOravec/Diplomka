#!/usr/bin/env python

"""Execute simulated searcher for testing GPU->GPU and input->input portability (does not need KTT)

Usage:
  ./autobench_gpu-gpu.py --benchmark <benchmark> --method <method> --experiments <experiments> --iterations <iterations>


Options:
  -h             Show this screen.
  --benchmark    Benchmark type: GPU, GEMM
  --method       Method for PC prediction: Exact, DecisionTree, LeastSquares
  --experiments  Number of experiments to perform.
  --iterations   Number of iterations per experiment.
"""

from docopt import docopt
import os

def runBenchmark(problFrom, problRun, gpuFrom, gpuRun, method, exps, iters) :
    if problFrom[0].startswith("gemm") :
        directory = "gemm-reduced"
    else:
        directory = problFrom[0]
    command = "python3  -W ignore ./simulated-profiling-searcher.py -o ../raw-autotuning-data/" + directory + "/" + gpuRun[0] + "-" + problRun[0] + "_output.csv --oc " + str(gpuRun[1]) + " --mp " + str(gpuRun[2]) + " --co " + str(gpuRun[3])
    if method == "Exact" :
        command = command + " --cm ../raw-autotuning-data/" + directory + "/" + gpuFrom[0] + "-" + problFrom[0] + "_output.csv"
    elif method == "DecisionTree" :
        command = command + " --dt ../models/decision-trees/" + directory + "/" + gpuFrom[0] + "-" + problFrom[0] + "_output_DT.sav"
    elif method == "LeastSquares" :
        command = command + " --ls ../models/least-squares-models/" + directory + "/" + gpuFrom[0] + "-" + problFrom[0]
    else :
        print("Unknown method, exiting.")
        exit()
    command = command + " --ic " + str(gpuFrom[1]) + " -p 1 -t 4:" + str(4+problFrom[1]) + " -c 2,3," + str(4+problFrom[1]) + ":" + str(4+problFrom[1]+gpuRun[4]) + " " + problRun[2] + " -e " + exps + " -i " + iters  + " > "
    if method == "Exact" :
        command = command + "benchmarks-exact"
    elif method == "DecisionTree" :
        command = command + "benchmarks-dt"
    elif method == "LeastSquares" :
        command = command + "benchmarks-ls"
    command = command + "/benchmark-" + problRun[0] + "-from-" + problFrom[0] + "-exe-" + gpuRun[0] + "-stat-" + gpuFrom[0] + ".log & "
    print ("Executing " + command)
    os.system(command)

arguments = docopt(__doc__)

print("Benchmarking all examples autotuning for cases of different GPU used for statistics")

# (processor_name, computing_capability, profiling_counters)
processors = [["680", 3.0, 8, 1536, 35], ["750", 5.0, 4, 512, 41], ["1070", 6.1, 15, 1920, 43], ["2080", 7.5, 46, 2944, 38]]
# (problem_name, tuning_parameters, boundary)
problemsGeneral = [["coulomb", 8, '--compute_bound'], ["mtran", 9, '--memory_bound'], ["gemm-reduced", 15, '--compute_bound'], ["nbody", 8, '--compute_bound'], ["conv", 10, "--compute_bound"]]
problemsGemm = [["gemm-reduced", 15, '--compute_bound'], ["gemm-128-128-128", 15, '--compute_bound'], ["gemm-16-4096-4096", 15, '--memory_bound'], ["gemm-4096-16-4096", 15, '--memory_bound'], ["gemm-4096-4096-16", 15, '--compute_bound']]
problemsFrom = [["gemm-reduced", 15]]

print (arguments)

if arguments['<benchmark>'] == "GPU" :
    for probl in problemsGeneral :
        for gpuRun in processors :
            for gpuFrom in processors :
                runBenchmark(probl, probl, gpuFrom, gpuRun, arguments['<method>'], arguments['<experiments>'], arguments['<iterations>'])
elif arguments['<benchmark>'] == "GEMM" :
    for problRun in problemsGemm :
        for problFrom in problemsGemm :
            for gpu in [processors[2]] :
                runBenchmark(problFrom, problRun, gpu, gpu, arguments['<method>'], arguments['<experiments>'], arguments['<iterations>'])
else :
    print("Unknown benchmark, exiting.")
    exit()

