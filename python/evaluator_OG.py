import pandas as pd
import numpy as np
import budget_estimator

from scipy import optimize

# related to loop_limit in budget_estimator
curve_limit_max = 2000

def history(HW, file_name, total_kernel_runs, overhead, number_of_tests = 1000):
    full_file_path = "/home/jaro/tuning_budget/data/"+HW+"/"+file_name

    full_file = pd.read_csv(full_file_path)

    if "Computation duration (us)" in full_file:
        all_config_runtimes = full_file["Computation duration (us)"]
    else:
        all_config_runtimes = full_file["Kernel duration (us)"] 

    max_tuning_steps = min(total_kernel_runs,len(all_config_runtimes))

    historical_optimum = 0

    curve_limit = min(max_tuning_steps,curve_limit_max)
    for _ in range(number_of_tests):
        tuning_run = all_config_runtimes.sample(n = curve_limit).values
        best_so_far = []
        best_so_far.append(tuning_run[0])
    
        for i in range(curve_limit-1):
            best_so_far.append(min(best_so_far[i],tuning_run[i+1]))

        tuning_costs = np.cumsum(tuning_run,dtype=np.int64)
        tuning_overhead = np.arange(1,len(tuning_costs)+1,dtype=np.int64) * overhead
        tuning_costs = [x + y for x, y in zip(tuning_costs, tuning_overhead)]
        running_costs = list(reversed(range(total_kernel_runs+1-curve_limit,total_kernel_runs+1)))# [total_kernel_runs, total_kernel_runs-1, ...]
        running_costs = np.multiply(best_so_far, running_costs, dtype=np.int64)# vector product of best_so_far & running_costs
        total_runtimes = [x + y for x, y in zip(tuning_costs, running_costs)]# vector sum of tuning_costs & running_costs

        historical_optimum += total_runtimes.index(min(total_runtimes))

    return round(historical_optimum / number_of_tests) + 1 # at least 1 TS

#?
def gemm_history(file_name, total_kernel_runs, overhead):
    return history("gemm-reduced",file_name,total_kernel_runs,overhead)

def fitting_function(x,a,b,c):
    return b/(x**a)+c
def get_history_regression_parameters(HW, file_name, total_kernel_runs, fit_start, number_of_tests = 1000):
    full_file_path = "/home/jaro/tuning_budget/data/"+HW+"/"+file_name

    full_file = pd.read_csv(full_file_path)
     
    if "Computation duration (us)" in full_file:
        all_config_runtimes = full_file["Computation duration (us)"]
    else:
        all_config_runtimes = full_file["Kernel duration (us)"]

    max_tuning_steps = min(total_kernel_runs,len(all_config_runtimes))

    curve_limit = min(max_tuning_steps,curve_limit_max)

    avg_curve = []

    for i in range(number_of_tests):
        tuning_run = all_config_runtimes.sample(n = curve_limit).values

        if i == 0:
            avg_curve.append(tuning_run[0])
        else:
            avg_curve[0] = (avg_curve[0]*i + tuning_run[0]) / (i+1)

        running_min = tuning_run[0]
	
        for j in range(1,len(tuning_run)):
            if i == 0:
                avg_curve.append(min(tuning_run[j],avg_curve[-1]))
            else:
                running_min = min(tuning_run[j],running_min)
                avg_curve[j] = (avg_curve[j]*i + running_min) / (i+1)

    Xvalues = np.arange(len(avg_curve))
    Yvalues = avg_curve

    parameters = []
    parameters,_ = optimize.curve_fit(fitting_function, bounds=([0.1,-np.inf,-np.inf],[3,np.inf,np.inf]), xdata=Xvalues[fit_start:], ydata=Yvalues[fit_start:], maxfev=20000)

    a,b,c = parameters

    return a,b

"""    
def history_regression(tuning_run, a, b, total_kernel_runs, fit_start, overhead):
    max_tuning_steps = min(len(tuning_run),total_kernel_runs)
    loop_limit = min(max_tuning_steps,1000)
    best_config = tuning_run[0]
    best_configs_so_far = []
    best_configs_so_far.append(tuning_run[0])
    average_runtime_so_far = tuning_run[0]
    budget = loop_limit
    for i in range(1,loop_limit):
        budget -= 1

        average_runtimes_so_far = (average_runtime_so_far * i + tuning_run[i]) / (i+1)

    
    return budget

"""

def evaluator(HW, file_name, total_kernel_runs, hist_HW, k = 1, overhead = 0, fit_start = 15, number_of_tests = 1000):
    #number_of_tests = 1####    
    full_file_path = "/home/jaro/tuning_budget/data/"+HW+"/"+file_name

    full_file = pd.read_csv(full_file_path)
     
    if "Computation duration (us)" in full_file:
        all_config_runtimes = full_file["Computation duration (us)"]
    else:
        all_config_runtimes = full_file["Kernel duration (us)"]

    max_tuning_steps = len(all_config_runtimes)
         
    if (total_kernel_runs < len(all_config_runtimes)):
        max_tuning_steps = total_kernel_runs

    crystal_balls = []
    extra_runtimes = []
    estimates = []
    misses = []

    a = -1
    b = -1

    if (k > 0.0) and (k < 1.0):
        historical_tuning_steps = history(hist_HW,file_name,total_kernel_runs,overhead)
        
	# for GEMM experiments (with different matrix dimensions)
        #historical_tuning_steps = history(hist_HW,"1070-gemm-reduced_output.csv",total_kernel_runs,overhead)

    if (k == 0.0):
        a,b = get_history_regression_parameters(hist_HW, file_name, total_kernel_runs, fit_start)

    for _ in range(number_of_tests):
        curve_limit = min(max_tuning_steps,curve_limit_max)
        tuning_run = all_config_runtimes.sample(n = curve_limit).values
        best_so_far = []
        best_so_far.append(tuning_run[0])
    
        for i in range(curve_limit-1):
            best_so_far.append(min(best_so_far[i],tuning_run[i+1]))

        tuning_runtimes = np.cumsum(tuning_run,dtype=np.int64)
        tuning_overhead = np.arange(1,len(tuning_runtimes)+1,dtype=np.int64) * overhead
        tuning_costs = [x + y for x, y in zip(tuning_runtimes, tuning_overhead)]
        running_costs = list(reversed(range(total_kernel_runs+1-curve_limit,total_kernel_runs+1)))
        running_costs = np.multiply(best_so_far, running_costs, dtype=np.int64)# vector product of best_so_far & running_costs
        total_runtimes = [x + y for x, y in zip(tuning_costs, running_costs)]# vector sum of tuning_costs & running_costs

        crystal_ball = min(total_runtimes)

        if k == 1:
            estimate = budget_estimator.tuning_length_recommendation(-1, tuning_run, HW, file_name, total_kernel_runs, 1, fit_start, overhead)
        elif k == 0:
            estimate = budget_estimator.tuning_length_recommendation(-1, tuning_run, HW, file_name, total_kernel_runs, 1, fit_start, overhead, a, b)
        else:
            if False:
                estimate = historical_tuning_steps
            else:
                estimate = budget_estimator.tuning_length_recommendation(historical_tuning_steps, tuning_run, HW, file_name, total_kernel_runs, k, fit_start, overhead)

        extra_runtime = total_runtimes[estimate] / crystal_ball - 1

        crystal_balls.append(total_runtimes.index(crystal_ball)+1)#at least 1 TS
        extra_runtimes.append(extra_runtime)
        estimates.append(estimate)
        misses.append(abs(total_runtimes.index(crystal_ball) - estimate))

    avg_extra_runtime = sum(extra_runtimes)/len(extra_runtimes)
    std_extra_runtime = np.std(extra_runtimes)
    avg_crystal_ball = sum(crystal_balls)/len(crystal_balls)
    std_crystal_ball = np.std(crystal_balls)
    avg_estimate = sum(estimates)/len(estimates)
    std_estimate = np.std(estimates)
    avg_miss = sum(misses)/len(misses)

    return avg_extra_runtime, std_extra_runtime, avg_crystal_ball, std_crystal_ball, avg_estimate, std_estimate, avg_miss
