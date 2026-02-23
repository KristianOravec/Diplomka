import pandas as pd
import numpy as np
from scipy import optimize
import warnings


# Page 29, 3.2, just before beginnign of 3.3 hybrid approach
# x -> tunning krok
# a -> strmost (steepness) funkcie
# b -> ako funkcia skaluje
# c -> hranica funkcie na ose x 
def fitting_function(x,a,b,c):
    return b/(x**a)+c
    #return a/x+b


#Page 27 T(x) 
# x, a, b, c -> parametre pre fitting funkciu
# starting step -> na ktorom kroku autotunningu je
# total_steps -> 
def total_runtime_remaining(x,a,b,c,starting_step,total_steps,average_runtime_so_far,overhead):
    return (average_runtime_so_far + overhead)*x + fitting_function(starting_step+x,a,b,c)*(total_steps-starting_step-x)


# funkncia je volana v evaluator.py
def tuning_length_recommendation(default_tuning_steps, tuning_run, HW, file_name, total_kernel_runs, regression_weight, fit_start, overhead,a=-1,b=-1):

    max_tuning_steps = min(total_kernel_runs,len(tuning_run))
    
    budget = max_tuning_steps
    best_config = tuning_run[0]
    best_configs_so_far = []
    best_configs_so_far.append(tuning_run[0])
    average_runtime_so_far = tuning_run[0]

    # related to curve_limit_max in evaluator
    loop_limit = min(max_tuning_steps,2000)
    for i in range(1,loop_limit):
        budget -= 1
        
        average_runtime_so_far = (average_runtime_so_far * i + tuning_run[i]) / (i+1)

        #tuning can't be stopped earlier than after FS+5 steps [not great for easy tuning spaces]
        if i > (fit_start+5):
            new_budget = local_budget_estimation(i, total_kernel_runs, average_runtime_so_far, best_configs_so_far, fit_start, overhead, a, b)

            if default_tuning_steps >= 0:
                regression_weight = min(1,i/default_tuning_steps)
                historical_budget = default_tuning_steps-i
                new_budget = regression_weight * new_budget + (1-regression_weight) * historical_budget
                
            if tuning_run[i] < best_config:
                budget = new_budget
            elif new_budget < budget:
                budget = new_budget

        best_config = min(best_config,tuning_run[i])
        best_configs_so_far.append(best_config)

        if (budget < 1):
            return i

    print("Oh snap! The loop limit has been reached, which probably means that the ideal budget is more than 2000 steps! (which is weird)")

    return min(3000,max_tuning_steps-1)

def local_budget_estimation(current_tuning_step, total_tuning_steps, average_runtime_so_far, best_configs_so_far, fit_start, overhead, hist_a, hist_b): # arguments: current tuning step #, best_so_far list, # of total kernel runs

    abcx = True #True: b/x^a+c; False: a/x+b asi ide o vyber jednoduchsieho resp zlozitejsieho fitting_function vzorca?


    Xvalues = np.arange(len(best_configs_so_far))
    Yvalues = best_configs_so_far

    parameters = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if abcx:
            if (hist_a == -1) and (hist_b == -1):
                parameters,_ = optimize.curve_fit(fitting_function, bounds=([0.1,-np.inf,-np.inf],[3,np.inf,np.inf]), xdata=Xvalues[fit_start:], ydata=Yvalues[fit_start:], maxfev=30000)
                a,b,c = parameters
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    #parameters,_ = optimize.curve_fit(lambda x, b, c: fitting_function(x,hist_a,b,c), xdata=Xvalues[fit_start:], ydata=Yvalues[fit_start:], maxfev=20000) # if abcx == False
                    parameters,_ = optimize.curve_fit(lambda x, c: fitting_function(x,hist_a,hist_b,c), xdata=Xvalues[fit_start:], ydata=Yvalues[fit_start:], maxfev=30000)
                a = hist_a
                #b = parameters[0] # if abcx == False
                b = hist_b
                #c = parameters[1] # if abcx == False
                c = parameters[0]
        else:
            parameters,_ = optimize.curve_fit(fitting_function, xdata=Xvalues[fit_start:], ydata=Yvalues[fit_start:], maxfev=20000)
            
    #a, b, c = parameters
    current_budget = optimize.minimize(total_runtime_remaining,1,bounds=[(1,total_tuning_steps-current_tuning_step)],args=(a,b,c,current_tuning_step,total_tuning_steps,average_runtime_so_far,overhead))
    int_current_budget = round(current_budget.x[0])


    if fitting_function(current_tuning_step + int_current_budget, a, b, c) < best_configs_so_far[-1]:
        return int_current_budget
    else:
        return 0

