from evaluator import evaluator
import csv
import time
import os

DATA_PATH = "../raw-data/raw-autotuning-data/"
OUTPUT_DIR = "output/"

def experiment(file_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_file_name = OUTPUT_DIR + "benchmarking_"+file_name
    extra_result_file_name = OUTPUT_DIR + "extra_results_"+file_name
    
    write_file = True

    if write_file:
        with open(result_file_name,'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["HW / # of runs / method / overhead","Performance decline (%)"])
        with open (extra_result_file_name,'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["HW / # of runs / method / overhead","Average extra runtime","Extra runtime deviation","Average GT TS","GT TS deviation","Average estimated TS","Estimated TS deviation","Average miss (tuning steps)"])

    hist_HW = "680" # the HW used to set the estimate
    # hw_list = ["1070"]
    hw_list = ["680", "750", "1070", "2080"]
    kernel_run_number_list = ["10000","10000000"]

    k_values = ["1.0","0.0","0.5"] # k: emphasis on regression [0: no regression, 1: regression only, 0 < x < 1: hybrid] ale radsej sa spytaj este Jirku
    overhead_values = ["10000","1000000"] # natvrdo dany overhead
    hw_kernel_combinations = [] # pole poli, kde je [<graficka karta>, <pocet behov kernelu>, <k_value>, <overhead>]

    for a in hw_list:
        for b in kernel_run_number_list:
            for c in k_values:
                for d in overhead_values:
                    benchmark = file_name.replace("_output.csv", "")
                    if os.path.exists(DATA_PATH + benchmark + "/" + a + "-" + file_name):
                        hw_kernel_combinations.append([a,b,c,d])

    start_time = time.time()
    for hw_kernel_combination in hw_kernel_combinations:
        #HW, file_name, total_kernel_runs_str = hw_kernel_combination
        HW, total_kernel_runs_str, k_str, overhead_str = hw_kernel_combination # tuto to cez n-tuple rozbali
        total_kernel_runs = int(total_kernel_runs_str) #prehodi z string -> int 
        k = float(k_str) #detto
        overhead = int(overhead_str) #detto

        #k: emphasis on regression [0: no regression, 1: regression only, 0 < x < 1: hybrid]
        #k = 0.5
        fit_start = 10
        #overhead = 500000

        #parameters: HW, file_name, total_kernel_runs, hist_HW, k = 1, overhead = 0, fit_start = 15, number_of_tests = 1000
        average_extra_runtime, std_extra_runtime, avg_crystal_ball, std_crystal_ball, avg_estimate, std_estimate, avg_miss = evaluator(HW,file_name,total_kernel_runs, hist_HW, k, overhead, fit_start, 1000)
 

        # Dalej je uz len nejaky exporter do CSV

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
        print("***************************\n\n\n")

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


experiment("gemm-reduced_output.csv")
