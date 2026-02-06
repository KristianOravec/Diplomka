import numpy as np

def get_average_best_curve(list_of_arrays):
    """
    Transforms a list of arrays into their running minimums and 
    returns their element-wise average.
    """
    running_mins = [np.minimum.accumulate(arr) for arr in list_of_arrays]
    result_array = np.mean(running_mins, axis=0)
    
    return result_array


array_a = [1000, 800, 1200, 1110, 750, 710]
array_b = [1100, 730, 1500, 1000, 800, 710]

avg_curve = get_average_best_curve([array_a, array_b])
print(avg_curve)
