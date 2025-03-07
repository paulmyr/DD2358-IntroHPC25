from timeit import default_timer as timer
import pandas as pd
import json

NUM_RUNS = 20
N = 128
T_END = 2
TERMINATE_USING = "T"
RUNTIME_JSON_LOCATION = "../../profilings/boxplot_runtimes.json"


def continue_experiment(terminate_using, curr_time, curr_iters, end_count):
    if terminate_using == "I":
        return curr_iters < end_count
    else:
        return curr_time < end_count
    

def get_runtimes_for_impl(fv_impl):
    runtimes = []

    for i in range(NUM_RUNS):
        t1 = timer()
        fv_impl(N=N, tEnd=T_END, terminate_using=TERMINATE_USING)
        t2 = timer()
        print(f"Run {i} finished!")
        runtimes.append(t2 - t1)
    
    return runtimes

def update_json_with_runtimes(json_key, runtimes, json_location=RUNTIME_JSON_LOCATION):
    curr_data = None
    with open(json_location, "r") as json_file:
        curr_data = json.load(json_file)

    curr_data[json_key] = runtimes

    with open(json_location, "w") as json_file:
        json.dump(curr_data, json_file)