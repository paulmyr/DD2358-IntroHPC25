from timeit import default_timer as timer
import matplotlib.pyplot as plt
import argparse
import numpy as np

def timed(f, *args, **kwargs):
    t0 = timer()
    f(*args, **kwargs)
    t1 = timer()
    return t1 - t0


def run_function_as_experiment(f, lbound, ubound, num_runs, tEnd):
    grid_sizes = [2 ** i for i in range(lbound, ubound)]
    wtimes = np.zeros(len(grid_sizes))
    print("================== EXPERIMENT INITIATED ===================")
    for i, grid_size in enumerate(grid_sizes):
        print(f"Expreiment STARTED for grid size: ({grid_size},{grid_size})")
        for _ in range(num_runs):
            curr_time = timed(f, N=grid_size, tEnd=tEnd)
            wtimes[i] += curr_time
            print(f"\t Run {_} completed: {curr_time}")
        wtimes[i] = wtimes[i] / num_runs
        print(f"Experiment COMPLETED for grid size: ({grid_size},{grid_size}), took {wtimes[i]}s")
    print("================== RESULTS ===================")

    print(f"ran {f.__name__}")
    print(f"each grid size ran {num_runs} runs, each run simulated {tEnd} iterations.")
    for i, j in zip(grid_sizes, wtimes):
        print(f"Grid: ({i}, {i})\n\tavg runtime: {j}s")
    # plt.plot(grid_sizes, wtimes, label=f"{f.__name__} (m1 (16') macbook pro, 2021)", marker='o')
    print(wtimes)
    # plt.plot(grid_sizes, wtimes, label=f"{f.__name__} (Apple MacBook M1 Air, 2020)", marker='o')

    # for i,j in zip(grid_sizes, wtimes):
    #     plt.annotate("%.3f s" % j, xy=(i,j), xytext=(5,-10), textcoords="offset points")

def measure_runtime(exp_function):
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lbound", help = "lower bound for N in power of two (default: 4)")
    parser.add_argument("-u", "--ubound", help = "upper bound for N in power of two (default: 12)")
    parser.add_argument("-n", "--nruns", help = "nubmer of runs per grid size (default: 3)")
    parser.add_argument("-t", "--tend", help = "number of seconds simulation should run (default: 2)")

    _args = parser.parse_args()

    if not _args.lbound:
        _args.lbound = 4
    else:
        _args.lbound = int(_args.lbound)
    
    if not _args.ubound:
        _args.ubound = 8 # (range goes up to 12-1 therefore 11
    else:
        _args.ubound = int(_args.ubound) + 1
        
    if not _args.nruns:
        _args.nruns = 3
    else:
        _args.nruns = int(_args.nruns)

    if not _args.tend:
        _args.tend = 2
    else:
        _args.tend = float(_args.tend)

    # main
    run_function_as_experiment(exp_function, _args.lbound, _args.ubound, _args.nruns, _args.tend)

    # plt.xscale("log", base=2)
    # plt.xlabel("N (gridsize: 2**N x 2**N)")
    
    # plt.yscale("log", base=10)
    # plt.ylabel("wtime (in s)")
    # plt.legend()

    # plt.show()