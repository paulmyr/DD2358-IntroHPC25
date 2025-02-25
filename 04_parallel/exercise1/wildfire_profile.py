from wildfiremontecarlodask import run_n_simulations_dask
from wildfiremontecarloparallel import run_n_simulations_parallel
from wildfiremontecarloserial import run_n_simulations_default
import matplotlib.pyplot as plt
from dask.distributed import Client
from timeit import default_timer as timer

NUM_SIMS = [1, 2, 4 ,8, 16]
NUM_RUNS = 3


def profile_function(func_to_profile, impl_type):
    """
    Profiles the provided implementation.
    The function to profile should be one of the "run_n_simulations_*" function, and it must accept 3 arguments:
    n_simulations, seeds, and no_print.
    """
    times = []
    print(f"PROFILING COMPUTATION FOR {impl_type} IMPLEMENTATION (runs={NUM_RUNS})")

    for curr_sims in NUM_SIMS:
        # Create as many workers as there are simulations to get as much parallelism as possible. Also just use 1 thread.
        with Client(n_workers=curr_sims, threads_per_worker=1) as client:
            total_time = 0
            seeds = [i for i in range(curr_sims)]

            for _ in range(NUM_RUNS):
                t1 = timer()
                # We don't care about the result here
                func_to_profile(n_simulations=curr_sims, seeds=seeds, no_print=True)
                t2 = timer()
                total_time += (t2 - t1)
        
            times.append(total_time / NUM_RUNS)
            print(f"Avg Time for {curr_sims} simulations: {total_time / NUM_RUNS} s ")
    
    # Plot the timess
    plt.loglog(NUM_SIMS, times, marker="o", linestyle='-', label=impl_type)
    
    return times


def profile():
    # Get the times
    profile_function(run_n_simulations_default, "serial")
    profile_function(run_n_simulations_parallel, "multiprocessing")
    profile_function(run_n_simulations_dask, "dask")


    # Adding labels to the axes
    plt.xlabel('Num Simulations')
    plt.ylabel('Running Time (s)')

    # Adding a title to the plot
    plt.title('Time Taken for Multiple Simulations For Different Impl. (log-log)')
    plt.legend(loc='upper left')
    # Show the plot
    plt.show()

if __name__ == "__main__":
    profile()