from wildfiremontecarlodask import run_n_simulations_dask
import matplotlib.pyplot as plt
from dask.distributed import Client
from timeit import default_timer as timer

# We keep the number of simulations at a constant 8.
NUM_SIMS = [8, 16]
NUM_WORKERS = [2, 4, 8, 14]
NUM_RUNS = 3


def profile_dask_worker(num_workers):
    """
    Profiles the dask implementation with a different number of workers. 
    """
    times = []
    print(f"PROFILING COMPUTATION FOR DASK IMPLEMENTATION (runs={NUM_RUNS}, workers={num_workers})")

    with Client(n_workers=num_workers, threads_per_worker=1) as client:
        for curr_sims in NUM_SIMS:
            curr_times = []
            total_time = 0
            seeds = [i for i in range(curr_sims)]

            for _ in range(NUM_RUNS):
                t1 = timer()
                # We don't care about the result here
                run_n_simulations_dask(n_simulations=curr_sims, seeds=seeds, no_print=True)
                t2 = timer()
                total_time += (t2 - t1)
            
            curr_times.append(total_time / NUM_RUNS)
            print(f"Avg Time for {curr_sims} simulations: {total_time / NUM_RUNS} s ")

            times.append(curr_times)
    
    return times


def profile():
    times_8, times_16 = [], []

    # Get the times (we keep the number of simulations to a constant 8 here)
    for curr_workers in NUM_WORKERS:
        curr_8, curr_16 = profile_dask_worker(curr_workers)
        times_8 += curr_8
        times_16 += curr_16

    # Plot the times
    plt.loglog(NUM_WORKERS, times_8, marker="o", linestyle='-', label="8 Simulations")
    plt.loglog(NUM_WORKERS, times_16, marker="o", linestyle='-', label="16 Simulations")

    # Adding labels to the axes
    plt.xlabel('Num Workers')
    plt.ylabel('Running Time (s)')

    # Adding a title to the plot
    plt.title('Time Taken for 8 & 16 Simulations Based on Worker Count (log-log)')
    plt.legend(loc='upper right')
    # Show the plot
    plt.show()

if __name__ == "__main__":
    profile()