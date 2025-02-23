from ocean_default import run_simulation_default
from ocean_dask import run_simulation_dask
import matplotlib.pyplot as plt

NUM_ITERS = [100, 200, 400, 800]
NUM_RUNS = 5


def profile_default():
    """
    Profiles the default implementation of the ocean currents code, returning the running times for the 
    NUM_ITERS runs in an array. The running times are averaged over NUM_RUNS runs
    """
    times = []
    print(f"PROFILING COMPUTATION FOR DEFAULT IMPLEMENTATION (runs={NUM_RUNS})")

    for curr_iters in NUM_ITERS:
        total_time = 0

        for _ in range(NUM_RUNS):
            _,_,_, curr_time = run_simulation_default(deterministic=True, profile_time=True, num_iters=curr_iters)
            total_time += curr_time
        
        times.append(total_time / NUM_RUNS)
        print(f"Avg Time for {curr_iters} iterations: {total_time / NUM_RUNS} s ")
    
    # Plot the timess
    plt.loglog(NUM_ITERS, times, marker="o", linestyle='-', label='default')
    
    return times


def profile_dask(chunk_size):
    """
    Profiles the dask implementation for the ocean currents code, returning the times for NUM_ITERS in an array, 
    with the running times averaged over NUM_RUNS runs.

    Depends on the chunk_size passed in to the function.
    """
    times = []
    print(f"PROFILING COMPUTATION FOR DASK IMPLEMENTATION (chunk_size={chunk_size}, runs={NUM_RUNS})")

    for curr_iters in NUM_ITERS:
        total_time = 0

        for _ in range(NUM_RUNS):
            _,_,_, curr_time = run_simulation_dask(deterministic=True, profile_time=True, num_iters=curr_iters, chunk_size=chunk_size)
            total_time += curr_time
        
        times.append(total_time / NUM_RUNS)
        print(f"Avg Time for {curr_iters} iterations: {total_time / NUM_RUNS} s ")
    
    # Plot the times
    plt.loglog(NUM_ITERS, times, marker="o", linestyle='-', label=f"dask (chunk_size={chunk_size})")
    return times


def profile_dask_default():
    # Get the times
    profile_default()
    profile_dask(chunk_size=50)
    profile_dask(chunk_size=100)
    profile_dask(chunk_size=200)

    # Adding labels to the axes
    plt.xlabel('Num Iterations')
    plt.ylabel('Running Time (s)')

    # Adding a title to the plot
    plt.title('Time Taken Updates to Ocean (log-log)')
    plt.legend(loc='upper left')
    # Show the plot
    plt.show()

if __name__ == "__main__":
    profile_dask_default()