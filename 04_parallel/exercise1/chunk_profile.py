from wildfiremontecarlodask import run_n_simulations_dask
import matplotlib.pyplot as plt
from dask.distributed import Client
from timeit import default_timer as timer

# We keep the number of simulations at a constant 8.
NUM_SIMS = [8]
# The chunk size here determines the NUMBER OF COLUMNS that are allocated to each chunk.
# We have 60 days (the maximum simulation time), and based on the number of simulations, our rows 
# will change. Thus, the final parallel computation of the average will be done on an array 
# of dimensions (num_simulations x 60). We can parallelize the computation of the average FOR EACH DAY
# here, and thus the chunks will be of dimensions (num_simulations, i). 
# To keep things neat, we will ensure that i is a factor of 60. 
CHUNK_SIZE = [10, 20, 30, 60]
NUM_RUNS = 3


def profile_dask_chunks(chunk_dims):
    """
    Profiles the dask implementation with a different number of workers. 
    """
    times = []
    print(f"PROFILING COMPUTATION FOR DASK IMPLEMENTATION (runs={NUM_RUNS}, chunk_columns={chunk_dims})")

    with Client() as client:
        for curr_sims in NUM_SIMS:
            total_time = 0
            seeds = [i for i in range(curr_sims)]

            for _ in range(NUM_RUNS):
                t1 = timer()
                # We don't care about the result here
                run_n_simulations_dask(n_simulations=curr_sims, seeds=seeds, no_print=True, chunk_size=(curr_sims, chunk_dims))
                t2 = timer()
                total_time += (t2 - t1)
            
            times.append(total_time / NUM_RUNS)
            print(f"Avg Time for {curr_sims} simulations: {total_time / NUM_RUNS} s ")
    
    return times


def profile():
    times = []

    # Get the times
    for curr_dims in CHUNK_SIZE:
        times += profile_dask_chunks(curr_dims)

    # Plot the times
    plt.loglog(CHUNK_SIZE, times, marker="o", linestyle='-', label="Running Time")

    # Adding labels to the axes
    plt.xlabel('Columns in Chunk')
    plt.ylabel('Running Time (s)')

    # Set the appropriate y-limit to ensure that times are clearly visible
    plt.ylim(1, 14)

    # Adding a title to the plot
    plt.title('Time Taken for 8 Simulations Based on Chunk Division (log-log)')
    plt.legend(loc='upper left')
    # Show the plot
    plt.show()

if __name__ == "__main__":
    profile()