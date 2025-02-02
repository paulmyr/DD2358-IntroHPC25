from timeit import default_timer as timer
from mandelbrot_set_cython import generate_image_cython
from mandelbrot_set_default import generate_image_default

# Dictionary for different update method implementations
GENERATE_DICT = {
    "default_generate": generate_image_default,
    "cython_generate": generate_image_cython
}

def profile_generation(generate_method_key):
    # grid sizes increase by powers of 2
    grid_sizes = [256, 512, 1024, 2048]
    # times for each of the computations (in seconds)
    times = []
    # The number of times the run will be performed for each grid size.
    # The final running time reported will be the average of these runs
    num_runs = 10

    print(f"PROFILING COMPUTATION FOR {generate_method_key} (runs={num_runs})")

    for curr_size in grid_sizes:
        total_time = 0

        for _ in range(num_runs):

            t1 = timer()
            GENERATE_DICT[generate_method_key](curr_size, curr_size)
            t2 = timer()

            total_time += (t2 - t1)
        
        times.append(total_time / num_runs)
        print(f"Avg Time for {curr_size}x{curr_size}: {total_time / num_runs}")

    return (grid_sizes, times)

# def plot_data(running_times_classic, running_times_vectorized, grid_sizes):
#     # Create a line plot with dots at each data point
#     plt.loglog(grid_sizes, running_times_classic, marker='o', linestyle='-', color='b', label='classic running times')
#     plt.loglog(grid_sizes, running_times_vectorized, marker='o', linestyle='-', color='r', label='vectorized running times')


#     # Adding labels to the axes
#     plt.xlabel('Grid Size')
#     plt.ylabel('Running Time (s)')

#     # Adding a title to the plot
#     plt.title('Time Taken for 50 Updates for Varying Grid Sizes (log-log)')
#     plt.legend(loc='upper left')
#     # Show the plot
#     plt.show()

if __name__ == "__main__":
    grid_sizes, classic_times = profile_generation(generate_method_key="default_generate")
    _, vectorized_times = profile_generation(generate_method_key="cython_generate")
    # plot_data(classic_times, vectorized_times, grid_sizes)