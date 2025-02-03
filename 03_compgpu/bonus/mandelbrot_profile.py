from timeit import default_timer as timer
from mandelbrot_set_cython import generate_image_cython
from mandelbrot_set_default import generate_image_default
import matplotlib.pyplot as plt

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

def plot_data(running_times_default, running_times_cython, running_times_torch, grid_sizes):
    # Create a line plot with dots at each data point
    plt.loglog(grid_sizes, running_times_default, marker='o', linestyle='-', color='b', label='default mandelbrot pure-python (m1 macbook pro)')
    plt.loglog(grid_sizes, running_times_cython, marker='o', linestyle='-', color='r', label='cython mandelbrot (m1 macbook pro)')
    plt.loglog(grid_sizes, running_times_torch, marker='o', linestyle="-", color="orange", label="torch mandelbrot (t4 colab gpu)")


    # Adding labels to the axes
    plt.xlabel('Grid Size')
    plt.ylabel('Running Time (s)')

    # Adding a title to the plot
    plt.title('Times for Mandelbrot Set Gen for Different Square Grids with max 100 iters (log-log)')
    plt.legend(loc='upper left')
    # Show the plot
    plt.show()

if __name__ == "__main__":
    grid_sizes, default_times = profile_generation(generate_method_key="default_generate")
    _, cython_times = profile_generation(generate_method_key="cython_generate")
    # Torch Times are hard-coded here but were obtained from "mandelbrot_torch.ipynb" notebook
    # executed on Google Colabs T4 GPU. See Notebook for reference.
    torch_times = [0.012661022200018125, 0.030808558399996855, 0.10812428999999497, 0.42733559330001186]
    plot_data(default_times, cython_times, torch_times, grid_sizes)