from timeit import default_timer as timer
import numpy as np
from gauss_seidel_base import solve_posisson
from gauss_seidel_cython import solve_posisson_cython
import matplotlib.pyplot as plt

# Dictionary for different update method implementations
SOLVE_DICT = {
    "default_gs": solve_posisson,
    "cython_gs": solve_posisson_cython
}

def grid_generator(grid_size):
    """
    Generates the square grid with 0s on the boundary of the required size. 
    Uses a seed to ensure that the grids are the same for multiple calls
    """
    np.random.seed(42)
    base_grid = np.zeros((grid_size, grid_size))
    base_grid[1:-1, 1:-1] = np.random.rand(grid_size-2, grid_size-2)

    return base_grid


def profile_generation(generate_method_key):
    # grid sizes increase by powers of 2
    grid_sizes = [64, 126, 256, 512, 1024]
    # times for each of the computations (in seconds)
    times = []
    # The number of times the run will be performed for each grid size.
    # The final running time reported will be the average of these runs
    num_runs = 1
    # Number of iterations over which profiling occurs
    num_iterations = 50   

    print(f"PROFILING COMPUTATION FOR {generate_method_key} (iters={num_iterations}, runs={num_runs})")

    for curr_size in grid_sizes:
        total_time = 0

        for _ in range(num_runs):
            # Generate the grid everytime to ensure it follows the requirements and 
            # we start fresh everytime
            curr_grid = grid_generator(curr_size)

            t1 = timer()
            # The Method signature is: (grid, num_iters). Use 500 iters for profiling.
            SOLVE_DICT[generate_method_key](curr_grid, num_iterations)
            t2 = timer()

            total_time += (t2 - t1)
        
        times.append(total_time / num_runs)
        print(f"Avg Time for {curr_size}x{curr_size}: {total_time / num_runs}")

    return (grid_sizes, times)

def plot_data(running_times_default, running_times_cython, running_times_torch=[], running_times_cupy=[], grid_sizes=[], local_platform_name="m1 mackbook pro"):
    # Create a line plot with dots at each data point
    plt.loglog(grid_sizes, running_times_default, marker='o', linestyle='-', color='b', label=f'default solver pure-python ({local_platform_name})')
    plt.loglog(grid_sizes, running_times_cython, marker='o', linestyle='-', color='r', label=f'cython solver ({local_platform_name})')
    if len(running_times_torch) > 0:
        plt.loglog(grid_sizes, running_times_torch, marker='o', linestyle="-", color="orange", label="torch solver (t4 colab gpu)")
    if len(running_times_cupy) > 0:
        plt.loglog(grid_sizes, running_times_torch, marker='o', linestyle="-", color="green", label="cupy solver (t4 colab gpu)")


    # Adding labels to the axes
    plt.xlabel('Grid Size')
    plt.ylabel('Running Time (s)')

    # Adding a title to the plot
    plt.title(f'Times for Gauss-Seidel Poisson Solvers for Different Square Grids with 500 iters (log-log)')
    plt.legend(loc='upper left')
    # Show the plot
    plt.show()

if __name__ == "__main__":
    grid_sizes, default_times = profile_generation(generate_method_key="default_gs")
    _, cython_times = profile_generation(generate_method_key="cython_gs")
    # Torch and CuPy Times are hard-coded here, obtained from relevent python notebook
    # TODO: Fill this with running times obtained from PyTorch GPU
    # torch_times = [..., ..., ..., ...]
    # TODO: Fill this with running times obtained from CuPy GPU
    # cupy_times = [..., ..., ..., ...]
    plot_data(default_times, cython_times, grid_sizes=grid_sizes)