import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from line_profiler import profile
import purepythonpoisson
import cythonpoisson
import h5py

# @profile
def run_pure_numpy(N, num_iterations):
    np.random.seed(42)
    grid = np.zeros((N, N))
    grid[1:-1, 1:-1] = np.random.rand(N-2, N-2)

    for _ in range(num_iterations):
        grid = purepythonpoisson.numpy_pure_gauss_seidel(grid)

def solve_poisson(fn, N, num_iterations):
    """
    Initializes a NxN according to the constraints given and then returns the resulting
    answer after calling the provided solving function on it.

    N: The dimensions of the square grid
    num_iterations: The number of iterations to run the gauss_seidel algorithm for
    fn: The solving funciton to call on each iteration

    The grid initialized has the same seed to ensure consistent initialization and results
    """
    np.random.seed(42)
    grid = np.zeros((N, N))
    grid[1:-1, 1:-1] = np.random.rand(N-2, N-2)

    for _ in range(num_iterations):
        grid = fn(grid)

    return grid 


def timed(f, *args, **kwargs):
    t0 = timer()
    solve_poisson(f, *args, **kwargs)
    t1 = timer()
    return t1 - t0


def run_function_as_experiment(f, num_iterations = 100):
    grid_sizes = [2**i for i in range(6, 11)]

    num_runs = 5
    wtimes = np.zeros(len(grid_sizes))
    for _ in range(num_runs):
        wtimes += [timed(f, curr_size, num_iterations) for curr_size in grid_sizes]
    wtimes /= num_runs 

    print(f"ran {f.__name__}")
    print(f"each grid ran {num_iterations} iterations. ({num_runs} runs)")
    for i, j in zip(grid_sizes, wtimes):
        print(f"Grid: ({i}, {i})\n\tavg runtime: {j}s")

    plt.plot(grid_sizes, wtimes, label=f"{f.__name__}", marker='o')

def save_1024_grid_hdf5():
    pure_python_result = solve_poisson(1024, 50, purepythonpoisson.numpy_pure_gauss_seidel)
    cython_result = solve_poisson(1024, 50, cythonpoisson.numpy_cython_gauss_seidel)

    f = h5py.File("matrix_python_cython.hdf5", "w")
    f["/1024/pure_python"] = pure_python_result
    f["/1024/cython"] = cython_result


if __name__ == '__main__':
    run_function_as_experiment(purepythonpoisson.numpy_pure_gauss_seidel)
    run_function_as_experiment(cythonpoisson.numpy_cython_gauss_seidel)


    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("wtime")
    plt.legend()

    plt.show()
    # save_1024_grid_hdf5()
