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

    plt.plot(grid_sizes, wtimes, label=f"{f.__name__} (m1 (16') macbook pro, 2021)", marker='o')

def save_1024_grid_hdf5():
    pure_python_result = solve_poisson(purepythonpoisson.numpy_pure_gauss_seidel, 1024, 50)
    cython_result = solve_poisson(cythonpoisson.numpy_cython_gauss_seidel, 1024, 50)

    f = h5py.File("matrix_python_cython.hdf5", "w")
    f["/1024/pure_python"] = pure_python_result
    f["/1024/cython"] = cython_result
    f.close()


# This is used to run the profliing experiment for the default python implementation
# and the optimized cython implementation, and then plot the results of these 2 along with
# the runtimes obtained for Jacobi based pytorch and cupy optimizations present
# in the torch_cupy.ipynb notebook.
# The visual results of the runtimes are present in the plot called "ex1_all4_comparison.png"
# present under the "images" directory for a3. However, to look at the raw numbers, you can
# refer to the ipynb notebook for the torch and cupy runtimes, and the below output for
# the pure-python (default) and cython implementations:
#
#
# --------- RAW OUTPUT FOR pure-python (default) AND cython BEGINS --------------------------
# $ python3 poisson_solver.py 
# ran numpy_pure_gauss_seidel
# each grid ran 100 iterations. (5 runs)
# Grid: (64, 64)
# 	avg runtime: 0.17771347500383855s
# Grid: (128, 128)
# 	avg runtime: 0.6884284336585551s
# Grid: (256, 256)
# 	avg runtime: 2.798375883419067s
# Grid: (512, 512)
# 	avg runtime: 12.197537625208497s
# Grid: (1024, 1024)
# 	avg runtime: 51.00448666685261s
# ran numpy_cython_gauss_seidel
# each grid ran 100 iterations. (5 runs)
# Grid: (64, 64)
# 	avg runtime: 0.0025755918119102716s
# Grid: (128, 128)
# 	avg runtime: 0.010063541820272803s
# Grid: (256, 256)
# 	avg runtime: 0.04073448325507343s
# Grid: (512, 512)
# 	avg runtime: 0.1747473582625389s
# Grid: (1024, 1024)
# 	avg runtime: 0.7243686002213507s
# ---------------------- RAW OUTPUT ENDS ----------------------------
if __name__ == '__main__':
    run_function_as_experiment(purepythonpoisson.numpy_pure_gauss_seidel)
    run_function_as_experiment(cythonpoisson.numpy_cython_gauss_seidel)

    # Hard-coding the runtimes for the pytorch and cupy implementations 
    # (which used the Jacobi numeric scheme). Their runtimes are visible in the
    # torch_cupy.ipynb notebook
    grid_sizes = [2**i for i in range(6, 11)]
    torch_runtimes = [0.07143062040004225, 0.01780876279999575, 0.018758892999994715, 0.020996075800030666, 0.06866929739996977]
    cupy_runtimes = [0.05095121419997213, 0.04684149640002033, 0.04514856820001114, 0.051510513400012316, 0.09725730279999426]
    plt.plot(grid_sizes, torch_runtimes, label=f"pytorch (colab t4 gpu)", marker='o')
    plt.plot(grid_sizes, cupy_runtimes, label=f"cupy (colab t4 gpu)", marker='o')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("wtime")
    plt.legend()

    plt.show()
    # save_1024_grid_hdf5()
