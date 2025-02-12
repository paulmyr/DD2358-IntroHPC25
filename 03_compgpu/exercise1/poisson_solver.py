import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from line_profiler import profile
import purepythonpoisson

# @profile
def run_pure_numpy(N, num_iterations):
    np.random.seed(42)
    grid = np.random.rand(N, N)
    grid[0,:] = 0
    grid[:,0] = 0

    for _ in range(num_iterations):
        grid = purepythonpoisson.numpy_pure_gauss_seidel(grid)


def timed(f, *args, **kwargs):
    t0 = timer()
    f(*args, **kwargs)
    t1 = timer()
    return t1 - t0


def run_function_as_experiment(f, num_iterations = 1000):
    n = [2**i for i in range(7, 8)]
    wtimes = [timed(f, i, num_iterations) for i in n]

    print(f"ran {f.__name__}")
    print(f"each grid ran {num_iterations} iterations.")
    for i, j in zip(n, wtimes):
        print(f"Grid: ({i}, {i})\n\truntime: {j}s")

    plt.plot(n, wtimes, label=f"{f.__name__}")


if __name__ == '__main__':
    run_function_as_experiment(run_pure_numpy)

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("wtime")
    plt.legend()

    plt.show()
