import pytest
import numpy as np
import purepythonpoisson
import cythonpoisson

# We fix the number of iterations here to prevent long test times.
NUM_ITERATIONS = 50

def solve_poisson(N, num_iterations, fn):
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

@pytest.mark.parametrize("grid_size", [64, 126, 256, 512, 1024])
def test_mandelbrot_grid_diff_sizes(grid_size):

    default_grid = solve_poisson(grid_size, NUM_ITERATIONS, purepythonpoisson.numpy_pure_gauss_seidel)
    cython_grid = solve_poisson(grid_size, NUM_ITERATIONS, cythonpoisson.numpy_cython_gauss_seidel)
    
    assert np.array_equal(cython_grid, default_grid)

