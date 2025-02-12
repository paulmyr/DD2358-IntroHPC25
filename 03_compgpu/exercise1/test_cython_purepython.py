import pytest
import numpy as np
import purepythonpoisson
import cythonpoisson
from poisson_solver import solve_poisson

# We fix the number of iterations here to prevent long test times.
NUM_ITERATIONS = 50

@pytest.mark.parametrize("grid_size", [64, 126, 256, 512, 1024])
def test_mandelbrot_grid_diff_sizes(grid_size):

    default_grid = solve_poisson(purepythonpoisson.numpy_pure_gauss_seidel, grid_size, NUM_ITERATIONS)
    cython_grid = solve_poisson(cythonpoisson.numpy_cython_gauss_seidel, grid_size, NUM_ITERATIONS)
    
    assert np.array_equal(cython_grid, default_grid)

