import pytest
import numpy as np
from gauss_seidel_base import solve_posisson
from gauss_seidel_cython import solve_posisson_cython
from gauss_seidel_perf_compare import grid_generator

# We fix the number of iterations here to prevent long test times.
NUM_ITERATIONS = 50

@pytest.mark.parametrize("grid_size", [64, 126, 256, 512, 1024])
def test_mandelbrot_grid_diff_sizes(grid_size):
    input_default = grid_generator(grid_size)
    input_cython = grid_generator(grid_size)

    default_grid = solve_posisson(input_default, NUM_ITERATIONS)
    cython_grid = solve_posisson_cython(input_cython, NUM_ITERATIONS)
    
    assert np.array_equal(cython_grid, default_grid)