import pytest
import numpy as np
from conway_profile import update, update_vectorized

def grid_creater_utility(n):
    # Utility function to create grids for testing (given grid size). Sets the seed so that the 
    # grid is returned on multiple calls for the same grid size.
    np.random.seed(42)
    return np.random.choice([255, 0], n * n, p=[0.2, 0.8]).reshape(n, n)

@pytest.mark.parametrize("grid_size", [64, 128, 256, 512, 1024])
def test_grid_diff_sizes(grid_size):
    normal_grid =grid_creater_utility(grid_size)
    vectorized_grid = grid_creater_utility(grid_size)
    # Make sure that both the nomral and vectorized grid are in the same state
    # after 10 operations
    max_iters = 10
    for i in range(max_iters):
        update(normal_grid, grid_size)
        update_vectorized(vectorized_grid, grid_size)

    assert np.array_equal(normal_grid, vectorized_grid)