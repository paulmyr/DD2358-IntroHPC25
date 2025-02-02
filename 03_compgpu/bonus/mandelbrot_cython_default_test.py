import pytest
import numpy as np
from mandelbrot_set_cython import generate_image_cython
from mandelbrot_set_default import generate_image_default

@pytest.mark.parametrize("grid_size", [256, 512, 1024, 2048])
def test_mandelbrot_grid_diff_sizes(grid_size):
    default_image = generate_image_default(grid_size, grid_size)
    cython_image = generate_image_cython(grid_size, grid_size)
    
    assert np.array_equal(cython_image, default_image)