import pytest
import numpy as np
import finitevolume_cython
import finitevolume_cython_dask

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import finitevolume_runtimeplots

@pytest.mark.parametrize("grid_size", [64, 128, 256, 512])
def test_rho(grid_size):
    """
    Sanity checks the different optimizations to ensure that the final rho (the thing that is plotted)
    obtained from all of them is close enough that any possible differences can be attributed to different 
    floating point precisions in different implementations (eg: C and Python/Numpy).

    We use np.allclose to check for similarity.
    """

    rho_default = finitevolume_runtimeplots.main(N=grid_size, tEnd=0.1, terminate_using="T")
    rho_cython = finitevolume_cython.main(N=grid_size, tEnd=0.1, terminate_using="T")
    rho_cython_dask = finitevolume_cython_dask.main(N=grid_size, tEnd=0.1, terminate_using="T")


    # Check that they are all close enough to the default implementation
    assert np.allclose(rho_default, rho_cython)
    assert np.allclose(rho_default, rho_cython_dask)