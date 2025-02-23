from ocean_default import run_simulation_default
from ocean_dask import run_simulation_dask
import numpy as np
import pytest


@pytest.mark.parametrize("num_iters", [100, 200, 400])
@pytest.mark.parametrize("chunk_size", [50, 100, 200])
def test_ocean_updates(num_iters, chunk_size):
    # We don't really care about the time here, but that stops the print statements from being generated
    u_default, v_default, temp_default, _ = run_simulation_default(deterministic=True, profile_time=True, num_iters=num_iters)
    u_dask, v_dask, temp_dask, _ = run_simulation_dask(deterministic=True, profile_time=True, num_iters=num_iters, chunk_size=chunk_size)

    assert np.array_equal(u_default, u_dask)
    assert np.array_equal(v_default, v_dask)
    assert np.array_equal(temp_default, temp_dask)