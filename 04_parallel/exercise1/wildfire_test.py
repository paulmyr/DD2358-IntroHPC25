from wildfiremontecarlodask import run_n_simulations_dask
from wildfiremontecarloparallel import run_n_simulations_parallel
from wildfiremontecarloserial import run_n_simulations_default
from dask.distributed import Client
import numpy as np
import pytest


@pytest.mark.parametrize("num_sims", [1, 2, 4, 8, 16])
def test_ocean_updates(num_sims):
    seeds = [i for i in range(num_sims)]
    # We don't really care about the time here, but that stops the print statements from being generated
    result_default = run_n_simulations_default(n_simulations=num_sims, seeds=seeds, no_print=True)
    result_parallel = run_n_simulations_parallel(n_simulations=num_sims, seeds=seeds, no_print=True)

    result_dask = None
    with Client() as client:
        result_dask = run_n_simulations_dask(n_simulations=num_sims, seeds=seeds, no_print=True)
    
    assert np.array_equal(result_default, result_parallel)
    assert np.array_equal(result_default, result_dask)
    assert np.array_equal(result_parallel, result_dask)