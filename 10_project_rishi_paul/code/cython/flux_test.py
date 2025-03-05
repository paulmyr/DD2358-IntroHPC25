import pytest
import numpy as np
from finitevolume_cython import getFlux
import finitevolume_cython_lib

GAMMA = 5/3

def get_matrices(grid_size):
    np.random.seed(40)
    rho_L = np.random.rand(grid_size, grid_size)
    np.random.seed(41)
    rho_R = np.random.rand(grid_size, grid_size)
    np.random.seed(42)
    vx_L = np.random.rand(grid_size, grid_size)
    np.random.seed(43)
    vx_R = np.random.rand(grid_size, grid_size)
    np.random.seed(44)
    vy_L = np.random.rand(grid_size, grid_size)
    np.random.seed(45)
    vy_R = np.random.rand(grid_size, grid_size)
    np.random.seed(65)
    P_L = np.random.rand(grid_size, grid_size)
    np.random.seed(66)
    P_R = np.random.rand(grid_size, grid_size)

    return rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R

@pytest.mark.parametrize("grid_size", [64, 128, 256, 512, 1024])
def test_flux(grid_size):

    rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R = get_matrices(grid_size)

    default_mass, default_momx, default_momy, default_energy = getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, GAMMA)
    cython_mass, cython_momx, cython_momy, cython_energy = finitevolume_cython_lib.getFluxRawC(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, GAMMA)


    assert np.allclose(default_mass, cython_mass)
    assert np.allclose(default_momx, cython_momx)
    assert np.allclose(default_momy, cython_momy)
    assert np.allclose(default_energy, cython_energy)