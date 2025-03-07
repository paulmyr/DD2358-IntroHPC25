import pytest
import numpy as np
import finitevolume_cython_lib

GAMMA = 5/3

# THE DEFAULT FLUX IMPLEMENTATION, USED HERE FOR TESTING
def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
    rho_L        is a matrix of left-state  density
    rho_R        is a matrix of right-state density
    vx_L         is a matrix of left-state  x-velocity
    vx_R         is a matrix of right-state x-velocity
    vy_L         is a matrix of left-state  y-velocity
    vy_R         is a matrix of right-state y-velocity
    P_L          is a matrix of left-state  pressure
    P_R          is a matrix of right-state pressure
    gamma        is the ideal gas gamma
    flux_Mass    is the matrix of mass fluxes
    flux_Momx    is the matrix of x-momentum fluxes
    flux_Momy    is the matrix of y-momentum fluxes
    flux_Energy  is the matrix of energy fluxes
    """

    # left and right energies
    en_L = P_L/(gamma-1)+0.5*rho_L * (vx_L**2+vy_L**2)
    en_R = P_R/(gamma-1)+0.5*rho_R * (vx_R**2+vy_R**2)

    # compute star (averaged) states
    rho_star  = 0.5*(rho_L + rho_R)
    momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
    en_star   = 0.5*(en_L + en_R)

    P_star = (gamma-1)*(en_star-0.5*(momx_star**2+momy_star**2)/rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass   = momx_star
    flux_Momx   = momx_star**2/rho_star + P_star
    flux_Momy   = momx_star * momy_star/rho_star
    flux_Energy = (en_star+P_star) * momx_star/rho_star

    # find wavespeeds
    C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(vx_L)
    C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(vx_R)
    C = np.maximum( C_L, C_R )

    # add stabilizing diffusive term
    flux_Mass   -= C * 0.5 * (rho_L - rho_R)
    flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * ( en_L - en_R )

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

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