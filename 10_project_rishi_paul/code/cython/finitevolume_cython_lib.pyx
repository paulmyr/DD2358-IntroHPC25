import numpy as np
cimport numpy as np
cimport cython
import math

@cython.boundscheck(False)
@cython.wraparound(False)
def getFluxAsArray(double[:,:] rho_L, double[:,:] rho_R, double[:,:] vx_L, double[:,:] vx_R, double[:,:] vy_L, double[:,:] vy_R, double[:,:] P_L, double[:,:] P_R, double gamma):
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
    cdef double[:,:] en_L = np.asarray(P_L)/(gamma-1)+0.5*np.asarray(rho_L) * (np.asarray(vx_L)**2+np.asarray(vy_L)**2)
    cdef double[:,:] en_R = np.asarray(P_R)/(gamma-1)+0.5*np.asarray(rho_R) * (np.asarray(vx_R)**2+np.asarray(vy_R)**2)

    # compute star (averaged) states
    cdef double[:,:] rho_star  = 0.5*(np.asarray(rho_L) + np.asarray(rho_R))
    cdef double[:,:] momx_star = 0.5*(np.asarray(rho_L) * np.asarray(vx_L) + np.asarray(rho_R) * np.asarray(vx_R))
    cdef double[:,:] momy_star = 0.5*(np.asarray(rho_L) * np.asarray(vy_L) + np.asarray(rho_R) * np.asarray(vy_R))
    cdef double[:,:] en_star   = 0.5*(np.asarray(en_L) + np.asarray(en_R))

    cdef double[:,:] P_star = (gamma - 1) * (np.asarray(en_star) - 0.5 * (np.asarray(momx_star)**2 + np.asarray(momy_star)**2) / np.asarray(rho_star))

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    cdef double[:,:] flux_Mass   = momx_star
    cdef double[:,:] flux_Momx   = (np.asarray(momx_star)**2 / np.asarray(rho_star)) + np.asarray(P_star)
    cdef double[:,:] flux_Momy   = np.asarray(momx_star) * np.asarray(momy_star)/np.asarray(rho_star)
    cdef double[:,:] flux_Energy = (np.asarray(en_star) + np.asarray(P_star)) * np.asarray(momx_star)/np.asarray(rho_star)

    # find wavespeeds
    cdef double[:,:] C_L = np.sqrt(gamma*np.asarray(P_L)/np.asarray(rho_L), dtype=np.double) + np.abs(vx_L, dtype=np.double)
    cdef double[:,:] C_R = np.sqrt(gamma*np.asarray(P_R)/np.asarray(rho_R), dtype=np.double) + np.abs(vx_R, dtype=np.double)
    cdef double[:,:] C = np.maximum( C_L, C_R )

    # add stabilizing diffusive term
    flux_Mass   = np.asarray(flux_Mass) - (np.asarray(C) * 0.5 * (np.asarray(rho_L) - np.asarray(rho_R)))
    flux_Momx   = np.asarray(flux_Momx) - (np.asarray(C) * 0.5 * (np.asarray(rho_L) * np.asarray(vx_L) - np.asarray(rho_R) * np.asarray(vx_R)))
    flux_Momy   = np.asarray(flux_Momy) - (np.asarray(C) * 0.5 * (np.asarray(rho_L) * np.asarray(vy_L) - np.asarray(rho_R) * np.asarray(vy_R)))
    flux_Energy = np.asarray(flux_Energy) - (np.asarray(C) * 0.5 * ( np.asarray(en_L) - np.asarray(en_R) ))

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy



#############################################################
# THE NESTED LOOP IMPLEMENTATION
#############################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def nested_constant_division(double[:,:] array_one, double constant):
    cdef unsigned int N = len(array_one)
    cdef unsigned int i, j
    cdef double[:,:] result = np.zeros((N, N), dtype=np.double)

    for i in range(N):
        for j in range(N):
            result[i, j] = array_one[i, j] / constant
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def nested_constant_mult(double[:,:] array, double constant):
    cdef unsigned int N = len(array)
    cdef unsigned int i, j
    cdef double[:,:] result = np.zeros((N, N), dtype=np.double)

    for i in range(N):
        for j in range(N):
            result[i, j] = array[i, j] * constant
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def nested_array_addition(double[:,:] array_one, double[:,:] array_two):
    cdef unsigned int N = len(array_one)
    cdef unsigned int i, j
    cdef double[:,:] result = np.zeros((N, N), dtype=np.double)

    for i in range(N):
        for j in range(N):
            result[i, j] = array_one[i, j] + array_two[i, j]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def nested_array_subtraction(double[:,:] array_one, double[:,:] array_two):
    cdef unsigned int N = len(array_one)
    cdef unsigned int i, j
    cdef double[:,:] result = np.zeros((N, N), dtype=np.double)

    for i in range(N):
        for j in range(N):
            result[i, j] = array_one[i, j] - array_two[i, j]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def nested_array_mult(double[:,:] array_one, double[:,:] array_two):
    cdef unsigned int N = len(array_one)
    cdef unsigned int i, j
    cdef double[:,:] result = np.zeros((N, N), dtype=np.double)

    for i in range(N):
        for j in range(N):
            result[i, j] = array_one[i, j] * array_two[i, j]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def nested_array_division(double[:,:] array_one, double[:,:] array_two):
    cdef unsigned int N = len(array_one)
    cdef unsigned int i, j
    cdef double[:,:] result = np.zeros((N, N), dtype=np.double)

    for i in range(N):
        for j in range(N):
            result[i, j] = array_one[i, j] / array_two[i, j]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def nested_constant_power(double[:,:] array, unsigned int const):
    cdef unsigned int N = len(array)
    cdef unsigned int i, j
    cdef double[:,:] result = np.zeros((N, N), dtype=np.double)

    for i in range(N):
        for j in range(N):
            result[i, j] = array[i, j] ** const
    
    return result


def getFluxAsLoops(double[:,:] rho_L, double[:,:] rho_R, double[:,:] vx_L, double[:,:] vx_R, double[:,:] vy_L, double[:,:] vy_R, double[:,:] P_L, double[:,:] P_R, double gamma):
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
    cdef double[:,:] en_L = nested_array_addition(
        nested_constant_division(P_L, gamma-1), 
        nested_constant_mult(
            nested_array_mult(
                rho_L, 
                nested_array_addition(
                    nested_constant_power(vx_L, 2), 
                    nested_constant_power(vy_L, 2)
                )
            ), 
            0.5
        )
    )
    cdef double[:,:] en_R = nested_array_addition(
        nested_constant_division(P_R, gamma-1), 
        nested_constant_mult(
            nested_array_mult(
                rho_R, 
                nested_array_addition(
                    nested_constant_power(vx_R, 2), 
                    nested_constant_power(vy_R, 2)
                )
            ), 
            0.5
        )
    )

    # compute star (averaged) states
    cdef double[:,:] rho_star  = nested_constant_mult(nested_array_addition(rho_L, rho_R), 0.5)
    cdef double[:,:] momx_star = nested_constant_mult(nested_array_addition(nested_array_mult(rho_L, vx_L), nested_array_mult(rho_R, vx_R)), 0.5)
    cdef double[:,:] momy_star = nested_constant_mult(nested_array_addition(nested_array_mult(rho_L, vy_L), nested_array_mult(rho_R, vy_R)), 0.5)
    cdef double[:,:] en_star   = nested_constant_mult(nested_array_addition(en_L, en_R), 0.5)

    cdef double[:,:] P_star = nested_constant_mult(
        nested_array_subtraction(
            en_star, 
            nested_constant_mult(
                nested_array_division(
                    nested_array_addition(
                        nested_constant_power(momx_star,2), 
                        nested_constant_power(momy_star,2)
                        ), 
                    rho_star
                    ), 
                0.5
                )
            ), 
        gamma-1
    )

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    cdef double[:,:] flux_Mass   = momx_star
    cdef double[:,:] flux_Momx   = nested_array_addition(nested_array_division(nested_constant_power(momx_star, 2), rho_star), P_star)
    cdef double[:,:] flux_Momy   = nested_array_mult(momx_star, nested_array_division(momy_star, rho_star))
    cdef double[:,:] flux_Energy = nested_array_mult(nested_array_addition(en_star, P_star), nested_array_division(momx_star, rho_star))

    # find wavespeeds
    cdef double[:,:] C_L = np.sqrt(nested_array_division(nested_constant_mult(P_L, gamma), rho_L), dtype=np.double) + np.abs(vx_L, dtype=np.double)
    cdef double[:,:] C_R = np.sqrt(nested_array_division(nested_constant_mult(P_R, gamma), rho_R), dtype=np.double) + np.abs(vx_R, dtype=np.double)
    cdef double[:,:] C = np.maximum( C_L, C_R )

    # add stabilizing diffusive term
    flux_Mass   = nested_array_subtraction(
        flux_Mass, 
        nested_array_mult(
            C, 
            nested_constant_mult(
                nested_array_subtraction(rho_L, rho_R), 
                0.5
            )
        )
    )
    flux_Momx   = nested_array_subtraction(
        flux_Momx, 
        nested_array_mult(
            C, 
            nested_constant_mult(
                nested_array_subtraction(nested_array_mult(rho_L, vx_L), nested_array_mult(rho_R, vx_R)), 
                0.5
            )
        )
    )
    flux_Momy   = nested_array_subtraction(
        flux_Momy, 
        nested_array_mult(
            C, 
            nested_constant_mult(
                nested_array_subtraction(nested_array_mult(rho_L, vy_L), nested_array_mult(rho_R, vy_R)), 
                0.5
            )
        )
    )
    flux_Energy = nested_array_subtraction(
        flux_Energy, 
        nested_array_mult(
            C, 
            nested_constant_mult(
                nested_array_subtraction( en_L, en_R ), 
                0.5
            )
        )
    )

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy


##########################################################
# AS FEW LOOPS AS POSSIBLE
#########################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def getFlux(double[:,:] rho_L, double[:,:] rho_R, double[:,:] vx_L, double[:,:] vx_R, double[:,:] vy_L, double[:,:] vy_R, double[:,:] P_L, double[:,:] P_R, double gamma):
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

    cdef unsigned int N = len(rho_L)
    
    cdef double[:,:] flux_Mass = np.zeros((N, N), dtype=np.double)
    cdef double[:,:] flux_Momx = np.zeros((N, N), dtype=np.double)
    cdef double[:,:] flux_Momy = np.zeros((N, N), dtype=np.double)
    cdef double[:,:] flux_Energy = np.zeros((N, N), dtype=np.double)

    cdef unsigned int i, j

    for i in range(N):
        for j in range(N):
            en_L = P_L[i, j]/(gamma-1)+0.5*rho_L[i, j] * (vx_L[i, j]*vx_L[i, j] + vy_L[i, j]*vy_L[i, j])
            en_R = P_R[i, j]/(gamma-1)+0.5*rho_R[i, j] * (vx_R[i, j]*vx_R[i, j] + vy_R[i, j]*vy_R[i, j])

            rho_star = 0.5*(rho_L[i, j] + rho_R[i, j])
            momx_star = 0.5*(rho_L[i, j] * vx_L[i, j] + rho_R[i, j] * vx_R[i, j])
            momy_star = 0.5*(rho_L[i, j] * vy_L[i, j] + rho_R[i, j] * vy_R[i, j])
            en_star = 0.5*(en_L + en_R)

            P_star = (gamma-1)*(en_star - 0.5*(momx_star**2+momy_star**2)/rho_star)

            # flux_Mass[i, j] = momx_star
            # flux_Momx[i, j] = momx_star**2/rho_star + P_star
            # flux_Momy[i, j] = momx_star * momy_star/rho_star
            # flux_Energy[i, j] = (en_star+P_star) * momx_star/rho_star

            C = max(
                math.sqrt(gamma*P_L[i, j]/rho_L[i, j]) + abs(vx_L[i, j]), 
                math.sqrt(gamma*P_R[i, j]/rho_R[i, j]) + abs(vx_R[i, j])
            )

            flux_Mass[i, j] = momx_star - (C * 0.5 * (rho_L[i, j] - rho_R[i, j]))
            flux_Momx[i, j] = (momx_star**2/rho_star + P_star) - (C * 0.5 * (rho_L[i, j] * vx_L[i, j] - rho_R[i, j] * vx_R[i, j]))
            flux_Momy[i, j] = (momx_star * momy_star/rho_star) - (C * 0.5 * (rho_L[i, j] * vy_L[i, j] - rho_R[i, j] * vy_R[i, j]))
            flux_Energy[i, j] = ((en_star+P_star) * momx_star/rho_star) - (C * 0.5 * (en_L - en_R))
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy




################################################
# USING RAW C CODE 
################################################
cdef extern from "flux.c":
    void compute_flux(double* rho_L, double* rho_R, double* vx_L, double* vx_R, double* vy_L, double* vy_R, double* P_L, double* P_R, double gamma, double* flux_Mass, double* flux_Momx, double* flux_Momy, double* flux_Energy, int dim)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def getFluxRawC(double[:,:] rho_L, double[:,:] rho_R, double[:,:] vx_L, double[:,:] vx_R, double[:,:] vy_L, double[:,:] vy_R, double[:,:] P_L, double[:,:] P_R, double gamma):
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

    cdef unsigned int N = rho_L.shape[0]
    
    cdef double[:,:] flux_Mass = np.zeros((N, N), dtype=np.double, order='C')
    cdef double[:,:] flux_Momx = np.zeros((N, N), dtype=np.double, order='C')
    cdef double[:,:] flux_Momy = np.zeros((N, N), dtype=np.double, order='C')
    cdef double[:,:] flux_Energy = np.zeros((N, N), dtype=np.double, order='C')

    compute_flux(&rho_L[0, 0], &rho_R[0, 0], &vx_L[0, 0], &vx_R[0, 0], &vy_L[0, 0], &vy_R[0, 0], &P_L[0, 0], &P_R[0, 0], gamma, &flux_Mass[0, 0], &flux_Momx[0, 0], &flux_Momy[0, 0], &flux_Energy[0, 0], N)
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy