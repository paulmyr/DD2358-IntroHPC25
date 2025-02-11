import numpy as np
cimport numpy as np

#cython: boundscheck=False
def gauss_seidel(double[:,:] f):
    """
    The simple, base, version of the gauss_seidel solver provided
    to us in the assignment description.
    """
    cdef unsigned int i, j

    cdef double[:,:] newf = f.copy()
    
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                   newf[i+1,j] + newf[i-1,j])
    
    return newf

#cython: boundscheck=False
def solve_posisson_cython(double[:,:] base_grid, unsigned int num_iters):
    """
    Calls the gauss_seidel method on the grid "num_iters" number of times.
    The grid should be initialized properly (ie, must be a square grid with 0s
    at the boundary.)

    The grid utilized here is initialzied using numpy
    """
    cdef unsigned int i

    for i in range(num_iters):
        base_grid = gauss_seidel(base_grid)
    
    return base_grid