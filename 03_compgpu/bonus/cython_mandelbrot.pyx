import numpy as np
cimport numpy as np

#cython: boundscheck=False
def mandelbrot(double complex c, unsigned int max_iter=100):
    """Computes the number of iterations before divergence."""
    # Cython Type Annotations
    cdef double complex z
    cdef unsigned int n
    
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

#cython: boundscheck=False
def mandelbrot_set(unsigned int width, unsigned int height, double x_min, double x_max, double y_min, double y_max, unsigned int max_iter=100):
    """Generates the Mandelbrot set image."""
    # Cython Type Annotations
    cdef unsigned int i, j
    cdef double complex c

    cdef double[:] x_vals = np.linspace(x_min, x_max, width, dtype=np.double)
    cdef double[:] y_vals = np.linspace(y_min, y_max, height, dtype=np.double)
    cdef unsigned int[:,:] image = np.zeros((height, width), dtype=np.uint32)

    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            image[i, j] = mandelbrot(c, max_iter)

    return image