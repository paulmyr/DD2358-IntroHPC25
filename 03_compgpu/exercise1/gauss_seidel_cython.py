import numpy as np
import gs_cython_solver

# Constants
NUM_ITERATIONS = 1000


def solve_posisson_cython(grid, num_iters):
    """
    The poisson gauss-seidel solver that uses the cython library function.
    """
    return gs_cython_solver.solve_posisson_cython(grid, num_iters)
    

# COMMENTING OUT TO NOT INTERFERE WITH TESTS
# if __name__ == "__main__":
#     solve_posisson_cython(512, NUM_ITERATIONS)