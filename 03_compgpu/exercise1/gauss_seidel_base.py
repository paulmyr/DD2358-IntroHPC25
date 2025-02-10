import numpy as np

# Constants
NUM_ITERATIONS = 1000

def gauss_seidel(f):
    """
    The simple, base, version of the gauss_seidel solver provided
    to us in the assignment description.
    """
    newf = f.copy()
    
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                   newf[i+1,j] + newf[i-1,j])
    
    return newf

def solve_posisson(grid_size):
    """
    Initializes a square grid of the provided size with the values on the boundary 
    fixed to 0 and the other values filled with random numbers. Then, calls the 
    gauss_seidel method on the grid NUM_ITERATIONS number of times.

    The grid utilized here is initialzied using numpy
    """
    base_grid = np.zeros((grid_size, grid_size))
    base_grid[1:-1, 1:-1] = np.random.rand(grid_size-2, grid_size-2)

    for _ in range(NUM_ITERATIONS):
        base_grid = gauss_seidel(base_grid)

    

if __name__ == "__main__":
    solve_posisson(512)