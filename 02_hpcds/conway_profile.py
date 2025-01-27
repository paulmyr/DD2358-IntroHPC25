"""
conway.py 

A simple Python/matplotlib implementation of Conway's Game of Life.

Author: Mahesh Venkitachalam

Updated for DD2358 to help with profiling by removing extra code (such as for command line args
or for animations)
"""

import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

ON = 255
OFF = 0
vals = [ON, OFF]


def random_grid(n):
    """returns a grid of NxN random values"""
    # TODO: Should we be setting a seed here? 
    return np.random.choice(vals, n * n, p=[0.2, 0.8]).reshape(n, n)



def update(grid, n):
    """
    Updates the grid by applying the rules of Conway's Game of Life. The rules for the game
    and more information on it can be found here:
    https://www.geeksforgeeks.org/conways-game-life-python-implementation/
    """
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line
    new_grid = grid.copy()
    for i in range(n):
        for j in range(n):
            # compute 8-neghbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulaton takes place on a toroidal surface.
            total = int(
                (
                    grid[i, (j - 1) % n]
                    + grid[i, (j + 1) % n]
                    + grid[(i - 1) % n, j]
                    + grid[(i + 1) % n, j]
                    + grid[(i - 1) % n, (j - 1) % n]
                    + grid[(i - 1) % n, (j + 1) % n]
                    + grid[(i + 1) % n, (j - 1) % n]
                    + grid[(i + 1) % n, (j + 1) % n]
                )
                / 255
            )
            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = OFF
            else:
                if total == 3:
                    new_grid[i, j] = ON

    grid[:] = new_grid[:]


def update_vectorized(grid, n):
    "The vectorized version of the grid update in conway's game of life"
    update(grid, n)



def profile_vanilla_computation():
    # grid sizes increase by powers of 2
    grid_sizes = [64, 128, 256, 512, 1024]
    # times for each of the computations (in seconds)
    times = []
    # number of iterations for which the updates will happen
    max_iters = 100
    # The number of times the run will be performed for each grid size.
    # The final running time reported will be the average of these runs
    num_runs = 5

    for curr_size in grid_sizes:
        total_time = 0

        for _ in range(num_runs):
            curr_grid = random_grid(curr_size)

            t1 = timer()
            for _ in range(max_iters):
                update(curr_grid, curr_size)
            t2 = timer()

            total_time += (t2 - t1)
        
        times.append(total_time / num_runs)
        print(f"Avg Time for {curr_size}x{curr_size}: {total_time / num_runs}")

    plot_data(times, grid_sizes)

def plot_data(running_times, grid_sizes):
    # Create a line plot with dots at each data point
    plt.plot(grid_sizes, running_times, marker='o', linestyle='-', color='b')

    # Adding labels to the axes
    plt.xlabel('Grid Size')
    plt.ylabel('Running Time (s)')

    # Adding a title to the plot
    plt.title('Time Taken for 200 Updates for Varying Grid Sizes')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    profile_vanilla_computation()
