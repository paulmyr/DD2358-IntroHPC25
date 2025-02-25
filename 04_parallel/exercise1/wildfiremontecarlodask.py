from dask import delayed
import dask
from dask.distributed import Client
import dask.array as da

import numpy as np
import matplotlib.pyplot as plt
import random
from timeit import default_timer as timer

import pdb

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time
NUM_SIMULATIONS = 5 # The number of simulations to run in parallel

# State definitions
EMPTY = 0  # No tree
TREE = 1  # Healthy tree
BURNING = 2  # Burning tree
ASH = 3  # Burned tree


def initialize_forest(custom_rand):
    """Creates a forest grid with all trees and ignites one random tree."""

    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns

    # Ignite a random tree
    x, y = custom_rand.randint(0, GRID_SIZE - 1), custom_rand.randint(0, GRID_SIZE - 1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning

    return forest, burn_time


def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors


@delayed
def simulate_wildfire(seed=None, continuous_plot=False):
    """Simulates wildfire spread over time."""
    custom_rand = random.Random(seed)
    forest, burn_time = initialize_forest(custom_rand)

    fire_spread = []  # Track number of burning trees each day

    for day in range(DAYS):
        new_forest = forest.copy()

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time

                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH

                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and custom_rand.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1

        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))

        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            if day < DAYS - 1:
                fire_spread += [0 for _ in range(DAYS - 1 - day)]
            break

        # Plot grid every 5 days
        if continuous_plot and (day % 5 == 0 or day == DAYS - 1):
            plt.figure(figsize=(6, 6))
            plt.imshow(forest, cmap='viridis', origin='upper')
            plt.title(f"Wildfire Spread - Day {day}")
            plt.colorbar(label="State: 0=Empty, 1=Tree, 2=Burning, 3=Ash")
            plt.show()

    # Returning a numpy array here to save some time on Dask Computation
    return np.array(fire_spread)

def run_n_simulations_dask(n_simulations=NUM_SIMULATIONS, seeds=None, no_print=False, show_line_plot=False, chunk_size=None):
    """
    Runs n_simulations PARALLELY, using Dask, and returns an average of the fire_spread over time that is 
    obtained from each run.
    seeds refers to an array of seeds, which MUST match the number of simulations. If provided, it is used
    to help with reproducability of the results
    """
    if seeds and len(seeds) != n_simulations:
        print("Number of seeds must match number of simulations")
        return

    # Set chunk size if not provided. We default to 1 chunk
    final_chunk_size = chunk_size
    if not chunk_size:
        final_chunk_size = (n_simulations, DAYS)

    actual_seeds = [None]*n_simulations if not seeds else seeds

    # Array of delayed Dask Tasks
    tasks = [simulate_wildfire(i) for i in actual_seeds]
    # Create this intermediate data structure so that we can eventually have a 2d Dask Array
    # Note that all tasks in `tasks` *will* return a 1d array *once* they are computed. However,
    # since `simulate_wildfire` is a delayed computation, we use `from_delayed` here instead of `from_array`
    output_as_dask = [da.from_delayed(burning, shape=(DAYS,), dtype=da.float32) for burning in tasks]
    # We specify to dask how we want it to combine the outputs of the simulations. We want the columns to match
    # ie, first index of sim1 to be "below" first index of sim2, etc. So, use axis=0
    dask_array = da.stack(output_as_dask, axis=0)
    # This is used for testing how chunk-size impacts computation speed. We chunk our arrays so that averages for 
    # each day(s) can be computed in parallel.
    rechunked_result = dask_array.rechunk(final_chunk_size)
    # Finally, compute the average along the columns and get the result.
    result = rechunked_result.mean(axis=0).compute()

    if show_line_plot:
        # Using matplotlib in the delayed simulate_wildfire function leads to an exception indicating that
        # "NSWindow should only be instantiated on the main thread!". For this reason, plot them here instead
        individual_results = dask_array.compute()
        for i in range(n_simulations):
            plt.plot(range(len(individual_results[i])), individual_results[i], label=f"Simulation no: {i}",  alpha=0.3, linestyle="--")
    
    return result

# Run simulation
if __name__ == "__main__":
    
    # Using default number of workers
    client = Client()
    print(f"Client dashboard available at: f{client.dashboard_link}")
    input("Press a key once you have gone to the dashboard...") 

    fire_spread_over_time = run_n_simulations_dask(n_simulations=5, seeds=[i for i in range(5)], show_line_plot=True)

    # Plot results
    plt.plot(range(len(fire_spread_over_time)), fire_spread_over_time, label="Average Burning Trees")
    plt.xlabel("Days")
    plt.ylabel("Number of Burning Trees")
    plt.title("Wildfire Spread Over Time [Dask]")
    plt.legend()
    plt.show()