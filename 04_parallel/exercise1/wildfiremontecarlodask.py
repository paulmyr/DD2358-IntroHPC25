from dask import delayed
import dask
from dask.distributed import Client
import dask.array as da

import numpy as np
import matplotlib.pyplot as plt
import random

import pdb

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time
NUM_SIMULATIONS = 10 # The number of simulations to run in parallel
CHUNK_SIZE = NUM_SIMULATIONS

# State definitions
EMPTY = 0  # No tree
TREE = 1  # Healthy tree
BURNING = 2  # Burning tree
ASH = 3  # Burned tree


def initialize_forest(seed):
    """Creates a forest grid with all trees and ignites one random tree."""
    np.random.seed(seed)
    random.seed(seed)
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns

    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
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
def simulate_wildfire(seed):
    """Simulates wildfire spread over time."""
    # TODO: Randomness does not really make much sense here? See print outputs for why
    forest, burn_time = initialize_forest(seed)
    random.seed(seed)

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
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1

        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))

        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            if day < DAYS - 1:
                fire_spread += [0 for _ in range(DAYS - 1 - day)]
            break
        #
        # # Plot grid every 5 days
        # if day % 5 == 0 or day == DAYS - 1:
        #     plt.figure(figsize=(6, 6))
        #     plt.imshow(forest, cmap='viridis', origin='upper')
        #     plt.title(f"Wildfire Spread - Day {day}")
        #     plt.colorbar(label="State: 0=Empty, 1=Tree, 2=Burning, 3=Ash")
        #     plt.show()

    print(f"----------------------------------------------- \n [random] The grid being returned for seed {seed} is: \n {np.array(fire_spread)} \n -----------------------------------------------")
    return fire_spread


# Run simulation
if __name__ == "__main__":
    client = Client()
    print(client)
    print(client.dashboard_link)
    seeds =[i for i in range(NUM_SIMULATIONS)]

    # The individual simulations to run in parallel, we get a 2D numpy array afterwards
    tasks = delayed(lambda arr: np.array(arr))([simulate_wildfire(i) for i in seeds])
    # Getting a single dask array of with counts for all 60 days across all simulations
    arr = da.from_delayed(tasks, shape=(NUM_SIMULATIONS, DAYS), dtype=da.float32)
    # Rechunk based on columns -- THIS IS WHERE WE CAN CONTROL THE CHUNK SIZE
    arr = arr.rechunk((CHUNK_SIZE, DAYS))
    # Compute the mean of the individual chunks 
    avg = arr.mean(axis=0)
    result = avg.compute()
    print("--------------- Printing Average --------------------")
    print(result)

    # n = result.shape[0]
    # print(n)
    # # Plot results
    # plt.figure(figsize=(8, 5))
    # # for i in range(n):
    # #     plt.plot(np.arange(0, m), result[i], label=f"Simulation no: {i}")
    # plt.plot(np.arange(0,n), result, label="Avg Fire Spread over Time")
    # plt.xlabel("Days")
    # plt.ylabel("Number of Burning Trees")
    # plt.title("Wildfire Spread Over Time")
    # plt.legend()
    # plt.show()
    # dask.visualize(*result)
