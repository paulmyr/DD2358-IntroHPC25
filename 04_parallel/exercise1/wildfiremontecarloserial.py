import numpy as np
import matplotlib.pyplot as plt
import pyvtk 
import random

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time

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

def save_data_to_vtk(filename, forest):
    """
    Save the forest data to VTK files
    """
    forest_np = np.array(forest)
    nx, ny = forest_np.shape

    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredPoints([nx, ny]),
        pyvtk.PointData(
            pyvtk.Scalars(forest_np.flatten(), name="forest_status") # The status of the forest is a scalar field
        )
    )
    vtk_data.tofile(filename)
    print(f"Saved VTK File: {filename}")


def simulate_wildfire(seed=None, continuous_plot=False, save_to_vtk=False):
    """Simulates wildfire spread over time."""
    custom_rand = random.Random(seed)
    forest, burn_time = initialize_forest(custom_rand)

    fire_spread = []  # Track number of burning trees each day
    output_count=1 # The VTK file output count

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

        if save_to_vtk and ((day % 5 == 0 or day == DAYS - 1)):
            vtk_filename = f"vtk/frame_{output_count:03d}.vtk"
            save_data_to_vtk(vtk_filename, forest)
            output_count += 1

        # Plot grid every 5 days
        if continuous_plot and (day % 5 == 0 or day == DAYS - 1):
            plt.figure(figsize=(6, 6))
            plt.imshow(forest, cmap='viridis', origin='upper')
            plt.title(f"Wildfire Spread - Day {day}")
            plt.colorbar(label="State: 0=Empty, 1=Tree, 2=Burning, 3=Ash")
            plt.show()

    return fire_spread


def run_and_save_vtk():
    """
    Run a SINGLE simulation, saving the forest output periodically to vtk files
    """
    # We don't care about the result returned here
    # For the VTK files present in the repository, number of days was changed to 500.
    simulate_wildfire(seed=1, save_to_vtk=True)
    


def run_n_simulations_default(n_simulations=1, seeds=None, no_print=False, show_line_plot=False):
    """
    Runs n_simulations SERIALLY, and returns an average of the fire_spread over time that is obtained
    from each run. 
    seeds refers to an array of seeds, which MUST match the number of simulations. If provided, it is used
    to help with reproducability of the results
    """
    if seeds and len(seeds) != n_simulations:
        print("Number of seeds must match number of simulations")
        return

    all_results = []
    for i in range(n_simulations):
        curr_seed = None if not seeds else seeds[i]
        # Continuous plotting does not make a lot of sense when multiple simulations are being run.
        # So, we prevent doing that here.
        curr_spread = simulate_wildfire(curr_seed, continuous_plot=False)
        all_results.append(curr_spread)
        if not no_print:
            print(f"[SERIAL] Simulation {i} with seed {curr_seed} completed.")
    
    if show_line_plot:
        for i in range(n_simulations):
            plt.plot(range(len(all_results[i])), all_results[i], label=f"Simulation no: {i}",  alpha=0.3, linestyle="--")

    # Return the average of the individual simulations, where the average is taken over the columns
    # This is because each column represents a single day. 
    return np.array(all_results).mean(axis=0)

if __name__ == "__main__":
    # Run Multiple Simulations Serially
    # fire_spread_over_time = run_n_simulations_default(n_simulations=5, seeds=[i for i in range(5)], show_line_plot=True)

    # # Plot results
    # plt.plot(range(len(fire_spread_over_time)), fire_spread_over_time, label="Burning Trees")
    # plt.xlabel("Days")
    # plt.ylabel("Number of Burning Trees")
    # plt.title("Wildfire Spread Over Time [Serial]")
    # plt.legend()
    # plt.show()

    # Uncomment this to save to VTK files
    run_and_save_vtk()