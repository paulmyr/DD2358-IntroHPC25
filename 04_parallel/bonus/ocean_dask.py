import numpy as np
import dask.array as da
from dask.distributed import Client
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Grid size 
grid_size = 200
TIME_STEPS = 100
CHUNK_SIZE = 100
NUM_WORKERS = 5


def run_simulation_dask(deterministic=False, profile_time=False, num_iters=TIME_STEPS, chunk_size=CHUNK_SIZE):
    """
    Similar to the `run_simulation_default` function in `ocean_default.py`, but uses DASK AWESOMENESS :)  

    deterministic: Is to be set to True if we want the final result to be consistent across multiple calls.
                   This sets the seed for the initial grid values in the experiment. Defaults to False.
    
    profile_time:  Is to be set to True if we want to measure the time (in seconds) of the main loop that 
                   SCHEDULES the ocean updates in a loop, and the final compute() call that does the computations of the 
                   generated task graph. Note that only the main loop is timed, so the time of initialization of 
                   grids is not considered here. If set to True, then no print statements are generated. 

    num_iters:     The numer of times the ocean is to be updated. Defaults to TIME_STEPS defined above in the file.

    chunk_size:    The size of chunk to use for the dask arrays. Defaults to tO CHUNK_SIZE in this file

    num_workers:   The number of workers to be used by the dask.distributed Client. Defaults to NUM_WORKERS at the top of this file.

    Returns the velocity vector (x and y directions, separately), and the temperature. If the profile_time is True, then the
    time is returned (in seconds) as the 4th value. Otherwise, the 4th value is 0.     
    """
    if deterministic:
        np.random.seed(42)

    # Initialize temperature field (random values between 5C and 30C), and convert them to a dask array with specified chunk_size
    temperature = da.from_array(np.random.uniform(5, 30, size=(grid_size, grid_size)), chunks=(chunk_size, chunk_size))

    # Initialize velocity fields (u: x-direction, v: y-direction), and convert them to a dask array with specified chunk_size
    u_velocity = da.from_array(np.random.uniform(-1, 1, size=(grid_size, grid_size)), chunks=(chunk_size, chunk_size))
    v_velocity = da.from_array(np.random.uniform(-1, 1, size=(grid_size, grid_size)), chunks=(chunk_size, chunk_size))

    # Initialize wind influence (adds turbulence), and convert them to a dask array with specified chunk_size
    wind = da.from_array(np.random.uniform(-0.5, 0.5, size=(grid_size, grid_size)), chunks=(chunk_size, chunk_size))

    def laplacian(field):
        """Computes the discrete Laplacian of a 2D field using finite differences."""
        lap = (
            np.roll(field, shift=1, axis=0) +
            np.roll(field, shift=-1, axis=0) +
            np.roll(field, shift=1, axis=1) +
            np.roll(field, shift=-1, axis=1) -
            4 * field
        )
        return lap

    def update_velocity(velocity, wind, alpha=0.1, beta=0.02):
        """Updates the velocity vector (for both x and y directions)"""
        velocity_new = velocity + alpha * laplacian(velocity) + beta * wind
        return velocity_new

    def update_temp(temperature):
        """Updates the temperature vector"""
        temp_new = temperature + 0.01 * laplacian(temperature)
        return temp_new
    
    # These values are used to measure the time for running the iteartions/updates.
    t1 = t2 = -1
    if profile_time:
        t1 = timer()

    for t in range(num_iters):
        # TODO: This seems to work (see ocean_test.py), BUT, this only uses "map_overlap". 
        #       The handout also suggests that map_block would/could be used. Am I doing something wrong here? 

        # We use the `map_overlap` with a ghost cell depth of 1 so that the rolls performed by the laplican calculation can
        # be done appropriately. Note that we use the "periodic" boundary, as that seems to give the correct output as with this settings,
        # the edges "wrap around", as seems to be the computation performed by the given code in the "ocean_default.py" file.
        u_velocity = da.map_overlap(update_velocity, u_velocity, wind, depth=1, boundary="periodic", dtype=da.float64)
        v_velocity = da.map_overlap(update_velocity, v_velocity, wind, depth=1, boundary="periodic", dtype=da.float64)
        temperature = da.map_overlap(update_temp, temperature, depth=1, boundary="periodic", dtype=da.float64)

        # Only print if we are not profiling time
        if (not profile_time) and (t % 10 == 0 or t == num_iters - 1):
            print(f"Time Step {t}: Ocean currents scheduled.")

    u_result = u_velocity.compute()
    v_result = v_velocity.compute()
    temperature_result = temperature.compute()

    if profile_time:
        t2 = timer()

    return u_result, v_result, temperature_result, (t2-t1)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# USES dask.distributed WITH THE SPECIFIED NUMBER OF WORKERS $
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# if __name__ == "__main__":
#     client = Client(n_workers=6)
#     print("Dask Dashboard running at:", client.dashboard_link)
#     input("Press a key once you have gone to the dashboard...")
#     u_result, v_result, temperature_result = run_simulation_dask(deterministic=True)




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# UNCOMMENT THE SECTION BELOW TO RUN THE SIMULATION, SAVE TO VTK FILES, AND GET A PLOT AT THE END.$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# u_result, v_result, temperature_result, _ = run_simulation_dask(deterministic=True)
# # Plot the velocity field
# plt.figure(figsize=(6, 5))
# plt.quiver(u_result[::10, ::10], v_result[::10, ::10])
# plt.title("Ocean Current Directions")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.show()

# # Plot temperature distribution
# plt.figure(figsize=(6, 5))
# plt.imshow(temperature_result, cmap='coolwarm', origin='lower')
# plt.colorbar(label="Temperature (°C)")
# plt.title("Ocean Temperature Distribution")
# plt.show()

# print("Simulation complete.")
