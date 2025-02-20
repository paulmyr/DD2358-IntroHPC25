import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

# Grid size 
grid_size = 200
TIME_STEPS = 100
CHUNK_SIZE = 100


def run_simulation_dask():
    np.random.seed(42)

    # Initialize temperature field (random values between 5C and 30C)
    temperature = da.from_array(np.random.uniform(5, 30, size=(grid_size, grid_size)), chunks=(CHUNK_SIZE, CHUNK_SIZE))

    # Initialize velocity fields (u: x-direction, v: y-direction)
    u_velocity = da.from_array(np.random.uniform(-1, 1, size=(grid_size, grid_size)), chunks=(CHUNK_SIZE, CHUNK_SIZE))
    v_velocity = da.from_array(np.random.uniform(-1, 1, size=(grid_size, grid_size)), chunks=(CHUNK_SIZE, CHUNK_SIZE))

    # Initialize wind influence (adds turbulence)
    wind = da.from_array(np.random.uniform(-0.5, 0.5, size=(grid_size, grid_size)), chunks=(CHUNK_SIZE, CHUNK_SIZE))

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
        velocity_new = velocity + alpha * laplacian(velocity) + beta * wind
        return velocity_new

    def update_temp(temperature):
        temp_new = temperature + 0.01 * laplacian(temperature)
        return temp_new

    for t in range(TIME_STEPS):
        # TODO: This seems to work (see ocean_test.py), BUT, this only uses "map_overlap". 
        #       The handout also suggests that map_block would/could be used. Am I doing something wrong here? 
        u_velocity = da.map_overlap(update_velocity, u_velocity, wind, depth=1, boundary="periodic", dtype=da.float64)
        v_velocity = da.map_overlap(update_velocity, v_velocity, wind, depth=1, boundary="periodic", dtype=da.float64)
        temperature = da.map_overlap(update_temp, temperature, depth=1, boundary="periodic", dtype=da.float64)

        if t % 10 == 0 or t == TIME_STEPS - 1:
            print(f"Time Step {t}: Ocean currents updated.")

    u_result = u_velocity.compute()
    v_result = v_velocity.compute()
    temperature_result = temperature.compute()

    return u_result, v_result, temperature_result


# u_result, v_result, temperature_result = run_simulation_dask()

# Plot the velocity field
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
