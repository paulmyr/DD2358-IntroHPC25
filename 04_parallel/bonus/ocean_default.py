import numpy as np
import matplotlib.pyplot as plt
import pyvtk 
from timeit import default_timer as timer

# Grid size 
grid_size = 200
TIME_STEPS = 100

def save_to_vtk(filename, u_velocity, v_velocity, temperature):
    """
    Save the velocity and the temperature grid to a VTK file.
    The velocity is saved as a vector, and the temperature is saved as a scalar
    """
    # Convert conserved variables to primitive variables

    # Grid size
    nx, ny = u_velocity.shape

    # Flatten data for VTK
    u_flat = u_velocity.T.flatten()
    v_flat = v_velocity.T.flatten()
    temp_flat = temperature.T.flatten()

    # Create VTK structure
    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredPoints([nx, ny, 1]),  # 2D structured grid
        pyvtk.PointData(
            pyvtk.Scalars(temp_flat, name="temperature"),  # Temperature as a scalar field
            pyvtk.Vectors(np.column_stack((u_flat, v_flat, np.zeros_like(u_flat))), name="velocity"),  # Velocity as vectors
        )
    )
    vtk_data.tofile(filename)
    print(f"Saved VTK file: {filename}")


def run_simulation_default(deterministic=False, profile_time=False, num_iters=TIME_STEPS):
    """
    Runs the ocean currents simulation. 

    deterministic: Is to be set to True if we want the final result to be consistent across multiple calls.
                   This sets the seed for the initial grid values in the experiment. Defaults to False.
    
    profile_time:  Is to be set to True if we want to measure the time (in seconds) of the main loop that 
                   updates the ocean in a loop. Note that only the main loop is timed, so the time of initialization of 
                   grids is not considered here. If set to True, then no print statements are generated and output is not
                   saved to any VTK files.

    num_iters:     The numer of times the ocean is to be updated. Defaults to TIME_STEPS defined above in the file.
                   NOTE: SHOULD BE A MULTIPLE OF 10 TO KEEP WITH NEATNESSS :) 

    Returns the velocity vector (x and y directions, separately), and the temperature. If the profile_time is True, then the
    time is returned (in seconds) as the 4th value. Otherwise, the 4th value is 0. 
    """
    if deterministic:
        np.random.seed(42)

    # Initialize temperature field (random values between 5C and 30C)
    temperature = np.random.uniform(5, 30, size=(grid_size, grid_size))

    # Initialize velocity fields (u: x-direction, v: y-direction)
    u_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))
    v_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))

    # Initialize wind influence (adds turbulence)
    wind = np.random.uniform(-0.5, 0.5, size=(grid_size, grid_size))

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

    def update_ocean(u, v, temperature, wind, alpha=0.1, beta=0.02):
        """Updates ocean velocity and temperature fields using a simplified flow model."""    
        u_new = u + alpha * laplacian(u) + beta * wind
        v_new = v + alpha * laplacian(v) + beta * wind
        temperature_new = temperature + 0.01 * laplacian(temperature)  # Small diffusion
        return u_new, v_new, temperature_new

    # Run the simulation
    # Output Count used to differentiate VTK files
    output_count = 1

    # These values are used to measure the time for running the iteartions/updates.
    t1 = t2 = -1
    if profile_time:
        t1 = timer()

    for t in range(num_iters):
        u_velocity, v_velocity, temperature = update_ocean(u_velocity, v_velocity, temperature, wind)

        # Only print and save outputs to VTK if we are not profiling the time. 
        if (not profile_time) and (t % 10 == 0 or t == num_iters - 1):
            vtk_filename = f"vtk/frame_{output_count:03d}.vtk"
            save_to_vtk(vtk_filename, u_velocity, v_velocity, temperature)
            print(f"Time Step {t}: Ocean currents updated.")
            output_count += 1

    if profile_time:
        t2 = timer()


    return u_velocity, v_velocity, temperature, (t2 - t1)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# UNCOMMENT THE SECTION BELOW TO RUN THE SIMULATION, SAVE TO VTK FILES, AND GET A PLOT AT THE END.$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# u_velocity, v_velocity, temperature, _ = run_simulation_default(deterministic=True)

# # Plot the velocity field
# plt.figure(figsize=(6, 5))
# plt.quiver(u_velocity[::10, ::10], v_velocity[::10, ::10])
# plt.title("Ocean Current Directions")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.show()

# # Plot temperature distribution
# plt.figure(figsize=(6, 5))
# plt.imshow(temperature, cmap='coolwarm', origin='lower')
# plt.colorbar(label="Temperature (°C)")
# plt.title("Ocean Temperature Distribution")
# plt.show()

# print("Simulation complete.")
