# Dask Dashboard Overview and VTK Visualization
This contains some videos of the Dask Dashboard from the Dask-Parallelization of Multiple Simulations of the Wildfire Burning implementation. This was recorded on a 2021
M1 Macbook Pro (16''), which has 10 cores. The default configuration for the `Client()` form `dask.distributed` were used, which on the aforementioned system was 
**5 workers, each with 2 threads**. The recordings below are from running 5 simulations in parallel using `dask.delayed` and the method described in the report.

Along with this, we also present a video of the VTK Visualization in Paraview that shows the fire-spread over time. This was done on the serial implementation on one seeded simulation (refer to report for more)

## The Status Panel
This is a video of the `Status` Panel of Dask Dashboard during the simulation

https://github.com/user-attachments/assets/c4db3710-7ff1-403f-9a8c-131caf144cfe

## The Workers Panel
This is a video of the `Workers` Panel of the Dask Dashboard during the simulation

https://github.com/user-attachments/assets/3b1dd300-e1c0-44ab-bb08-e7a06a6f8734

## The VTK Visualization on Paraview
The video below shows the spread of the fire over time. This was obtained from the serial implementation, running one simulation with a seed of 1. Refer to the `wildfiremontecarloserial.py` file for more. For the purpose of this visualization, the number of days for which the simulation runs was increased from 60 to 500.

https://github.com/user-attachments/assets/f8969c56-4020-41b6-841b-55f8595da530

