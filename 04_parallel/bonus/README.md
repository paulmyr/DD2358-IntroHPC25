# Dask Dashboard Data & VTK Visualizations
Here, we present video recordings for observations of our Dask Dashboard that we spoke about in the report. The aim of these recordings here is to provide more context for our conclusions drawn 
in the report, so that the reader can see the evolution of parameters (such as worker memory usage, job status, etc) for themselves.

We also present a video of the VTK Visualization in Paraview that shows the ocean temperature and velocity changes over time for a simulation. This was done on the serial implementation on one seeded simulation (refer to report for more).

## Worker Memory and CPU Usage (for Different Chunk Sizes)
Below, we provide video recordings of the `Worker` tab of the Dask Dashboard. Of partcicular interest to us, according to the assignment handout, were the **CPU Usage** and the **Memory Usage**
of each of the 5 workers. We provide recordings for the `Worker` tab for different chunk sizes below. Only the recordings are provided here. Please refer to the report for our conclusions and to read us
talk more about these observations. 

### Chunk Size 50

https://github.com/user-attachments/assets/bf2656f6-4402-403f-bf2e-c99f5c79d156

### Chunk Size 100

https://github.com/user-attachments/assets/f1cde640-b0d4-4b07-9dd7-5e993881f173

### Chunk Size 200

https://github.com/user-attachments/assets/b9ccf27f-1664-4dd9-bc1a-510aad861e97

## Job Status (for Different Chunk Sizes)
Below, we provide video recordings for the `Progress` tab (available from under the `More...` tab in the default Dask Dashboard screen). Information on what this tab displays can be found [here](https://docs.dask.org/en/latest/dashboard.html#progress).
We utilized the information from this tab to note if there were any `queued` tasks. Again, we request you to refer to the report where we do a brief discussion of the observations here.

### Chunk Size 50



https://github.com/user-attachments/assets/8e891f55-b1e0-4837-906f-9b8e1b399ef0



### Chunk Size 100



https://github.com/user-attachments/assets/3b9c62da-c1b2-4559-b9f4-63c3da3f86f1



### Chunk Size 200


https://github.com/user-attachments/assets/5878516b-7339-44d7-9359-c0cd72bc5c5e



## VTK Visualizations
Below we show videos of the changes in the ocean currents **temperature** and **velocity** over time. This was obtained from the serial implementation. Refer to the `ocean_default.py` file for more. For the purpose of this visualization, the number of days for which the simulation runs was increased from 100 to 1000.

### Temperature Visualization



https://github.com/user-attachments/assets/725b6eef-451a-4084-a1db-8283343204d4



### Velocity Visualization



https://github.com/user-attachments/assets/b4e4dd18-e1a2-469a-8229-baa562bd559d

