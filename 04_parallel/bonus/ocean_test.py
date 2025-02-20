from ocean_default import run_simulation_default
from ocean_dask import run_simulation_dask
import numpy as np

print("============= Computing Default Ocean ==============")
u_default, v_default, temp_default = run_simulation_default()
print("================= Computing Dask Ocean ====================")
u_dask, v_dask, temp_dask = run_simulation_dask()

print(np.array_equal(u_default, u_dask))
print(np.array_equal(v_default, v_dask))
print(np.array_equal(temp_default, temp_dask))

# print(u_default)
# print(u_dask)