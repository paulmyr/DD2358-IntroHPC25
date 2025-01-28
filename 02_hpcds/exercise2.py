import os

import numpy as np
import array as arr
import csv
from line_profiler import profile
from functools import wraps
from timeit import default_timer as timer


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        fn(*args, **kwargs)
        t2 = timer()
        diff = t2 - t1
        #print(f"@timefn: {fn.__name__} took {diff} seconds")
        return diff
    return measure_time

# @profile
@timefn
def mult_add_np(a, b, c):
    """
    Task 2.1, 2.5
    """
    return c + a @ b

# @profile
@timefn
def mult_add_py(a, b, c):
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            for k in range(0, len(a)):
                c[i][j] = c[i][j] + a[i][k] * b[k][j]
    return c

# @profile
@timefn
def mult_add_np_large(a, b, c):
    """
    Task 2.1
    """
    return c + a @ b

# @profile
@timefn
def mult_add_py_large(a, b, c):
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            for k in range(0, len(a)):
                c[i][j] = c[i][j] + a[i][k] * b[k][j]
    return c

def mult_arrays(n, type):
    ret = []
    np.random.seed(seed=1234)
    a = np.random.randint(low=0, high=2**20-1, size=(n,n))
    b = np.random.randint(low=0, high=2**20-1, size=(n,n))
    c = np.random.randint(low=0, high=2**20-1, size=(n,n))
    if type == 'np':
        pass
    elif type == 'array':
        a, b, c = arr.array(a), arr.array(b), arr.array(c)
    elif type == 'list':
        a, b, c = a.tolist(), b.tolist(), c.tolist()
    return a, b, c

if __name__ == '__main__':
    a_small, b_small, c_small = mult_arrays(128, "np")
    a_large, b_large, c_large = mult_arrays(256, "np")

    # a_small, b_small, c_small = mult_arrays(128, "array")
    # a_large, b_large, c_large = mult_arrays(256, "array")
    #
    # a_small, b_small, c_small = mult_arrays(128, "list")
    # a_large, b_large, c_large = mult_arrays(256, "list")


    py_sample = [mult_add_py(a_small, b_small, c_small) for i in range(20)]
    np_sample = [mult_add_np(a_small, b_small, c_small) for i in range(20)]

    py_avg, py_var, py_std = np.average(py_sample), np.var(py_sample), np.std(py_sample)
    np_avg, np_var, np_std = np.average(np_sample), np.var(np_sample), np.std(np_sample)

    mult_add_py_large(a_large, b_large, c_large)
    mult_add_np_large(a_large, b_large, c_large)

    py_sample_large = [mult_add_py(a_small, b_small, c_small) for i in range(20)]
    np_sample_large = [mult_add_np(a_small, b_small, c_small) for i in range(20)]

    py_avg_large, py_var_large, py_std_large = np.average(py_sample), np.var(py_sample), np.std(py_sample)
    np_avg_large, np_var_large, np_std_large = np.average(np_sample), np.var(np_sample), np.std(np_sample)


    with open("exercise2_measurements.csv", "w") as f:
        f.write("Time: avg, var, std\n")
        f.write(repr([py_avg, py_var, py_std]))
        f.write(repr([py_avg_large, py_var_large, py_std_large]))
        f.close()