
import array as arr
import csv
import numpy as np
import os
import scipy.stats as st
from functools import wraps
from line_profiler import profile
from math import sqrt
from timeit import default_timer as timer

TIMEIT = False


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        if TIMEIT:
            t1 = timer()
            fn(*args, **kwargs)
            t2 = timer()
            diff = t2 - t1
            #print(f"@timefn: {fn.__name__} took {diff} seconds")
            return diff
        else:
            return fn(*args, **kwargs)
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
    TIMEIT = True

    a_small, b_small, c_small = mult_arrays(128, "np")
    a_large, b_large, c_large = mult_arrays(256, "np")

    # a_small, b_small, c_small = mult_arrays(128, "array")
    # a_large, b_large, c_large = mult_arrays(256, "array")
    #
    # a_small, b_small, c_small = mult_arrays(128, "list")
    # a_large, b_large, c_large = mult_arrays(256, "list")
    #

    if TIMEIT:
        py_sample = [mult_add_py(a_small, b_small, c_small) for i in range(20)]
        np_sample = [mult_add_np(a_small, b_small, c_small) for i in range(20)]

        py_avg, py_var, py_std = np.average(py_sample), np.var(py_sample), np.std(py_sample)
        py_ci = st.norm.interval(0.90, loc=py_avg, scale=py_std/sqrt(len(py_sample)))

        np_avg, np_var, np_std = np.average(np_sample), np.var(np_sample), np.std(np_sample)
        np_ci = st.norm.interval(0.90, loc=np_avg, scale=np_std/sqrt(len(np_sample)))

        py_sample_large = [mult_add_py_large(a_large, b_large, c_large) for i in range(20)]
        np_sample_large = [mult_add_np_large(a_large, b_large, c_large) for i in range(20)]

        py_avg_large, py_var_large, py_std_large = np.average(py_sample_large), np.var(py_sample_large), np.std(py_sample_large)
        py_ci_large = st.norm.interval(0.90, loc=py_avg_large, scale=py_std_large/sqrt(len(py_sample_large)))
        np_avg_large, np_var_large, np_std_large = np.average(np_sample_large), np.var(np_sample_large), np.std(np_sample_large)
        np_ci_large = st.norm.interval(0.90, loc=np_avg_large, scale=np_std_large/sqrt(len(np_sample_large)))

        with open("data.dat", "w") as f:
            # f.write("name; avg; var; std\n")
            f.write(f"\"np small\"\t{np_avg}\t{np_var}\t{np_std}\t{str(np_ci[0])} {str(np_ci[1])}\n")
            f.write(f"\"np large\"\t{np_avg_large}\t{np_var_large}\t{np_std_large}\t{str(np_ci_large[0])} {str(np_ci_large[1])}\n")
            f.write(f"\"py small\"\t{py_avg}\t{py_var}\t{py_std}\t{str(py_ci[0])} {str(py_ci[1])}\n")
            f.write(f"\"py large\"\t{py_avg_large}\t{py_var_large}\t{py_std_large}\t{str(py_ci_large[0])} {str(py_ci_large[1])}\n")
            f.close()
