"""Julia set generator without optional PIL-based image drawing"""
import numpy as np
import time
from timeit import default_timer as timer
import cpuprofiler

from functools import wraps

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193

# decorator to time
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        diff = t2 - t1
        #print(f"@timefn: {fn.__name__} took {diff} seconds")
        return result, diff
    return measure_time


# @profile
def calc_pure_python(desired_width, max_iterations):
    """Create a list of complex coordinates (zs) and complex parameters (cs),
    build Julia set"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of coordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our
    # function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    print(f"width:\t{desired_width}\noutput:\t{sum(output)}")

    if desired_width == 1000:
        # This sum is expected for a 1000^2 grid with 300 iterations
        assert sum(output) == 33219980
    elif desired_width == 10000:
        assert sum(output) == 3323787446
    else:
        print("no asserts for this dimension...")


# @profile
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output

@timefn
def calc_pure_python_profiled(desired_width, max_iterations):
    """Create a list of complex coordinates (zs) and complex parameters (cs),
    build Julia set"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of coordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our
    # function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    #print("Length of x:", len(x))
    #print("Total elements:", len(zs))
    output, t = calculate_z_serial_purepython_profiled(max_iterations, zs, cs)
    # print(f"width:\t{desired_width}\noutput:\t{sum(output)}")

    if desired_width == 1000:
        # This sum is expected for a 1000^2 grid with 300 iterations
        assert sum(output) == 33219980
    elif desired_width == 10000:
        assert sum(output) == 3323787446
    else:
        print("no asserts for this dimension...")

    return t


@timefn
def calculate_z_serial_purepython_profiled(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output


def do_profiling():
    n = 20
    desired_width = 1000
    max_iterations = 300
    t_inner = np.zeros(n)
    t_outer = np.zeros(n)

    for i in range(n):
        t1, t2 = calc_pure_python_profiled(desired_width=desired_width, max_iterations=max_iterations)
        t_inner[i] = t2
        t_outer[i] = t1

    print(f"number of runs: {n}\ndesired width: {desired_width}\nmax iterations: {max_iterations}\n\n" + \
          f"calc_pure_python\n\tmean: {np.mean(t_outer)} s\n\tstd: {np.std(t_outer, dtype=np.float64)} s\n\n" + \
          f"calculate_z_serial_purepython\n\tmean: {np.mean(t_inner)} s\n\tstd: {np.std(t_inner, dtype=np.float64)} s")

def do_cpu_usage_estimation():
    cpu = cpuprofiler.CPUProfiler(granularity=5, interval=1, experiment_name="JuliaSet")

    cpu.initiate_observation()
    calc_pure_python(desired_width=5000, max_iterations=300)
    cpu.end_observation()

    cpu.generate_usage_graph()
    cpu.generate_tabular_summary()


if __name__ == "__main__":
    # Calculate the Julia set using a pure Python solution with
    # reasonable defaults for a laptop

    # do_profiling()
    # t1 = timer()
    # calc_pure_python(desired_width=100, max_iterations=300)
    # t2 = timer()
    # print(f"total calculation time: {t2 - t1} s")

    do_cpu_usage_estimation()
