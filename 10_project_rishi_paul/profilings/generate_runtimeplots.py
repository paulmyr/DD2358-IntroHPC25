import matplotlib.pyplot as plt
import numpy as np

if __name__== "__main__":
    grid_sizes = [2**i for i in range(4, 11)]
    N = 3
    tEnd = 0.1
    wtimes = {
        "Apple MacBook M1 Air, 2020 (baseline)":
            [0.06273317, 0.07764285, 0.13479832, 0.36281343, 1.38195608, 6.00546406, 29.00729133],
        "Apple MacBook M1 Air (Cython)":
            [0.04825463, 0.05819958, 0.09835089, 0.24178517, 0.89203914, 3.72730593, 17.08468867],
        "Apple MacBook M1 Air (Dask)":
            [0.48807631, 0.554035, 0.62355225, 0.85751908, 1.72211189, 7.72396924, 36.2255344],
        "Google Colab (baseline)":
            [1.53179673, 1.53811932, 1.61198598, 2.45895291, 7.93281146, 31.82661883, 150.10271399],
        "Google Colab (torch accelerated)":
            [1.8893, 1.4385, 1.4069, 1.4556, 2.1378, 3.3768, 8.5547],
    }
    
    # plt.plot(grid_sizes, wtimes, label=f"{f.__name__} (m1 (16') macbook pro, 2021)", marker='o')
    for label, wtime in wtimes.items():
        plt.plot(grid_sizes, wtime, label=f"{label}", marker='o')
        for i,j in zip(grid_sizes, wtime):
            plt.annotate("%.3f s" % j, xy=(i,j), xytext=(5,-10), textcoords="offset points")

    plt.xscale("log", base=2)
    plt.xlabel("N (gridsize: 2**N x 2**N)")
    
    plt.yscale("log", base=10)
    plt.ylabel("wtime (in s)")
    plt.legend()

    plt.show()
