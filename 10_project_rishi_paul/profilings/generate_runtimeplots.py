import matplotlib.pyplot as plt
import numpy as np

if __name__== "__main__":
    grid_sizes = [2**i for i in range(5, 13)]
    N = 3
    tEnd = 0.1
    # wtimes = {
    #     "Apple MacBook M1 Air, 2020 (baseline)":
    #         [0.06273317, 0.07764285, 0.13479832, 0.36281343, 1.38195608, 6.00546406, 29.00729133],
    #     "Apple MacBook M1 Air (Cython)":
    #         [0.04825463, 0.05819958, 0.09835089, 0.24178517, 0.89203914, 3.72730593, 17.08468867],
    #     "Apple MacBook M1 Air (Dask)":
    #         [0.48807631, 0.554035, 0.62355225, 0.85751908, 1.72211189, 7.72396924, 36.2255344],
    #     "Google Colab (baseline)":
    #         [1.53179673, 1.53811932, 1.61198598, 2.45895291, 7.93281146, 31.82661883, 150.10271399],
    #     "Google Colab (torch accelerated)":
    #         [1.8893, 1.4385, 1.4069, 1.4556, 2.1378, 3.3768, 8.5547],
    # }
    
    # TIMES TAKEN ON MACBOOK PRO
    wtimes = {
        "default": [8.71669860e-02, 1.48659958e-01, 4.15250028e-01, 1.53531331e+00, 6.66445868e+00, 2.99840715e+01, 2.01926428e+02, 8.18251801e+02],
        "dask": [0.58260736,   0.69863889,   0.95681943,   1.88058876,   6.57225839, 23.48419025,  122.6945265,  543.14461222],
        "cython": [6.61961530e-02, 1.04462014e-01, 2.56057278e-01, 1.02005108e+00, 3.92652232e+00, 1.68934285e+01, 1.29175774e+02, 5.09121211e+02],
        "cython and dask": [0.49529989,   0.57565614,   0.80512675,   1.403971,     5.03933383, 16.16864372,  77.65571207, 313.08173706],
        "macbook air w/ gpu (torch)": [2.4943, 2.6036, 2.6901, 2.6627, 3.2372, 16.0080, 66.9563, 262.0508],
        "gcolab w/ gpu (torch)": [1.4385, 1.4069, 1.4556, 2.1378, 3.3768, 8.5547, 33.9086, 131.9917],
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
