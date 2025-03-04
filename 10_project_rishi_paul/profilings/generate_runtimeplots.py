import matplotlib.pyplot as plt
import numpy as np

if __name__== "__main__":
    grid_sizes = [2**i for i in range(4, 11)]
    N = 3
    tEnd = 0.1
    wtimes = {
        "Apple MacBook M1 Air, 2020": [4.12682945e-06, 2.82851509e-05, 2.87202675e-04, 4.62396811e-03, 1.08565917e-01, 3.26654147e+00, 1.05220198e+02],
        "Google Colab no GPU":        [2.6181e-04, 7.7943e-04, 5.9743e-03, 5.0694e-02, 6.6716e-01, 1.5944e+01, 4.2580e+02],
        "Google Colab w/ GPU":        [7.2519e-04, 1.3657e-03, 5.9638e-03, 2.4870e-02, 1.4589e-01, 9.6252e-01, 2.2244e+01]
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
