import matplotlib.pyplot as plt
import numpy as np

# KEYS IN THE DICTIONARY FOR RUNNING TIMES
DEFAULT_KEY = "default (baseline)"
DASK_KEY = "dask"
CYTHON_KEY = "cython"
CYTHON_DASK_KEY = "cython + dask"
TORCH_MPS_KEY = "torch (gpu: mps)"
TORCH_COLAB_KEY = "torch (gcolab w/ gpu)"

ALL_KEYS = [DEFAULT_KEY, CYTHON_KEY, TORCH_COLAB_KEY, TORCH_MPS_KEY, DASK_KEY, CYTHON_DASK_KEY]

# Running times for MacBook Pro M1 (16' 2021). Used for plotting
WTIMES_PRO = {
    DEFAULT_KEY : [8.39499580e-02, 1.48597334e-01, 4.13450195e-01, 1.54879474e+00, 6.80613208e+00, 2.91244473e+01, 2.07594952e+02, 8.25890252e+02],
    DASK_KEY: [0.56476908,   0.68364637,   0.95509778,   1.8998761,    6.44783811, 23.43153089, 123.50528556, 483.67777446],
    CYTHON_KEY: [6.51034030e-02, 1.05412833e-01, 2.57549167e-01, 1.05769446e+00, 4.02322717e+00, 1.72108557e+01, 1.29203993e+02, 5.07835946e+02],
    CYTHON_DASK_KEY: [0.53706646,   0.56078451,   0.79523136,   1.39823129,   4.72460328, 15.88043772,  77.46793522, 296.93671167],
    TORCH_MPS_KEY: [2.34905328, 2.42914114,  2.41929667,  2.76244899,  2.5385459,   5.39078251, 25.66523169, 92.87822497],
    TORCH_COLAB_KEY: [1.4385, 1.4069, 1.4556, 2.1378, 3.3768, 8.5547, 33.9086, 131.9917],
}


# To be filled with running times for MacBook Air (if needed)
WTIMES_AIR = {

}

def line_plot_given_runtimes(key_list, wtime_dict, grid_sizes, title):
    """
    Plots the runtimes from the provided dictionary for the keys given in the first argument
    """
    wtimes = {key: wtime_dict.get(key) for key in key_list}

    for label, wtime in wtimes.items():
        plt.plot(grid_sizes, wtime, label=f"{label}", marker='o')
        # for i,j in zip(grid_sizes, wtime):
        #     plt.annotate("%.3f s" % j, xy=(i,j), xytext=(5,-10), textcoords="offset points")

    plt.xscale("log", base=2)
    plt.xlabel("N (gridsize: 2**N x 2**N)")
    
    plt.yscale("log", base=10)
    plt.ylabel("wtime (in s)")
    plt.legend()
    plt.title(title)

    plt.show()

def bar_plot_last_n_runtimes(key_list, wtime_dict, last_n, title, required_sizes = None):
    """
    Creates a bar plot of the last n runtimes (depending on the argument "last_n") for the runtimes keys provided 
    in keys_list. For eg, if we want to plot the last 2 runtimes for cython and baseline as a bar_plot, the way to 
    call this function would be:

    bar_plot_last_n_runtimes([DEFAULT_KEY, CYTHON_KEY], WTIMES_PRO, 2, "Bar Plot of Last 2 Runtimes")
    """
    wtimes = {}
    for key, value in wtime_dict.items():
        if key in key_list:
            wtimes[key] = value[len(value)-last_n:]

    grid_sizes = required_sizes
    if not grid_sizes:
        last_size = 2**12
        grid_sizes = []

        for _ in range(last_n):
            grid_sizes.append(last_size)
            last_size = last_size // 2
        
    keys = list(wtimes.keys())
    values = list(wtimes.values())
    num_bars = len(values[0])

    x_pos = np.arange(len(keys))

    bar_width = 0.2

    _, ax = plt.subplots()

    for i in range(num_bars):
        bars = ax.bar(x_pos + i * bar_width, [v[i] for v in values], bar_width, label=f"{grid_sizes[len(grid_sizes) - 1 - i]} x {grid_sizes[len(grid_sizes) - 1 - i]}")

        for bar in bars:
            height = bar.get_height()  # Get the height of the bar (value)
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{str(round(height,1))}s", ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x_pos + (num_bars - 1) * bar_width / 2)
    ax.set_xticklabels(keys)

    ax.set_xlabel("Implementation")
    ax.set_ylabel("wtimes (seconds)")
    ax.set_title(title)

    ax.legend()

    plt.tight_layout()
    plt.show()




        

if __name__== "__main__":
    # UNCOMMENT THIS TO PLOT LINE-PLOTS
    # grid_sizes = [2**i for i in range(5, 13)]
    # line_plot_given_runtimes([DEFAULT_KEY, DASK_KEY, CYTHON_DASK_KEY, CYTHON_KEY], WTIMES_PRO, grid_sizes, "Cython vs Cython + Dask")

    # UNCOMMENT THIS TO PLOT BAR-GRAPHS (ZOOMED IN VERSION TO SHOW HOW MUCH ADVANTAGE WE HAVE)
    bar_plot_last_n_runtimes(ALL_KEYS, WTIMES_PRO, 2, "Comparison for Bigger Grid Sizes")
    