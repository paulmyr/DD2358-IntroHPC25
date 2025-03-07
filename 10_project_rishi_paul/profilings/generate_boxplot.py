import json
import matplotlib.pyplot as plt
import numpy as np

def create_boxplot_from_json(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    
    labels = list(data.keys())
    runtimes = list(data.values())

    plt.boxplot(runtimes, labels=labels)
    plt.title("Boxplot of Runtimes for Finite Volume Implementation")
    plt.ylabel("Runtime (seconds)")
    plt.show()


create_boxplot_from_json("boxplot_runtimes.json")

