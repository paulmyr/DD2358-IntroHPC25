import psutil
from collections import defaultdict
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import time
import threading
from tabulate import tabulate
import numpy as np

class CPUProfiler:
    """
    Can be used to determine the CPU usage percentage per core.

    Usage: 
    # Values for being used to instantiate profiler are dummy values.
    profiler = CPUProfiler(experiment_name="Test", granularity=5, interval=1)
    ....other code...

    profiler.initiate_observation()
    ...code to profile...
    profiler.end_observation()

    ...other code...
    # Generates a graph showing evolution of usage over elapsed time
    profiler.generate_usage_graph()
    # Prints fine-grained and coarse-grained usage info in a pretty tabular format
    profiler.generate_tabular_summary()
    """

    def __init__(self, experiment_name = "Experiment", granularity=0.1, interval=0.05):
        # Condition: Granularity needs to be bigger than interval.
        # See https://stackoverflow.com/questions/8600161/executing-periodic-actions/18180189#comment68938773_18180189 for reasoning.
        if granularity < interval:
            raise Exception("Granularity should be at least as big as interval")
        # "granularity" defines the increments (in seconds) at which the cpu usage %age per core is to be measured.
        # Eg: if granularity is 5, then cpu usage is measured at 0s, 5s, 10s, ... until the experiment is ended
        self.granularity = granularity
        # This defines the value of the interval argument to cpu_percent. 
        # Read more about that value here: https://psutil.readthedocs.io/en/latest/index.html#psutil.cpu_percent
        self.interval = interval
        # The title of the experiment: used in graphing and tabular generation
        self.experiment_name = experiment_name
        # Used to accumulate output of psutil.cpu_percent to get usage % of each core at specified time
        self.usage_info = defaultdict(list)
        # Cache used for graphing information. Consists of "n" lists (n = number of cores), where each list 
        # has m percentage recordings (where m is the number of recordings made)
        self.corewise_percentage_cache = []
        # Flags to help determine when experiment has started/ended
        self.experiment_running = self.experiment_ended = False
        # Used to count increments of time in granularity required.
        self.time_elapsed = 0

    def initiate_observation(self):
        """
        To be called **just before** the code we are interested in profiling begins.

        Precondition: The experiment MUST NOT be running 
        """
        if self.experiment_running:
            raise Exception("The experiment is already running!")
        
        # Code adapted from this SO answer: https://stackoverflow.com/a/18180189
        def record_cpu_usage():
            next_call = timer()
            while not self.experiment_ended:
                if self.time_elapsed == 0:
                    # we already have measurements for when time elapsed is 0
                    self.time_elapsed += self.granularity
                else:
                    curr_percent_usage = psutil.cpu_percent(interval=self.interval, percpu=True)
                    self.usage_info[self.time_elapsed] = curr_percent_usage
                    self.time_elapsed += self.granularity
                # This helps get rid of "drift". See SO answer above for more.
                next_call = next_call + self.granularity
                time.sleep(next_call - timer())
            
            # Once we are out of this loop, the experiment has ended. So, set the flag appropriately
            self.experiment_running = False
        
        timerThread = threading.Thread(target=record_cpu_usage)
        self.experiment_running = True
        timerThread.daemon = True
        # Make a baseline measurement of cpu usage
        curr_percent_usage = psutil.cpu_percent(interval=self.interval, percpu=True)
        self.usage_info[self.time_elapsed] = curr_percent_usage
        # start the thread
        timerThread.start()


    def end_observation(self):
        """
        To be called **just after** the code we are interested in profiling ends.

        Precondition: The experiment MUST be running
        """
        if not self.experiment_running:
            raise Exception("The experiment is not running, there is nothing to end!")
        self.experiment_ended = True
        self.experiment_running = False

    def _compute_corewise_percentage_cache(self):
        """
        NOTE: THIS IS AN INTERNAL METHOD. NOT TO BE USABLE OUTSIDE THE CLASS.

        Computes an array of n arrays (where n is the number of cores). Each of these arrays in turn has 
        m entries (where m is the number of times cpu usage percentage was observed during the experiment).
        This information can then be used to generate plots.
        """
        time_ticks = sorted(self.usage_info.keys())
        # We are guaranteed to have at least the base line usage (at time-tick 0). Use that to determine
        # number of caches and initialize the usage info cache
        self.corewise_percentage_cache = [[] for _ in range(len(self.usage_info[0]))]

        for curr_tick in time_ticks:
            curr_tick_usages = self.usage_info[curr_tick]
            for idx, curr_usage in enumerate(curr_tick_usages):
                self.corewise_percentage_cache[idx].append(curr_usage)


    def generate_usage_graph(self):
        """
        Generates the graph of the observed per-core CPU usage percentages. The number of observations
        depends on the provided "granularity" and the duration for which the code to be profiled executed. 
        The number of lines in the graph depends on the number of cores we are observing.

        Precondition: The experiment MUST NOT be running and the experiment MUST HAVE ended.
        """
        if self.experiment_running:
            raise Exception("Cannot generate grpah when experiment is running.")
        if not self.experiment_ended:
            raise Exception("Cannot generate graph when experiment has not ended.")

        if len(self.corewise_percentage_cache) == 0:
            self._compute_corewise_percentage_cache()
        
        time_ticks = sorted(self.usage_info.keys())
        
        for idx in range(len(self.corewise_percentage_cache)):
            plt.plot(time_ticks, self.corewise_percentage_cache[idx], label = f"Core #{idx}")
        
        plt.xlabel("Experiment Elapsed Time (sec)")
        plt.ylabel("Usage Percentage")
        plt.title(f"CPU Usage Percentage (per core) When Running {self.experiment_name}")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()



    def generate_tabular_summary(self):
        """
        Generates a tabular summary of the observed per-core CPU usage percentages. There are n+1 columns and m rows in the 
        tabular report: where "n" is the number of cores and "m" are the number of observations (m depends on the provided 
        "granularity" during instantiation and the duration for which the process runs). 

        Precondition: The experiment MUST NOT be running and the experiment MUST HAVE ended.
        """
        if self.experiment_running:
            raise Exception("Cannot generate grpah when experiment is running.")
        if not self.experiment_ended:
            raise Exception("Cannot generate graph when experiment has not ended.")
        
        # Get data in format that can be used by the tabulate library
        ticks_with_per_core_usage = []
        time_ticks = sorted(self.usage_info.keys())

        for curr_tick in time_ticks:
            ticks_with_per_core_usage.append([curr_tick] + self.usage_info[curr_tick])
        
        fine_grained_headers = ["Elapsed Time (sec)"]
        # There will always be at least one observation of usage times: the one taken at 0sec
        num_cores = len(self.usage_info[0])
        for idx in range(num_cores):
            fine_grained_headers.append(f"Core #{idx}")
        
        # Prints the fine grained statistics (usages of the cores at different points in terms of elapsed time)
        fancy_tabular_fine_grained = tabulate(ticks_with_per_core_usage, headers=fine_grained_headers, tablefmt="fancy_grid")
        print("\n\n =============== CPU Profiler Fine-Grained and Coarse-Grained Data =============== \n\n")
        print(" "*10 + f"Fine-Grained Information of Usage Percentages (per core) at Elapsed Times When Running {self.experiment_name}")
        print(fancy_tabular_fine_grained)
        print("\n\n\n")
        # Getting Averages and Standard deviation for usage percentages per core
        if len(self.corewise_percentage_cache) == 0:
            self._compute_corewise_percentage_cache()
        
        summary_stats_data = []
        for idx in range(num_cores):
            curr_core_usage_percent = self.corewise_percentage_cache[idx]
            summary_stats_data.append([f"Core #{idx}", np.mean(curr_core_usage_percent), np.std(curr_core_usage_percent, dtype=np.float64)])
        coarse_grained_headers = ["Core Number", "Mean Usage Percentage", "Std. Dev. in Usage Percentage"]
        fancy_tabular_coarse_grained = tabulate(summary_stats_data, headers=coarse_grained_headers, tablefmt="fancy_grid")
        print(f"Coarse Grained Info on Usage Percentages (per core) While Running {self.experiment_name}")
        print(fancy_tabular_coarse_grained)
        


