import psutil
from collections import defaultdict
from timeit import default_timer as timer
import time
import threading

class CPUProfiler:
    """
    Can be used to determine the CPU usage percentage per core.
    """

    def __init__(self, granularity=0.1, interval=0.05):
        # Condition: Granularity needs to be bigger than interval.
        # See https://stackoverflow.com/questions/8600161/executing-periodic-actions/18180189#comment68938773_18180189 for reasoning.
        if granularity < interval:
            raise Exception("Granularity should be at least as big as interval")
        self.granularity = granularity
        self.interval = interval
        self.usage_info = defaultdict(list)
        self.experiment_running = self.experiment_ended = False
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
                next_call = next_call + self.granularity
                # print(f"sleeping for {next_call - timer()}")
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
        # TODO: Should we be using a lock here? 
        self.experiment_ended = True
        self.experiment_running = False

    def generate_usage_graph(self):
        """
        Generates the graph of the observed per-core CPU usage percentages. The number of observations
        depends on the provided "granularity" and the duration for which the code to be profiled executed. 
        The number of lines in the graph depends on the number of cores we are observing.

        Precondition: The experiment MUST NOT be running and the experiment MUST HAVE ended.
        """
        pass


    def generate_tabular_summary(self):
        """
        Generates a tabular summary of the observed per-core CPU usage percentages. There are n+1 columns and m rows in the 
        tabular report: where "n" is the number of cores and "m" are the number of observations (m depends on the provided 
        "granularity" during instantiation and the duration for which the process runs). 

        Precondition: The experiment MUST NOT be running and the experiment MUST HAVE ended.
        """
        print(self.usage_info)