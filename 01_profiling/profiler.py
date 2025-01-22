# import time
#
# import psutil
# import tabulate
# from threading import Thread
#
# def cpu_profiler(interval=1):
#
#     def decorator(func):
#         measurements = dict()
#
#         running = False
#
#         def formatter(results: dict):
#             headers = ["Time"] + [f"CPU {i}" for i in range(len(next(iter(measurements.values()))))]
#             rows = []
#             for timestamp, core_values in results.items():
#                 readable_time = time.strftime('%H:%M:%S', time.localtime(timestamp))
#                 rows.append([readable_time] + list(core_values))
#
#             print(tabulate.tabulate(rows, headers=headers, tablefmt='grid'))
#
#         def threader():
#             while running:
#                 timestamp = time.time()
#                 res = psutil.cpu_percent(interval, percpu=True)
#
#                 measurements[timestamp] = res
#                 # time.sleep(interval)
#
#         def wrapper(*args, **kwargs):
#             nonlocal running
#             running = True
#
#             thread = Thread(target=threader)
#             thread.start()
#
#             func(*args, **kwargs)
#
#             running = False
#             thread.join()
#
#             formatter(measurements)
#
#         return wrapper
#     return decorator
#
# @cpu_profiler(interval=1)
# def julia():
#     for x in range(5):
#         time.sleep(1)
#
#
# if __name__ == "__main__":
#     julia()