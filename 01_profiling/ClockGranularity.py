#!/usr/bin/env python3

import numpy as np

import time
from timeit import default_timer as timer


def checktick(timefn):
   M = 10000
   timesfound = np.empty((M,))
   for i in range(M):
      t1 =  timefn() # get timestamp from timer
      t2 = timefn() # get timestamp from timer
      while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
          t2 = timefn() # get timestamp from timer
      t1 = t2 # this is outside the loop
      timesfound[i] = t1 # record the time stamp
   minDelta = 1000000
   Delta = np.diff(timesfound) # it should be cast to int only when needed
   minDelta = Delta.min()
   return minDelta


def main():
    print(f'time.time: {checktick(time.time)}')
    print(f'time.time_ns: {checktick(time.time_ns) * 1.0e-9}')
    print(f'timeit.default_timer: {checktick(timer)}')


if __name__ == '__main__':
    main()
