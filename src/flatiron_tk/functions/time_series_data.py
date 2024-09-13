import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

class TimeSeriesData():

    def __init__(self, data_file, delimiter=',', skip_header=0):

        """
        linear interpolator for time series data from a delimited file (defaults to csv without header)
        the data file first column should be the time array, and the other columns are the data

        e.g. for 3D time dependent velocity 
          0,   1,   3,  5
          0.1, 2,   4,  6
          ...

        where 
        t  = [0, 0.1, ...]
        ux = [1, 2, ...]
        uy = [3, 4, ...]
        uz = [5, 6, ...]
        """

        self.file = data_file
        self.data = np.genfromtxt(self.file, delimiter=delimiter, skip_header=skip_header)

    def query(self, t):
        """
        Perform linear interpolation at time t
        """
        return np.array( [ np.interp(t, x[:,0], x[:,i]) for i in range(1, x.shape[1]) ] )



