import matplotlib.pyplot as plt
import numpy as np

class Data:
    def __init__(self):
        self.X = None
        self.Y = None

    def random_normal(self, size, distance, one_hot=False):
        """ Generate two groups of data points from two Gaussian distributions.
        One groupd of data points are labeled as 1s, and the other group labeled
        as 0s.

        Parameters
        ----------
        size: number of data points in each group
        distance: distance from the point (0,0), width and height
        one_hot: the usage of one hot encoding

        Returns
        -------
        N/A
        """
        n_data = np.ones((size, 2))
        x_0 = np.random.normal(distance*n_data, 1)
        if one_hot:
            y_0 = np.zeros(size*2)
            y_0 = y_0.reshape((size,2))
            y_0 = np.apply_along_axis(lambda x: [x[0]+1, x[1]], 1, y_0)
        else:
            y_0 = np.zeros(size)

        x_1 = np.random.normal(-distance*n_data, 1)
        if one_hot:
            y_1 = np.zeros(size*2)
            y_1 = y_1.reshape((size,2))
            y_1 = np.apply_along_axis(lambda x: [x[0], x[1]+1], 1, y_1)
        else:
            y_1 = np.ones(size)

        self.X = np.vstack((x_0, x_1))
        if one_hot:
            self.Y = np.vstack((y_0, y_1))
        else:
            self.Y = np.hstack((y_0, y_1))
