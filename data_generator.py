import matplotlib.pyplot as plt
import numpy as np
import random

class Data:
    def __init__(self):
        self.X = None
        self.Y = None
        self.dist_batches = None
        self.bin_labels = None

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

    def parabola_distribution(self, batch, simulate_points):
        """ Generate batches of points that are from a parabola. Currently, the
        parabola parameters are fixed.

        Parameters
        ----------
        batch: number of batches of points
        simulate_points: generated batches of points from a line space

        Returns
        -------
        N/A
        """
        a = np.random.uniform(1, 2, size=batch)[:, np.newaxis]
        parabolas = a * np.power(simulate_points, 2) + (a-1)
        bin_labels = (a - 1) > 0.5
        bin_labels = bin_labels.astype(np.float32)
        self.dist_batches = parabolas
        self.bin_labels = bin_labels


class SequenceData:
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            rand_len = random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(rand_len)
            if random.random() < .5:
                # linear sequence
                rand_start = random.randint(0, max_value - rand_len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + rand_len)]
                s += [[0.] for i in range(max_seq_len - rand_len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(rand_len)]
                s += [[0.] for i in range(max_seq_len - rand_len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.

        Parameters
        ----------
        batch_size: length of a batch of data

        Returns
        -------
        batch_data: data in the next batch
        batch_labels: labels in the next batch
        batch_seqlen: sequence length in the next batch
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

