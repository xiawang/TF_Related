import matplotlib.pyplot as plt
import numpy as np

class Data:
    def __init__(self):
        self.X = None
        self.Y = None

    def random_normal(self, size):
        n_data = np.ones((size, 2))
        x0 = np.random.normal(2*n_data, 1)
        y0 = np.zeros(size)
        x1 = np.random.normal(-2*n_data, 1)
        y1 = np.ones(size)
        self.X = np.vstack((x0, x1))
        self.Y = np.hstack((y0, y1))