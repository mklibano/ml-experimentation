import numpy as np


def gaussian_kernel(sigma, x1, x2):

    dist = np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

    return dist

