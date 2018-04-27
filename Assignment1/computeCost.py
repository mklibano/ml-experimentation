import numpy as np


def compute_cost(X, y, theta):

    h = X*theta.T
    s = np.sum(np.power(h-y, 2))
    return s/(2*len(X))

