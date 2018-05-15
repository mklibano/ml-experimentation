import numpy as np
from Assignment4.sigmoid import sigmoid


def sigmoid_gradient(z):

    return np.multiply(sigmoid(z), (1-sigmoid(z)))
