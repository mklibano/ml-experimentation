import numpy as np

def sigmoid(z):
    z = np.float128(z)
    return 1 / (1+np.exp(-z))

