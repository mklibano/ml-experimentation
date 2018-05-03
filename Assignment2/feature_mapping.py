import numpy as np


def map_feature(x1, x2):

    degree = 6

    out = np.ones(x1[:,0].shape)

    for i in range(1, degree+1):
        for j in range(0,i+1):
            temp = np.multiply(np.power(x1, i-j),np.power(x2, j))
            out = np.append(out, temp, axis=1)

    return out
