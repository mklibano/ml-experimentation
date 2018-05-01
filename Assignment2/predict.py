import numpy as np
from Assignment2.sigmoid import sigmoid


def predict(X, theta):
    z=X*theta.T
    y = sigmoid(z)

    pos = np.where(y > 0.5)
    neg = np.where(y < 0.5)

    y[pos[0],:] = 1
    y[neg[0],:] = 0

    return y
