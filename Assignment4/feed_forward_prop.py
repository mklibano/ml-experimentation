import numpy as np
from Assignment4.sigmoid import sigmoid


def feed_forward_prop(X, theta1, theta2):

    X = np.matrix(X)
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)

    # Feed-forward propogation for the hidden layer
    z2 = theta1 * X.T
    a2 = sigmoid(z2)

    # Feed-forward propagation for the output layer
    a2 = np.c_[np.ones((a2.shape[1], 1)), a2.T]
    z3 = theta2 * a2.T

    # prediction vector for each training instance
    h = sigmoid(z3)

    return z2, a2, z3, h