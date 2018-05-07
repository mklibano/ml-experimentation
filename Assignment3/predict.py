import numpy as np
from Assignment3.sigmoid import sigmoid

def predict(X, theta1, theta2):

    X = np.matrix(X)
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)

    m = X.shape[0]

    # initialize prediction vector
    p = np.zeros((m, 1))

    # Feed-forward propogation for the hidden layer
    z2 = theta1 * X.T
    a2 = 1 / (1+np.exp(-z2))

    # Feed-forward propagation for the output layer
    a2 = np.c_[np.ones((a2.shape[1], 1)), a2.T]
    z3 = theta2 * a2.T

    # prediction vector for each training instance
    h = 1 / (1 + np.exp(-z3))

    return h



