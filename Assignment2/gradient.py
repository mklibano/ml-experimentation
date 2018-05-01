import numpy as np
from Assignment2.sigmoid import sigmoid


def gradient(theta, X, y):

    """
    Computes gradient for a logistic regression algorithm
    :param X: The feature matrix
    :param y: The output vector
    :param theta: the parameter vector
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad