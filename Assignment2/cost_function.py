import numpy as np
from Assignment2.sigmoid import sigmoid


def cost_function(theta, X, y):

    """
    Computes the cost  for a logistic regression algorithm
    :param X: The feature matrix
    :param y: The output vector
    :param theta: the parameter vector
    """

    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))




