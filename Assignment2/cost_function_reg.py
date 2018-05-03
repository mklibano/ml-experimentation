import numpy as np
from Assignment2.sigmoid import sigmoid


def cost_function_reg(theta, X, y, lam):

    """
    Computes the cost  for a logistic regression algorithm
    :param X: The feature matrix
    :param y: The output vector
    :param theta: the parameter vector
    :param lam: lambda regularization parameter
    """
    m = len(y)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    if X.shape[1] == 28:
        first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
        second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
        third = (lam / (2 * m)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
        return third + np.sum(first - second) / m
    else:
        return 0


def gradient_reg(theta, X, y, lam):

    """
    Computes gradient for a logistic regression algorithm
    :param X: The feature matrix
    :param y: The output vector
    :param theta: the parameter vector
    :param lam: lambda regularization parameter
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term)/m
        else:
            grad[i] = np.sum(term)/m + lam * theta[0,i] / m

    return grad
