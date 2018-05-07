import numpy as np
from Assignment2.sigmoid import sigmoid


def lr_cost_function(theta, X, y, lam):

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

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    third = (lam / (2 * m)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / m + third


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

    error = sigmoid(X * theta.T) - y
    grad = (X.T * error) / m
    grad_reg =grad.T + (theta*lam/m)

    # The theta zero parameter is not regularized
    grad_reg[0,0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    return np.array(grad).ravel()
