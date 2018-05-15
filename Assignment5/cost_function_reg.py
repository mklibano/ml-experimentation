import numpy as np


def cost_function_reg(theta, X,y,reg):

    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = len(y)

    error = np.sum(np.power(X*theta.T - y, 2))

    reg = reg*(theta * theta.T)

    return error/(2*m) + reg/(2*m)


def gradient_reg(theta, X, y, reg):

    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = len(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = X*theta.T-y

    gradient = (error.T*X)/m + (reg/m)*theta

    gradient[0] = gradient[0] - (reg/m)*theta

    np.array(gradient.T).flatten()

    return np.array(gradient.T).flatten()
