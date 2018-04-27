import numpy as np
from computeCost import compute_cost
import matplotlib.pyplot as plt


def gradient_descent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    m = len(y)
    cost = np.zeros(iterations)


    for i in range(iterations):
        error = X*theta.T - y

        for j in range(parameters):
            der = np.multiply(error, X[:, j])
            temp[0,j] = theta[0,j] - ((alpha/m)*np.sum(der))
        theta = temp
        cost[i] = compute_cost(X, y, theta)



    return theta, cost



