import numpy as np
from computeCost import compute_cost
import matplotlib.pyplot as plt
import time


def gradient_descent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    m = len(y)
    cost = np.zeros(iterations)

    plt.ion()
    fig, ax = plt.subplots()

    xdata, ydata = [], []
    plot = ax.scatter(0, compute_cost(X, y, theta))
    plt.xlim(0, iterations)
    plt.ylim(0, compute_cost(X, y, theta))
    plt.draw()

    for i in range(iterations):
        xdata.append(i)
        ydata.append(cost[i])
        error = X*theta.T - y

        for j in range(parameters):
            der = np.multiply(error, X[:, j])
            temp[0,j] = theta[0,j] - ((alpha/m)*np.sum(der))
        theta = temp
        cost[i] = compute_cost(X, y, theta)

        xdata.append(i)
        ydata.append(cost[i])
        plot.set_offsets(np.c_[xdata, ydata])
        fig.canvas.draw_idle()
        plt.pause(0.1)

    plt.show()
    return theta, cost



