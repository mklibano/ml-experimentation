import numpy as np
from scipy.optimize import minimize
from Assignment3.lr_cost_function import lr_cost_function, gradient_reg


def one_vs_all(X, y, n_labels, reg):

    X = np.matrix(X)
    y = np.matrix(y)

    m = X.shape[0]
    n = X.shape[1]

    # initialize k x (n+1) array  for the parameters of each of the k classifiers
    theta = np.zeros((n_labels, n))
    initial_theta = np.zeros(n)

    # iterate through k classifiers, training each individually
    for i in range(1, n_labels+1):

        labels = 1*(y == i)

        # minimize the cost function
        result = minimize(fun=lr_cost_function, x0=initial_theta,
                          args=(X, labels, reg), method=None, jac=gradient_reg, options={'maxiter': 3000})

        cost = lr_cost_function(result.x, X, y, reg)
        print('Cost: \n', cost)

        theta[i-1] = result.x

    return theta