import numpy as np
from scipy.optimize import minimize
from Assignment5.cost_function_reg import cost_function_reg, gradient_reg


def train_linear_reg(X, y, reg):

    # initialize theta
    initial_theta = np.zeros((1, X.shape[1]))

    # find optimal theta parameters
    result = minimize(cost_function_reg, initial_theta, args=(X, y, reg),
                      jac=gradient_reg, options={'maxiter': 20000})

    return result.x



