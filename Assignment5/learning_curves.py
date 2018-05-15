import numpy as np
from Assignment5.train_linear_reg import train_linear_reg
from Assignment5.cost_function_reg import cost_function_reg


def learning_curve(X,y, Xval, yval, reg):

    # This function generates the train and cross validation
    # set errors needed to plot the learning curve
    m = len(y)
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        theta = train_linear_reg(X[0:i + 1, :], y[0:i + 1], reg)
        error_train[i] = cost_function_reg(theta, X[0:i + 1, :], y[0:i + 1], reg)
        error_val[i] = cost_function_reg(theta, Xval, yval, reg)

    return error_train, error_val
