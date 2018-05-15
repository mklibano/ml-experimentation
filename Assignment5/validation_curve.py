import numpy as np
from Assignment5.train_linear_reg import train_linear_reg
from Assignment5.cost_function_reg import cost_function_reg


def validation_curve(X,y,Xval,yval):

    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape((-1, 1))
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):

        # Compute optimal model parameters
        theta = train_linear_reg(X, y,lambda_vec[i])

        # Calculate error for training and cross validation set
        error_train[i] = cost_function_reg(theta,X,y, 0)
        error_val[i] = cost_function_reg(theta, Xval, yval, 0)

    return lambda_vec, error_train, error_val