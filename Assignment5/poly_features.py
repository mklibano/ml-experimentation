import numpy as np


def poly_features(X, p):

    X_poly = np.zeros((len(X), p))
    X = np.array(X.T)[0]
    for i in range(1, p + 1):
        X_poly[:, i - 1] = X ** i

    return X_poly
