import numpy as np


def feature_normalize(X):

    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    normalized_X = np.divide(X - mu,sigma)

    return (normalized_X, mu, sigma)