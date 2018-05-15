import numpy as np
from Assignment5.poly_features import poly_features
import matplotlib.pyplot as plt


def plot_fit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape((-1,1))
    X_poly = poly_features(x,p)
    X_poly = np.divide(X_poly - mu, sigma)
    X_poly = np.hstack((np.ones(len(X_poly)).reshape((-1,1)),X_poly))
    plt.plot(x,np.dot(X_poly,theta),'b--',linewidth=2)
    return