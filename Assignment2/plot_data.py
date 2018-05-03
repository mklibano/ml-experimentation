import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y, ax=None):

    positive = np.where(y > 0)[0]
    np.reshape(positive, (len(positive), 1))
    pos1 = np.squeeze(np.asarray(X[positive, 0]))
    pos2 = np.squeeze(np.asarray(X[positive, 1]))

    negative = np.where(y < 1)[0]
    np.reshape(negative, (len(negative), 1))
    neg1 = np.squeeze(np.asarray(X[negative, 0]))
    neg2 = np.squeeze(np.asarray(X[negative, 1]))

    if ax is None:
        ax = plt.gca()
    ax.scatter(pos1, pos2, s=50, c='b', marker='o', label='Admitted')
    ax.scatter(neg1, neg2, s=50, c='r', marker='x', label='Not Admitted')
    ax.legend(frameon=True, fancybox=True);
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')