import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(theta, X, y):

    positive = np.where(y > 0)[0]
    np.reshape(positive, (len(positive), 1))
    pos1 = np.squeeze(np.asarray(X[positive, 1]))
    pos2 = np.squeeze(np.asarray(X[positive, 2]))

    negative = np.where(y < 1)[0]
    np.reshape(negative, (len(negative), 1))
    neg1 = np.squeeze(np.asarray(X[negative, 1]))
    neg2 = np.squeeze(np.asarray(X[negative, 2]))

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.scatter(pos1, pos2, s=50, c='b', marker='o', label='Admitted')
    ax2.scatter(neg1, neg2, s=50, c='r', marker='x', label='Not Admitted')
    ax2.legend()
    ax2.set_xlabel('Exam 1 Score')
    ax2.set_ylabel('Exam 2 Score')

    # Line that represents the decision boundary
    plot_x = np.arange(np.min(X[:, 1]), 100, 1)
    plot_y = (-1. / theta[2])*(theta[1]*plot_x + theta[0])
    ax2.plot(plot_x, plot_y)
    plt.show()

