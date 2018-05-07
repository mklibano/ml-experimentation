import numpy as np
from Assignment3.sigmoid import sigmoid


def predict_one_vs_all(X, theta, y):

    X = np.matrix(X)
    theta = np.matrix(theta)

    # probability of each training instance belonging to each of 10 classes
    h = sigmoid(X * theta.T)

    # select the index with the highest probability for each training instance
    h_max = np.argmax(h, axis=1)

    # For true label prediction, add one due to zero-indexing
    h_max = h_max + 1

    # calculate classifier accuracy
    comparison_vector = np.equal(h_max, y)
    correct = np.sum(comparison_vector == True)
    accuracy = correct / len(y)

    return h_max, accuracy