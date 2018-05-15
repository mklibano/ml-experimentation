import numpy as np
from Assignment4.feed_forward_prop import feed_forward_prop


def accuracy(X, y, nn_params, input_layer_size, hidden_layer_size, num_labels):

    # convert the weights(parameter vector) back to matrix format for each layer
    theta1 = np.matrix(np.reshape(nn_params[0:(input_layer_size + 1) * hidden_layer_size],
                                  (hidden_layer_size, input_layer_size + 1)))

    theta2 = np.matrix(np.reshape(nn_params[(input_layer_size + 1) * hidden_layer_size:],
                                  (num_labels, hidden_layer_size + 1)))

    # Classify each instance in the test set using forward propagation
    z2, a2, z3, prediction = feed_forward_prop(X, theta1, theta2)

    # select the index with the highest probability for each training instance
    pred_max = np.argmax(prediction, axis=0)

    # For true label prediction, add one due to zero-indexing
    pred_max = pred_max + 1

    # calculate classifier accuracy
    comparison_vector = np.equal(pred_max.T, y)
    correct = np.sum(comparison_vector == True)
    acc = correct / len(y)

    return acc
