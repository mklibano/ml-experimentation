import numpy as np
from Assignment4.sigmoid import sigmoid
from sklearn.preprocessing import OneHotEncoder
from Assignment4.feed_forward_prop import feed_forward_prop


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg):

    # cast arrays to matrices
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # One hot encode the output labels
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)

    # convert the parameter vector back to matrix format for each layer
    theta1 = np.matrix(np.reshape(nn_params[0:(input_layer_size+1)*hidden_layer_size],(hidden_layer_size,input_layer_size+1)))
    theta2 = np.matrix(np.reshape(nn_params[(input_layer_size+1)*hidden_layer_size:],(num_labels, hidden_layer_size+1)))

    # feed-forward propagation for all training instances
    z2, a2, z3, h = feed_forward_prop(X, theta1, theta2)

    # compute the cost for each training instance
    Cost = 0
    for i in range(m):
        first = np.multiply(-y_onehot[i,:],np.log(h.T[i,:]))
        second = np.multiply((1-y_onehot[i,:]),np.log(1-h.T[i,:]))
        Cost += np.sum(first-second)

    regularization = np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2))

    Cost = Cost/m + (reg/(2*m))*regularization



    return Cost

