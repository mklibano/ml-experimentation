import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Assignment4.feed_forward_prop import feed_forward_prop
from Assignment4.sigmoid_gradient import sigmoid_gradient


def backprop(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg):

    """Feedforward propagation and cost function computation"""
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

    """Perform backpropagation"""
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    for t in range(m):
        a1_t = X[t,:]                   # (1, 401)
        z2_t = z2[:,t].T                # (1, 25)
        a2_t = a2[t,:]                  # (1, 26)
        h_t = h[:,t].T            # (1, 10)
        y_t = np.matrix(y_onehot[t,:].T)  # (1,10)

        # Compute error for the output layer
        d3_t = h_t - y_t                # (1, 10)

        # Compute error for the hidden layer
        z2_t = np.insert(z2_t,0,values=np.ones(1))
        d2_t = np.multiply((theta2.T*d3_t.T).T,sigmoid_gradient(z2_t))  # (1, 26)

        # Accumulate the gradient
        delta1 = delta1 + (d2_t[:,1:]).T*a1_t
        delta2 = delta2 + d3_t.T*a2_t

    delta1 = delta1/m
    delta2 = delta2/m

    # add lambda regularizaton
    delta1[:,1:] = delta1[:,1:] + (reg/m)*theta1[:,1:]
    delta2[:,1:] = delta2[:,1:] + (reg/m)*theta2[:,1:]

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))


    return Cost, grad









