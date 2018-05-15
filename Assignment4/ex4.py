import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Assignment4.backprop import backprop
from Assignment4.nn_cost_function import nn_cost_function
from Assignment4.rand_initialize_weights import rand_initialize_weights
from scipy.optimize import minimize
from Assignment4.accuracy import accuracy

"""load and visualize the matlab data"""
# load data file
path = os.getcwd() + '/data/ex4data1.mat'
data = loadmat(path)

# load weights file
path = os.getcwd() + '/data/ex4weights.mat'
weights = loadmat(path)

# load data into feature matrix X and output vector y
y = data['y']
X = data['X']

# load weights into parameters matrices
theta1, theta2 = weights['Theta1'], weights['Theta2']

# Add constant intercept term to the feature matrix
X = np.c_[np.ones((X.shape[0], 1)), X]

# randomly select 20 data point to display
rand = X[np.random.choice(X.shape[0], 20),1:]
temp = rand.flatten()
temp = temp.reshape(400, 20)
plt.imshow(temp.T)


"""compute the cost of the model with initial parameters"""
# set parameters we will use for this exercise
reg = 1
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

# unroll parameter matrices into a single vector
nn_params = np.r_[theta1.ravel(),theta2.ravel()]

# compute the cost function with initial theta parameter vector
Cost = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg)

print('Cost (with parameters loaded from w4weights: \n ', Cost)
print('Sanity check for shapes of X, y, theta1, theta2, \n', X.shape, y.shape, theta1.shape, theta2.shape)

"""randomly initialize weights before performing backprop"""

epsilon = 0.12
initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size, epsilon);
initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels, epsilon);

# Unroll parameters
initial_nn_params = np.r_[initial_theta1.ravel(),initial_theta2.ravel()]

cost, grad = backprop(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg)


print('Cost with randomly initialized weights \n ', cost)
print('Sanity check for unrolled gradient vector \n', grad.shape)


"""Minimize the cost function and compute theta parameters (weights)"""

# Calculate optimal NN weights by minimizing the cost function over 50 iterations
result = minimize(fun=backprop, x0=initial_nn_params, args=(input_layer_size, hidden_layer_size, num_labels, X, y, reg),
                  method='TNC', jac=True, options={'maxiter': 50})

# Calculate optimal NN weights by minimizing the cost function over 250 iterations
#result2 = minimize(fun=backprop, x0=initial_nn_params,
#                  args=(input_layer_size, hidden_layer_size, num_labels, X, y, reg),
#                  method='TNC', jac=True, options={'maxiter': 10})


accuracy = accuracy(X, y, result.x, input_layer_size, hidden_layer_size, num_labels)

#accuracy2 = accuracy(X, y, result.x, input_layer_size, hidden_layer_size, num_labels)

print('Classifier accuracy after 50 training iterations is {0}%'.format(accuracy*100))

# print('Classifier accuracy after 250 training iterations is {0}%'.format(accuracy2*100))

# randomly select 20 data point to display

# convert the weights(parameter vector) back to matrix format for each layer
theta1 = np.matrix(np.reshape(nn_params[0:(input_layer_size + 1) * hidden_layer_size],
                                  (hidden_layer_size, input_layer_size + 1)))

theta2 = np.matrix(np.reshape(nn_params[(input_layer_size + 1) * hidden_layer_size:],
                                  (num_labels, hidden_layer_size + 1)))

theta1 = theta1[:,1:]
theta2 = theta2[:,1:]

