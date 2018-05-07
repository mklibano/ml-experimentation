import os
import numpy as np
import matplotlib.pyplot as plt
from Assignment3.lr_cost_function import lr_cost_function, gradient_reg
from Assignment3.predict_one_vs_all import predict_one_vs_all
from scipy.io import loadmat
from Assignment3.one_vs_all_classifier import one_vs_all

# load data file
path = os.getcwd() + '/data/ex3data1.mat'
data = loadmat(path)

# load weights file
path = os.getcwd() + '/data/ex3weights.mat'
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
plt.show()

# Test case for lrCostFunction
theta_test = np.matrix([-2, -1, 1, 2 ])
X_test = np.ones((5,1))
X_test = np.append(X_test, (np.arange(1,16).reshape((3,5)).T)/10, axis=1)
y_test = np.matrix('1;0;1;0;1')
lambda_test = 3

cost = lr_cost_function(theta_test, X_test, y_test, lambda_test)
grad = gradient_reg(theta_test, X_test, y_test, lambda_test)

print('Testing Cost: \n', cost)
print('Testing Gradients: \n', grad)

# training the classifier
n_labels = 10 # from 1 to 10, note that zero 0 is mapped to 10
reg = 0.1

theta = one_vs_all(X, y, n_labels, reg)

predictions, accuracy = predict_one_vs_all(X, theta, y)

print('Classifier accuracy is {0}%'.format(accuracy*100))




























