import os
import numpy as np
import matplotlib.pyplot as plt
from Assignment3.lr_cost_function import lr_cost_function, gradient_reg
from Assignment3.predict_one_vs_all import predict_one_vs_all
from scipy.io import loadmat
from Assignment3.predict import predict

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

prediction = predict(X, theta1, theta2)

# select the index with the highest probability for each training instance
pred_max = np.argmax(prediction, axis=0)

# For true label prediction, add one due to zero-indexing
pred_max = pred_max + 1

# calculate classifier accuracy
comparison_vector = np.equal(pred_max.T, y)
correct = np.sum(comparison_vector == True)
accuracy = correct / len(y)

print('Classifier accuracy is {0}%'.format(accuracy*100))

