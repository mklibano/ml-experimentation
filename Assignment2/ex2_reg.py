import os
import numpy as np
from Assignment2.plot_data import plot_data
from Assignment2.feature_mapping import map_feature
from Assignment2.cost_function_reg import cost_function_reg, gradient_reg
import scipy.optimize as opt
from Assignment2.predict import predict
import matplotlib.pyplot as plt
from Assignment2.sigmoid import sigmoid
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

# load training data
path = os.getcwd() + '/data/ex2data2.txt'
data = np.loadtxt(path, delimiter=',')

# load data into feature matrix and output vector
X = np.matrix(data[:, 0:2])
y = np.matrix(data[:, 2:3])


# plotting the data
plot_data(X, y)


# feature mapping
feature_matrix = map_feature(X[:,0],X[:,1])


# Computing cost with initial parameters set to zero
lam = 1
initial_theta = np.zeros(feature_matrix[0,:].shape[1])
cost = cost_function_reg(initial_theta, feature_matrix, y, lam)
grad = gradient_reg(initial_theta, feature_matrix, y, lam)

print('Cost at initial theta (zeros): \n', cost)
print('Gradient at initial theta (zeros) - '
      'first five values only: \n', grad)

# Computing cost with initial parameters set to one
lam = 10
initial_theta = np.ones(feature_matrix[0,:].shape[1])
cost = cost_function_reg(initial_theta, feature_matrix, y, lam)
grad = gradient_reg(initial_theta, feature_matrix, y, lam)

print('Cost at test theta (with lambda = 10): \n', cost)
print('Gradient at initial theta (zeros) - '
      'first five values only: \n', grad)

# minimizing the cost function with lambda regularization
fig2, ax2 = plt.subplots(1,3, sharey = True, figsize=(17, 5))

poly = PolynomialFeatures(6)

# Decision boundaries
# Lambda = 0 : No regularization --> too flexible, overfitting the training data
# Lambda = 1 : Looks about right
# Lambda = 100 : Too much regularization --> high bias

for i, C in enumerate([0, 1, 100]):
    # Find parameters that minimize cost
    initial_theta = np.zeros(feature_matrix[0,:].shape[1])
    result = minimize(cost_function_reg, initial_theta, args=(feature_matrix, y, C), method=None, jac=gradient_reg, options={'maxiter': 3000})
    parameters = result.x
    cost = cost_function_reg(parameters, feature_matrix, y, C)

    print('Optimized Cost: \n', cost)
    print('Parameters are: \n', parameters)

    # Calculate training accuracy
    parameters = np.matrix(parameters)
    prediction_vector = predict(feature_matrix,parameters)
    comparison_vector = np.equal(prediction_vector, y)

    Correct = np.sum(comparison_vector == True)

    training_accuracy = Correct/len(y)
    print('The training accuracy of this model is: \n', training_accuracy)

    # scatter plot of X, y
    plot_data(X, y, ax2.flatten()[i])

    # Plot Decision Boundary
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(parameters.T))
    h = h.reshape(xx1.shape)
    ax2.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    ax2.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(training_accuracy, decimals=2), C))
    plt.show()






