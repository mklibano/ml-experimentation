import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Assignment5.cost_function_reg import cost_function_reg
from Assignment5.cost_function_reg import gradient_reg
from scipy.optimize import minimize
from Assignment5.learning_curves import learning_curve
from Assignment5.poly_features import poly_features
from Assignment5.train_linear_reg import train_linear_reg
from Assignment5.feature_normalize import feature_normalize
from Assignment5.plot_fit import plot_fit
from Assignment5.validation_curve import validation_curve

"""load data from file and visualize it"""

# load the data from file
path = os.getcwd() + '/data/ex5data1.mat'
data = loadmat(path)

# load the training set
X = np.matrix(data['X'], dtype=np.float64)
y = data['y']

# load the cross-validation set
Xval = np.matrix(data['Xval'], dtype=np.float64)
yval = data['yval']

# load the test set
Xtest = np.matrix(data['Xtest'], dtype=np.float64)
ytest = data['ytest']

# visualize the data
plt.scatter(np.array(X),y, marker='x', c='r')
plt.xlabel('Change on water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

"""Compute cost with initial parameters of theta = [1,1]"""
# initialize parameters
theta = np.matrix([1,1])
reg = 1
m = len(y)

# add ones column for the intercept term in the feature matrix
X = np.c_[np.ones((len(y),1)), X]
Xval = np.c_[np.ones((len(yval),1)), Xval]
Xtest = np.c_[np.ones((len(yval),1)), Xtest]

# compute initial error
cost = cost_function_reg(theta, X,y,reg)
grad = gradient_reg(theta, X,y,reg)

print('Cost with initial parameters: \n ', cost)
print('Gradient with initial parameters: \n ', grad)

"""minimize the cost function and find the optimal model parameters"""
reg = 0

# iterate for max 50 cycles
result = minimize(cost_function_reg, theta, args=(X, y, reg),
                  method=None, jac=gradient_reg, options={'maxiter': 100})

theta = result.x

print('Model parameters that minimize the cost function are: : \n ', theta)

"""plotting the linear hypothesis function"""
# Line that represents the decision boundary
plot_x = np.arange(np.min(X[:, 1])-5, np.max(X[:, 1])+5, 1)
plot_y = theta[0] + plot_x*theta[1]
plt.plot(plot_x, plot_y.T)

"""obtaining learning curves"""
reg = 0
error_train, error_val = learning_curve(X,y, Xval, yval, reg)

fig2, ax2 = plt.subplots()
ax2.plot(error_train, c='r')
ax2.plot(error_val, c='b')
ax2.set_xlabel('Number of training examples')
ax2.set_ylabel('Error')

"""Generating a more complex model to prevent underfitting"""
p = 8
reg = 0


# Map X onto polynomial features and normalize
X_poly = poly_features(X[:,1], p)
X_poly, mu, sigma = feature_normalize(X_poly)
X_poly = np.c_[np.ones((len(y),1)), X_poly]

# Map Xval onto polynomial features and normalize
Xval_poly = poly_features(Xval[:,1], p)
Xval_poly = np.divide(Xval_poly - mu, sigma)
Xval_poly = np.c_[np.ones((len(yval),1)), Xval_poly]

# Map Xtest onto polynomial features and normalize
Xtest_poly = poly_features(Xtest[:,1], p)
Xtest_poly = np.divide(Xtest_poly - mu, sigma)
Xtest_poly = np.c_[np.ones((len(ytest),1)), Xtest_poly]


# Train the polynomial model to minimize the cost function
theta = train_linear_reg(X_poly,y, reg)
fig3, ax3 = plt.subplots()
plt.scatter(np.array(X[:,1]),y, marker='x', c='r')
plot_fit(np.min(X[:,1]), np.max(X[:,1]), mu, sigma, theta, p)
ax3.set_xlabel('Change in water level (x)')
ax3.set_ylabel('Water flowing out of the dam (y)')


error_train, error_val = learning_curve(X_poly, y, Xval_poly, yval,reg)

fig4, ax4 = plt.subplots()
ax4.plot(error_train, c='b')
ax4.plot(error_val, c='g')
ax4.set_xlabel('Number of training examples')
ax4.set_ylabel('Error')


"""Using the validation set to select the optimal regularization value"""

lambda_vec, error_train, error_val = validation_curve(X_poly,y,Xval_poly,yval)

plt.subplot()
plt.plot(lambda_vec, error_train, lambda_vec, error_val);
plt.title('Selecting \lambda using a cross validation set')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()



