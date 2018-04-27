import os
import numpy as np
import matplotlib.pyplot as plt
from computeCost import compute_cost
from gradientDescent import gradient_descent


# load data from file into matrix
path = os.getcwd() + '/data/ex1data1.txt'
data = np.loadtxt(path, delimiter=',')

X = data[:, 0:1]
y = data[:, 1:2]

m = len(y)

# plot data
plt.scatter(X, y, alpha=0.5)


# append a column of ones to X
z = np.ones((m, 1), dtype=int)
X = np.append(z, X, axis=1)


# Initialize fitting parameters
theta = np.zeros((2, 1), dtype=int)
iterations = 1500
alpha = 0.01

X = np.matrix(X)
y = np.matrix(y)
theta = np.matrix(np.array([0,0]))


# compute cost function
cost = compute_cost(X, y, theta)
print(cost)

# minimize error
theta, cost = gradient_descent(X, y, theta, alpha, iterations)

# print the minimized error and the resulting parameter vector
print(theta)
print(cost[len(cost)-1])

# predict profit for 35K and 70K people

predict1 = np.matrix([1,3.5])*theta.T
print(predict1)
predict2 = np.matrix([1, 7])*theta.T
print(predict2)


# plotting the regression model
x = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
f = theta[0, 0] + (theta[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter([X[:, 1]],[y],label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# plotting the cost function
fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.plot(np.arange(iterations), cost, 'r')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Error vs. Training Epoch')
plt.show()









