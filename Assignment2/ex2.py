import os
import numpy as np
import matplotlib.pyplot as plt
from Assignment2.sigmoid import sigmoid
from Assignment2.cost_function import cost_function
from Assignment2.gradient import gradient
from Assignment2.plotDecisionBoundary import plot_decision_boundary
import scipy.optimize as opt
from Assignment2.predict import predict


# load training data
path = os.getcwd() + '/data/ex2data1.txt'
data = np.loadtxt(path, delimiter=',')

# load data into feature matrix and output vector
X = np.matrix(data[:,0:2])
y = np.matrix(data[:,2:3])

"""
# plotting the data
positive = np.where(y > 0)[0]
np.reshape(positive, (len(positive),1))
pos1 = np.squeeze(np.asarray(X[positive,0]))
pos2 = np.squeeze(np.asarray(X[positive,1]))

negative = np.where(y < 1)[0]
np.reshape(negative, (len(negative),1))
neg1 = np.squeeze(np.asarray(X[negative,0]))
neg2 = np.squeeze(np.asarray(X[negative,1]))

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(pos1, pos2, s=50, c='b', marker='o', label='Admitted')
ax.scatter(neg1, neg2, s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
"""

# visualizing the sigmoid function
z = np.arange(-10,10, step=1)

fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.plot(z, sigmoid(z), 'r')


"""Computing the cost and gradient"""

# adding the intercept term
m, n = X.shape
z = np.ones((m, 1), dtype=int)
X = np.append(z, X, axis=1)

# initializing fitting parameters
initial_theta = np.zeros(3)

cost = cost_function(initial_theta, X, y)
grad = gradient(initial_theta, X, y)


print('Cost: \n', cost)
print('Grad: \n', grad)


# computing that parameters that minimize error
result = opt.fmin_tnc(func=cost_function, x0=initial_theta, fprime=gradient, args=(X, y))
a = result[0]
cost = cost_function(result[0], X, y)

print('Optimized Cost: \n', cost)
print('Parameters are: \n', result[0])

# plotting the decision boundary
plot_decision_boundary(result[0],X,y)

# predict admission probability given exam scores of 45 and 85
feature = np.matrix('1;45;85')
theta = np.matrix(result[0])
z = theta*feature
prediction = sigmoid(z)
print('The admission probability for this student is: \n', str(prediction[0,0]))

# Calculate training accuracy
prediction_vector = predict(X,theta)
comparison_vector = np.equal(prediction_vector, y)

Correct = np.sum(comparison_vector == True)

training_accuracy = Correct/m
print('The training accuracy of this model is: \n', training_accuracy)











