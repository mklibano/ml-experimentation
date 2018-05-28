import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Assignment6.plot_data import plot_data, plot_decision_boundary
from sklearn import svm
from Assignment6.gaussian_kernel import gaussian_kernel

# load data from file
path = os.getcwd() + "/data/ex6data1.mat"
data = loadmat(path)
X = data['X']
y = data['y']

# plotting the data
fig, ax = plt.subplots(figsize=(8,6))
ax = plot_data(X,y, ax)

# Train SVM with C=1 and plot decision boundary
svc = svm.LinearSVC(C=1)
svc.fit(X,y.ravel())
weights = svc.coef_[0]
intercept = svc.intercept_[0]
print('The accuracy score of this classifier with C=1: \n', svc.score(X,y))
plot_decision_boundary(X, weights, intercept, ax)

# Train SVM with C=100 and plot decision boundary
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2 = plot_data(X,y, ax2)

svc2 = svm.LinearSVC(C=100)
svc2.fit(X,y.ravel())
weights2 = svc2.coef_[0]
intercept2 = svc2.intercept_[0]
plot_decision_boundary(X, weights2, intercept2, ax2)
print('The accuracy score of this classifier with C=100: \n', svc2.score(X,y))

"""Implementing Gaussian Kernel"""

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(sigma, x1, x2)
print('Simulated result of the kernel function is \n', sim)


"""load and plot dataset 2"""
# load data from file
path = os.getcwd() + "/data/ex6data2.mat"
data = loadmat(path)
X = data['X']
y = data['y']

# plotting the data
plt.close()
fig3, ax3 = plt.subplots(figsize=(8,6))
ax = plot_data(X,y, ax3)

# setting initial parameters
C = 1
sigma = 0.1

# Training the SVM
svc3 = svm.SVC(C=100,kernel='rbf', gamma=10, probability = True)
svc3.fit(X,y.ravel())

# plotting the decision boundary
yp = svc3.predict_proba(X)[:,0]
fig4, ax4 = plt.subplots(figsize=(8,6))
ax4.scatter(X[:,0], X[:,1], s=30, c=yp, cmap='Reds')


"""Working with data set 3"""
# load data from file
path = os.getcwd() + "/data/ex6data3.mat"
data = loadmat(path)
X = data['X']
y = data['y']
yval = data['yval']
Xval = data['Xval']

C_params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,100]
Sigma_params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_C = 0
best_sigma = 0

for C in C_params:
    for sigma in Sigma_params:
        svc = svm.SVC(C=C, gamma=sigma)
        svc.fit(X,y.ravel())
        score = svc.score(Xval,yval.ravel())
        print('Accuracy of the model with C = {}, and sigma = {} is: {}'.format(C, sigma, score))

        if score > best_score:
            best_score, best_C, best_sigma = score, C, sigma

print('The best performing model has a test set accuracy of {} with C = {} and sigma = {} '.format(best_score, best_C, best_sigma))

# plot the decision boundary
fig5, ax5 = plt.subplots(figsize=(8,6))
plot_data(X,y, ax5)

svc = svm.SVC(C=best_C, gamma=best_sigma, probability=True)
svc.fit(X,y.ravel())
yp = svc.predict_proba(X)[:,0]
fig6, ax6 = plt.subplots(figsize=(8,6))
ax6.scatter(X[:,0], X[:,1], s=30, c=yp, cmap='Reds')
plt.show()













