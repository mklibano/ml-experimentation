import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = np.divide(X - mu, sigma)

    return X_norm, mu


def pca(X):
    # normalize the dataset before PCA
    X_norm, mu = feature_normalize(X)

    # compute the covariance matrix
    X_norm = np.matrix(X_norm)
    sigma = (X_norm.T*X_norm)/(X_norm.shape[0])

    # perform SVD
    U,S,V = np.linalg.svd(sigma)

    return U, S


def project_data(X_norm, U, K):

    X_norm = np.matrix(X_norm)
    U_reduce = U[:,0:K]
    Z = X_norm*U_reduce
    return Z


def recover_data(Z, U, K):

    X_rec = Z*U[:,0:K].T
    return X_rec


def display_data(data):
    fig, axs = plt.subplots(1, data.shape[0], figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(wspace=.001)

    axs = axs.ravel()

    for i in range(data.shape[0]):
        face = np.reshape(data[i,:], (32,32))
        axs[i].imshow(face.T, cmap='gray')
        axs[i].set_title(str(250 + i))

    return fig


def main():
    # Load and visualize example dataset 1
    path = '/Users/markklibanov/Documents/GitHub/ml-experimentation/Assignment7/data/ex7data1.mat'
    data = loadmat(path)
    X = data['X']

    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1])

    # compute the principal components
    X_norm, mu = feature_normalize(X)
    U, S = pca(X)
    print('The top eigenvector is: \n', U)

    # Plot the normalized dataset
    fig,ax1 = plt.subplots()
    ax1.scatter(X_norm[:,0], X_norm[:,1], c='b')

    # Project data onto K dimensions
    K=1
    Z = project_data(X_norm, U, K)
    print('The first projected example is: \n', Z[0])

    # Recover data from projected space
    X_rec = recover_data(Z, U, K)
    print('The first recovered example is: \n', X_rec[0])

    """PCA on faces"""
    # plot the recovered data
    ax1.scatter(np.array(X_rec[:,0]), np.array(X_rec[:,1]),c='r')

    # Load the faces dataset from file
    path = '/Users/markklibanov/Documents/GitHub/ml-experimentation/Assignment7/data/ex7faces.mat'
    data = loadmat(path)
    X = data['X']

    # visualize the first 10 faces in the data
    fig = display_data(X[0:10,:])
    fig.suptitle('First 10 faces')

    # normalize the data before performing PCA
    X_norm, mu = feature_normalize(X)

    # Compute the PCA
    U, S = pca(X_norm)

    # Visualize the top 10 Eigenfaces
    fig = display_data(U[:,0:10].T)
    fig.suptitle('Top 10 Eigenfaces')

    # Project the face data into the PCA space using the top k components
    K = 100
    Z = project_data(X_norm, U, K)
    print('The projected data has size of: \n', Z.shape)

    # Visualizing data after dimensionality reduction
    X_rec = recover_data(Z, U, K)
    fig = display_data(X_rec[0:10,:])
    fig.suptitle('First 10 Recovered Faces')
    plt.show()

if __name__ == "__main__":
    main()




