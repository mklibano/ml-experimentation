import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# find closest centroids to each example in the dataset
def closest_centroids(X, centroids):

    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        min_dist = 100000000
        for k in range(K):
            dist = np.sum((X[i,:] - centroids[k,:])**2)
            if dist < min_dist:
                idx[i] = k
                min_dist = dist

    return idx


def compute_centroids(X, idx, K):

    n = X.shape[1]

    centroids = np.zeros((K,n))

    for i in range(K):
        Ck = np.where(idx == i)[0]
        num_Ck = len(Ck)
        centroids[i, :] = np.sum(X[Ck, :], axis=0)/num_Ck

    return centroids


def run_k_means(X, init_centroids, max_iters):

    m, n = X.shape
    K = init_centroids.shape[0]
    centroids = init_centroids
    idx = np.zeros(m)

    for i in range(max_iters):

        # Centroid assignment step
        idx = closest_centroids(X, centroids)

        # move centroid step
        centroids = compute_centroids(X, idx, K)

    return centroids, idx


def kmeans_init_centroids(X, K):

    return X[np.random.choice(X.shape[0], K)]


"""K-means on example datasets"""
# Load example dataset from file
path = os.getcwd() + '/data/ex7data2.mat'
data = loadmat(path)
X = data['X']

# select the initial set of centroids
K = 3
init_centroid = np.array([[3,3], [6,2], [8,5]])

# Perform cluster assignment step (find closest centroid for each example)
idx = closest_centroids(X,init_centroid)
print('Closest centroids for the first 3 examples: \n', idx[0:3])

# Perform move centroid step (compute mean of all assigned examples)
centroids_1 = compute_centroids(X, idx, K)
print('Centroids computed after first iterations: \n', centroids_1)


# run the k-means clustering algorithm and plot the data
init_centroids = kmeans_init_centroids(X, K)
print('Randomly initialized centroids: \n', init_centroids)
centroids, idx = run_k_means(X, init_centroid, 10)


cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
ax.legend()


"""K-means image compression"""

# Load image from file
path = os.getcwd() + '/data/bird_small.mat'
data = loadmat(path)
A = data['A']

# normalize image
A = A/255

# Reshape the image into a Nx3 matrix with N = # of pixels
X = np.reshape(A, (A.shape[0]*A.shape[1], A.shape[2]))

# Initialize k-means parameters
K = 16
max_iters = 10
init_centroids = kmeans_init_centroids(X, K)

# run the k-means clustering algorithms
centroids, idx = run_k_means(X, init_centroids, max_iters)

# Assign each pixel to the closest centroid and recover the image from the indeices
idx = closest_centroids(X, centroids)
X_recovered = centroids[idx.astype(int),:]

# Reshape the recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

# plot the images side by side
plt.subplot(121)
plt.imshow(A)
plt.subplot(122)
plt.imshow(X_recovered)
plt.show()
