import os
import sys
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from kmeans import KMeans

class KMeansPlusPlus(KMeans):
    def __init__(self, X, n_clusters, max_iterations):
        self.K = n_clusters # Number of clusters
        self.max_iterations = max_iterations # max iteration so it won't run for inf time
        self.X = X
        self.n_rows, self.n_cols = X.shape # num of examples, num of features
        self.plot_figure = True # plot figure
        
    def initialize_centroids(self, X):
        '''
        initializes the centroids for K-means++
        inputs:
            data - numpy array of data points having shape (200, 2)
            k - number of clusters
        '''
        # Randomly select data point and add to the list
        centroids = []
        centroids.append(X[np.random.randint(
                X.shape[0]), :])
        
        # Compute remaining k - 1 centroids
        for c_id in range(self.K - 1):
            # Initialize a list to store distances of data
            ## points from nearest centroid
            dist = []
            for i in range(X.shape[0]):
                point = X[i, :]
                d = sys.maxsize
                # Compute distance of 'point' from each of the previously
                # selected centroid and store the minimum distance
                for j in range(len(centroids)):
                    temp_dist = np.sqrt(np.sum((point-centroids[j])**2))
                    d = min(d, temp_dist)
                dist.append(d)
            # Select data point with maximum distance as our next centroid
            dist = np.array(dist)
            next_centroid = X[np.argmax(dist), :]
            centroids.append(next_centroid)
            dist = []
        return centroids 
    
    def plot_fig(self):
        '''
        Visualize the results in 2D
        '''
        m, r = 2, 0
        n = int(np.ceil(len(self.Y) / m))
        fig, ax = plt.subplots(n, m)
        fig.set_size_inches(18.5, 6*m)
        fig.suptitle('Visualize K-means iterations (2D)', fontsize=20)
        for i in range(n):
            for j in range(m):
                ax[i][j].scatter(self.X[:, 0], self.X[:, 1],c=self.Y[r], alpha=0.5)
                ax[i][j].scatter(self.Centroids[r][:, 0], self.Centroids[r][:, 1],c='black', s=80, marker='d')
                ax[i][j].set_title(f'Iteration: {r+1}')
                ax[i][j].set_xlabel('X_0')
                ax[i][j].set_ylabel('X_1')
                r+=1
                if r == len(self.Y):
                    for k in range(j+1, m):
                        ax[i][k].set_axis_off()
                    plt.show()
                    return
        plt.show() 

### TESTING ###
if __name__ == "__main__":
    np.random.seed(82)
    n_clusters = 4 # num of cluster
    # create dataset using make_blobs from sklearn datasets
    X, _ = make_blobs(n_samples=2000, n_features=2, centers=n_clusters)
    max_itrs = 100
    kmeans_ = KMeansPlusPlus(X, n_clusters, max_itrs)
    y_pred, centroids = kmeans_.fit(X)
    kmeans_.plot_fig()