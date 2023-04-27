import os
import sys
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, X, n_clusters, max_iterations):
        self.K = n_clusters # Number of clusters
        self.X = X
        self.max_iterations = max_iterations # max iteration so it won't run for inf time
        self.n_rows, self.n_cols = X.shape # num of examples, num of features
        self.plot_figure = True # plot figure
        
    def initialize_centroids(self, X):
        '''
        Randomly initializes centroids and returns them.
        '''
        centroids = np.zeros((self.K, self.n_cols)) # row , column full with zero 
        for k in range(self.K):
            centroid = X[np.random.choice(range(self.n_rows))] 
            centroids[k] = centroid
        return centroids 
    
    def create_cluster(self, X, centroids):
        '''
        Cluster func to compute the closest centroid using Euclidean distance  
        by calculating the distance of every point from centroid. 
        
        returns the clusters generated.
        '''
        clusters = [[] for _ in range(self.K)]
        for _idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point-centroids)**2, axis=1))
            ) 
            clusters[closest_centroid].append(_idx)
        return clusters 
    
    def calculate_new_centroids(self, cluster, X):
        '''
        Create new centroids by calculating the mean value of all the 
        samples assigned to each previous centroid.
        '''
        centroids = np.zeros((self.K, self.n_cols)) # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0) # find the value for new centroids
            centroids[idx] = new_centroid
        return centroids
    
    def predict_cluster(self, clusters, X):
        '''
        Predict cluster for set of data points
        '''
        y_pred = np.zeros(self.n_rows) # row1 fillup with zero
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
    
    def plot_fig(self):
        '''
        Visualize the results in 2D
        '''
        m, r = 4, 0
        n = int(np.ceil(len(self.Y) / m))
        fig, ax = plt.subplots(n, m)
        fig.set_size_inches(18.5, 2.5*m)
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
        
    def fit(self, X):
        '''
        Implements actual K-means algorithm 
        - Initialize random centroids
        - Create cluster
        - Calculate new centroids
        - Calculate difference
        - Do the prediction
        '''
        Y_pred, Centroids = [], []
        centroids = self.initialize_centroids(X)
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids)
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X) 
            diff = centroids - previous_centroids 
            if not diff.any():
                break
            y_pred = self.predict_cluster(clusters, X)
            Y_pred.append(y_pred)
            Centroids.append(centroids) 
        self.Y = Y_pred
        self.Centroids = Centroids
        return (y_pred, centroids)

### TESTING ###
if __name__ == "__main__":
    np.random.seed(82)
    n_clusters = 4 # num of cluster
    X, _ = make_blobs(n_samples=2000, n_features=2, centers=n_clusters)
    max_itrs = 100
    kmeans = KMeans(X, n_clusters, max_itrs)
    y_pred, centroids = kmeans.fit(X)
    kmeans.plot_fig()