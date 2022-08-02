import numpy as np
from numpy.random import uniform
import random


def euclidean(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class MyKMeans:
    def __init__(self, n_clusters=8, max_iter=750):
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = [random.choice(X)]
        for _ in range(self.n_clusters - 1):
            dists = np.sum([euclidean(centroid, X) for centroid in self.centroids], axis=0)
            # calculate distances from points to the centroids
            dists /= np.sum(dists)  # normalize the distances
            new_centroid_index, = np.random.choice(range(len(X)), size=1, p=dists)
            # choose remaining points based on their distances
            self.centroids += [X[new_centroid_index]]

        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]  # assign each data to the nearest centroid
            for x in X:
                dists = euclidean(x, self.centroids)
                centroid_index = np.argmin(dists)  # choose the minimum distance
                sorted_points[centroid_index].append(x)
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            # reassign centroids as mean of the points belonging to them

            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_indexes = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_index = np.argmin(dists)
            centroids.append(self.centroids[centroid_index])
            centroid_indexes.append(centroid_index)
        return centroids, centroid_indexes
