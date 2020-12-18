import numpy as np
from k_means import *


class AgglomerativeKMeans:

    def __init__(self, data, initial_clusters=100):
        self.data = data
        self.centroids = self.initialize_centroids(initial_clusters)

    def initialize_centroids(self, cluster_count):
        classifier = KMeansCluster(cluster_count)
        classifier.fit(self.data, max_iter=300)
        return classifier.centroids

    def mean_clustering(self, target_clusters):
        dist_matrix = np.sqrt(np.square(self.centroids[:, np.newaxis, :] - self.centroids).sum(axis=-1))
        np.fill_diagonal(dist_matrix, np.Inf)
        while len(self.centroids) > target_clusters:
            closest_pair = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
            new_centroid = self.centroids[closest_pair, :].mean(axis=0)
            self.centroids[closest_pair[0], :] = self.centroids[closest_pair, :].mean(axis=0)
            new_diff = np.sqrt(np.square(self.centroids - new_centroid).sum(axis=-1))
            self.centroids = np.delete(self.centroids, closest_pair[1], 0)
            dist_matrix[closest_pair[0], :] = new_diff
            dist_matrix[:, closest_pair[0]] = new_diff
            dist_matrix = np.delete(dist_matrix, closest_pair[1], 0)
            dist_matrix = np.delete(dist_matrix, closest_pair[1], 1)

    def evaluate(self):
        pass
