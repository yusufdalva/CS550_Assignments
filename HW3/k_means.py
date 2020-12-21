import numpy as np


class KMeansCluster:

    def __init__(self, no_of_clusters):
        self.cluster_count = no_of_clusters
        self.centroids = np.array([])

    def initialize_mean_vectors(self, data):
        final_dim = data.shape[-1]
        var_count = np.prod(data.shape) // final_dim
        reshaped_data = np.reshape(data, (var_count, final_dim))
        uniform_data = np.unique(reshaped_data, axis=0)  # Selecting centroids from unique pixels for not getting empty clusters
        np.random.shuffle(uniform_data)
        assert self.cluster_count <= len(uniform_data)
        mean_centroids = uniform_data[:self.cluster_count, :]
        self.centroids = mean_centroids

    def get_closest_centroids(self, data):
        final_dim = data.shape[-1]
        var_count = np.prod(data.shape) // final_dim
        data = data.reshape((var_count, final_dim))
        dist = np.sqrt(np.sum((data - self.centroids[:, np.newaxis])**2, axis=-1))
        return np.argmin(dist, axis=0)

    def update_centroids(self, data, clusters):
        final_dim = data.shape[-1]
        var_count = np.prod(data.shape) // final_dim
        data = data.reshape((var_count, final_dim))
        mean_centroids = []
        for cluster in range(len(self.centroids)):
            centroid = np.mean(data[clusters == cluster], axis=0)
            mean_centroids.append(centroid)
        return np.array(mean_centroids)

    def fit(self, data, max_iter=300):
        self.initialize_mean_vectors(data)
        prev_centroids = self.centroids
        for iter_idx in range(max_iter):
            clusters = self.get_closest_centroids(data)
            self.centroids = self.update_centroids(data, clusters)
            diff = np.mean(np.abs(self.centroids - prev_centroids), axis=0)
            if np.mean(diff) <= 0.1:
                return
            prev_centroids = self.centroids

    def evaluate(self, data):
        final_dim = data.shape[-1]
        var_count = np.prod(data.shape) // final_dim
        data = data.reshape((var_count, final_dim))
        dist = np.sqrt(np.sum((data - self.centroids[:, np.newaxis]) ** 2, axis=-1))
        return np.amin(dist, axis=0).mean(axis=0)

    def apply_model(self, data):
        clusters = self.get_closest_centroids(data)
        clusters = np.reshape(clusters, data.shape[:-1])
        return self.centroids[clusters]
