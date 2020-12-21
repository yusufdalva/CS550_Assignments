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
        final_dim = self.data.shape[-1]
        var_count = np.prod(self.data.shape) // final_dim
        flat_data = np.reshape(self.data, (var_count, final_dim))
        np.fill_diagonal(dist_matrix, np.Inf)
        while len(self.centroids) > target_clusters:
            closest_pair = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
            clusters = np.argmin(self.get_distances_to_clusters(), axis=0)
            first_cluster = flat_data[clusters == closest_pair[0]]
            second_cluster = flat_data[clusters == closest_pair[1]]
            merged_cluster = np.concatenate((first_cluster, second_cluster), axis=0)
            new_centroid = merged_cluster.mean(axis=0)
            self.centroids[closest_pair[0], :] = new_centroid
            new_diff = np.sqrt(np.square(self.centroids - new_centroid).sum(axis=-1))
            self.centroids = np.delete(self.centroids, closest_pair[1], 0)
            dist_matrix[closest_pair[0], :] = new_diff
            dist_matrix[:, closest_pair[0]] = new_diff
            dist_matrix = np.delete(dist_matrix, closest_pair[1], 0)
            dist_matrix = np.delete(dist_matrix, closest_pair[1], 1)

    def get_distances_to_clusters(self):
        final_dim = self.data.shape[-1]
        var_count = np.prod(self.data.shape) // final_dim
        data = np.reshape(self.data, (var_count, final_dim))
        dist = np.sqrt(np.sum((data - self.centroids[:, np.newaxis]) ** 2, axis=-1))
        return dist

    def evaluate(self):
        dist = self.get_distances_to_clusters()
        return np.amin(dist, axis=0).mean(axis=0)

    def apply_model(self):
        dist = self.get_distances_to_clusters()
        clusters = np.argmin(dist, axis=0)
        clusters = np.reshape(clusters, self.data.shape[:-1])
        return self.centroids[clusters]
