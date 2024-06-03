import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from scipy.special import softmax

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes=12, lam=5e-2, T=1e-3):
        kmeans = KMeans(n_clusters=n_nodes).fit(X)

        affinity = .5 * lam * pairwise_distances(kmeans.cluster_centers_, metric='sqeuclidean')
        np.fill_diagonal(affinity, np.inf)

        for i in range(n_nodes):
            data_i = X[kmeans.labels_ == i]
            for j in range(i):
                data_j = X[kmeans.labels_ == j]
                data = np.vstack((data_i, data_j))

                segment = kmeans.cluster_centers_[j] - kmeans.cluster_centers_[i]
                projs = np.dot(data - kmeans.cluster_centers_[i], segment) / np.dot(segment, segment)
                nearests = kmeans.cluster_centers_[i] + np.outer(np.clip(projs, 0, 1), segment)
                diffs = data - nearests
                inertia_segment = np.einsum('ij,ij->i', diffs, diffs).mean()
                nearests = kmeans.cluster_centers_[i] + np.outer((projs > .5).astype(np.float64), segment)
                diffs = data - nearests
                inertia_points = np.einsum('ij,ij->i', diffs, diffs).mean()

                affinity[i, j] += inertia_segment - inertia_points


        affinity = softmax((affinity + affinity.T) / -T)
        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)
        self.clusters = [kmeans.cluster_centers_[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
