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
            for j in range(i):
                data = np.vstack((X[kmeans.labels_ == i], X[kmeans.labels_ == j]))

                diffs_null = data - np.vstack((kmeans.cluster_centers_[i], kmeans.cluster_centers_[j]))[:, np.newaxis]
                dists_null = np.min(np.einsum('ijk,ijk->ij', diffs_null, diffs_null), axis=0)
                inertia_null = dists_null.mean()

                segment = kmeans.cluster_centers_[j] - kmeans.cluster_centers_[i]
                dots = np.dot(data - kmeans.cluster_centers_[i], segment) / np.dot(segment, segment)
                projs = kmeans.cluster_centers_[i] + np.outer(np.clip(dots, 0, 1), segment)
                diffs_alt = data - projs
                dists_alt = np.einsum('ij,ij->i', diffs_alt, diffs_alt)
                inertia_alt = dists_alt.mean()

                affinity[i, j] += inertia_alt - inertia_null

        affinity = softmax((affinity + affinity.T) / -T)
        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)
        self.clusters = [kmeans.cluster_centers_[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
