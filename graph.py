import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from scipy.special import softmax

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes, M=1e16):
        kmeans = KMeans(n_clusters=n_nodes).fit(X)

        affinity = pairwise_distances(kmeans.cluster_centers_, metric='sqeuclidean')
        np.fill_diagonal(affinity, np.inf)

        q1 = np.quantile(affinity, .25)
        q3 = np.quantile(affinity, .75)

        gamma = np.log(M) / (q1 - q3)
        affinity = softmax(gamma * affinity)

        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)
        self.clusters = [kmeans.cluster_centers_[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
