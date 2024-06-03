import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import make_circles
from graph import Graph
from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score

np.random.seed(0)

X, y = make_circles(n_samples = 2048, factor=.5, noise=.05)

plt.scatter(X[:, 0], X[:, 1], color='black', s=.5)

net = Graph(n_classes=2)

net.fit(X)

colors = plt.cm.tab10(np.arange(2))

guess = net.predict(X)

for i in range(2):
    plt.scatter(net.clusters[i][:, 0], net.clusters[i][:, 1], color=colors[i], label=f'Cluster {i}')

plt.show()


ari = adjusted_rand_score(y, guess)
print(f"Adjusted Rand Index: {ari}")

