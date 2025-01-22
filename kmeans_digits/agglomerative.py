import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
import numpy as np


digits = load_digits()
X = digits.data
y = digits.target

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)

n_clusters = 10
hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

hierarchical_labels = hierarchical.fit_predict(reduced_data)

hierarchical_silhouette = silhouette_score(reduced_data, hierarchical_labels)


def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster = data[labels == label]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {label}")
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid()

plot_clusters(reduced_data, hierarchical_labels, f"Hierarchical Clustering\nSilhouette Score: {hierarchical_silhouette:.2f}")
plt.show()

