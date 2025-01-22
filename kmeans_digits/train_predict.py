import numpy as np
import matplotlib.pyplot as plt
from model import KMeans
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Custom KMeans and Scikit-Learn KMeans
custom_kmeans = KMeans(n_clusters=4)
custom_kmeans.fit(X)

sklearn_kmeans = SKLearnKMeans(n_clusters=4, random_state=42)
sklearn_kmeans.fit(X)

# New data points for prediction
new_data = np.array([[0, 5], [5, 5], [-3, -2]])

# Predicting clusters for new data
custom_predictions = custom_kmeans.predict(new_data)
sklearn_predictions = sklearn_kmeans.predict(new_data)

# Calculate silhouette scores for both models
custom_silhouette = silhouette_score(X, custom_kmeans.labels_)
sklearn_silhouette = silhouette_score(X, sklearn_kmeans.labels_)

# Colormap
colormap = plt.cm.get_cmap("viridis", custom_kmeans.n_clusters)

# Visualize synthetic data and clustering results

# Plot the synthetic data
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, edgecolor='k')
plt.title("Synthetic Data Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Custom KMeans plot with silhouette score
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels_, cmap=colormap, alpha=0.6, edgecolor="k", label="Original Data")
plt.scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1], c=np.arange(custom_kmeans.n_clusters), 
            cmap=colormap, marker='X', s=200, edgecolor="black", label="Centroids")
for i, point in enumerate(new_data):
    plt.scatter(point[0], point[1], c=colormap(custom_predictions[i]), edgecolor="black", s=150, label=f"New Data (Cluster {custom_predictions[i]})")
plt.title(f"Custom KMeans (Silhouette Score: {custom_silhouette:.2f})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Legend")

# Scikit-Learn KMeans plot with silhouette score
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_, cmap=colormap, alpha=0.6, edgecolor="k", label="Original Data")
plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], c=np.arange(sklearn_kmeans.n_clusters), 
            cmap=colormap, marker='X', s=200, edgecolor="black", label="Centroids")
for i, point in enumerate(new_data):
    plt.scatter(point[0], point[1], c=colormap(sklearn_predictions[i]), edgecolor="black", s=150, label=f"New Data (Cluster {sklearn_predictions[i]})")
plt.title(f"Scikit-Learn KMeans (Silhouette Score: {sklearn_silhouette:.2f})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Legend")

plt.tight_layout()
plt.show()
