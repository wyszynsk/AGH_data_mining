import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)] #initialize centroids 
        
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2) #distances between each point and each centroid
            clusters = np.argmin(distances, axis=1) #assigning each point to closest cluster
            new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(self.n_clusters)]) #calculating new centroids

            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break                                   
            self.centroids = new_centroids              

        self.labels_ = clusters

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2) 
        return np.argmin(distances, axis=1)
