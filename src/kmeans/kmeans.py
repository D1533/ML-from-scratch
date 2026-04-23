
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.centroids = None
        self.labels_ = None
        self.centroids_history = []
        self.labels_history = []
        self.inertia_history = []

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        return X[idx]

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        self.centroids_history = []
        self.labels_history = []
        self.inertia_history = []

        for _ in range(self.max_iter):
            # assignment step
            distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # store history BEFORE update (like epoch snapshot)
            self.centroids_history.append(self.centroids.copy())
            self.labels_history.append(labels.copy())

            inertia = np.sum((X - self.centroids[labels])**2)
            self.inertia_history.append(inertia)

            # update step
            new_centroids = np.zeros_like(self.centroids)

            for k in range(self.n_clusters):
                points = X[labels == k]
                if len(points) == 0:
                    new_centroids[k] = X[np.random.randint(len(X))]
                else:
                    new_centroids[k] = points.mean(axis=0)

            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels_ = labels
        return self
        
    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


