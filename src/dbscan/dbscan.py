import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts

        self.labels_ = None
        self.visited_ = None

    def _region_query(self, X, i):
        distances = np.linalg.norm(X - X[i], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, i, neighbors, cluster_id):
        self.labels_[i] = cluster_id
        queue = list(neighbors)
        while queue:
            j = queue.pop()

            if not self.visited_[j]:
                self.visited_[j] = True
                j_neighbors = self._region_query(X, j)

                if len(j_neighbors) >= self.min_pts:
                    queue.extend(j_neighbors)

            if self.labels_[j] == -1:
                self.labels_[j] = cluster_id

    def fit_predict(self, X):
        X = np.asarray(X)
        n = X.shape[0]

        self.labels_ = np.full(n, -1)   
        self.visited_ = np.zeros(n, dtype=bool)

        cluster_id = 0

        for i in range(n):
            if self.visited_[i]:
                continue
            self.visited_[i] = True
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_pts:
                self.labels_[i] = -1
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

        return self.labels_
