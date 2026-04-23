
import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k, mode="classification"):
        if mode not in ["classification", "regression"]:
            raise ValueError("mode must be 'classification' or 'regression'")
        self.mode = mode
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X):
        X = np.array(X)
        y = []
        for x in X:
            distances = np.linalg.norm(x - self.X_train, axis=1)
            idx = np.argsort(distances)[:self.k]
            neighbors = self.y_train[idx]

            if self.mode == "classification":
                values, counts = np.unique(neighbors, return_counts=True)
                y.append(values[np.argmax(counts)])
            elif self.mode == "regression":
                y.append(np.mean(neighbors))

        return np.array(y)

    def score(self, X, y):
        X = np.array(X) 
        y_pred = self.predict(X)

        if self.mode == "classification":
            return np.mean(y_pred == y)
        else:
            return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


