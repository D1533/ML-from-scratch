
import numpy as np


class LogisticRegression:
    def __init__(self):
        self.coeffs = None
        self.coeffs_history = []
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, lr=0.01, epochs=1000):
        X = np.array(X)
        y = np.array(y)

        X = np.c_[np.ones(X.shape[0]), X]

        n_samples, n_features = X.shape
        self.coeffs = np.zeros(n_features)
        for _ in range(epochs):
            z = X @ self.coeffs
            p = self._sigmoid(z)

            grad = (X.T @ (p - y)) / n_samples
            self.coeffs -= lr * grad
            self.coeffs_history.append(self.coeffs.copy())
            
            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            self.loss_history.append(loss)

    def predict_prob(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(X @ self.coeffs)

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

    
