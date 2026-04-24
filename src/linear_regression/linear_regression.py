
import numpy as np

class LinearRegression:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.coeffs = None
        self.coeffs_history = []
        self.loss_history = []

    def _design_matrix(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        return np.hstack([np.ones((X.shape[0], 1)), X])       

    def fit(self, X, y, lr=1e-3, epochs=1000):
        X = self._design_matrix(X)
        n, d = X.shape
        self.coeffs = np.zeros(d)

        for epoch in range(epochs):
            reg = np.zeros_like(self.coeffs)
            reg[1:] = 2 * self.lambda_ * self.coeffs[1:]
            grad = (2/n) * X.T @ (X @ self.coeffs - y) + reg
            self.coeffs -= lr * grad
            
            self.coeffs_history.append(self.coeffs.copy())
            self.loss_history.append(np.mean((X @ self.coeffs - y) ** 2))
            
            if epoch > 10 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-8:
                break

    def predict(self, X):
        X = self._design_matrix(X)
        return X @ self.coeffs
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

