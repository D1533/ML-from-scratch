
import numpy as np
from itertools import combinations_with_replacement

class PolynomialRegression:
    def __init__(self, degree, lambda_):
        self.degree = degree
        self.lambda_ = lambda_
        self.coeffs = None
        self.coeffs_history = []
        self.loss_history = []

    def _design_matrix(self, X):
        n_samples, n_features = X.shape
        terms = []
        for degree in range(self.degree + 1):
            for comb in combinations_with_replacement(range(n_features), degree):
                terms.append(comb)
        
        Phi = np.ones((n_samples, len(terms)))
        for i, comb in enumerate(terms):
            for j in comb:
                Phi[:, i] *= X[:, j]
        
        return Phi

    def fit(self, X, y, lr=1e-3, epochs=1000):
        X = self._design_matrix(X)
        n, d = X.shape
        self.coeffs = np.zeros(d)

        for epoch in range(epochs):
            grad = (2/n) * X.T @ (X @ self.coeffs - y) + 2 * self.lambda_ * self.coeffs
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

