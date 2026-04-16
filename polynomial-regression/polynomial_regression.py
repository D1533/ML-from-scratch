

import numpy as np
from itertools import combinations_with_replacement

class PolynomialRegression:
    def __init__(self, degree, lambda_):
        self.degree = degree
        self.lambda_ = lambda_
        self.coeffs = None 

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

    def fit(self, X, y):
        X = self._design_matrix(X)
        I = np.eye(X.shape[1])
        I[0, 0] = 0
        self.coeffs = np.linalg.pinv(X.T @ X + self.lambda_ * I) @ X.T @ y          

    def predict(self, X):
        X = self._design_matrix(X)
        return X @ self.coeffs
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

