import numpy as np

class SVM:
    def __init__(self):
        self.w = None
        self.b = 0
        self.coeffs_history = []
        self.loss_history = []

    def fit(self, X, y, lr=1e-3, epochs=1000, C=1.0):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0
        for epoch in range(epochs):
            dw = np.zeros(n_features)
            db = 0
            loss = 0
            for i in range(n_samples):
                margin = y[i] * (X[i] @ self.w + self.b)
                if margin < 1:
                    dw -= C * y[i] * X[i]
                    db -= C * y[i]
                    loss += 1 - margin

            dw += self.w
            self.w -= lr * dw
            self.b -= lr * db

            self.coeffs_history.append((self.w.copy(), self.b))
            self.loss_history.append(0.5 * np.dot(self.w, self.w) + C * loss)

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
