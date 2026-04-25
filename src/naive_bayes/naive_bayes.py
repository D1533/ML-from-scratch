import numpy as np

class NaiveBayes:
    def __init__(self, eps=1e-9):
        self.eps = eps
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]

            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + self.eps
            self.priors[idx] = X_c.shape[0] / n_samples

    def _log_gaussian(self, class_idx, X):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        return -0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(((X - mean) ** 2) / var, axis=1))

    def _predict_log_proba(self, X):
        X = np.asarray(X)
        log_probs = []
        for idx, _ in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = self._log_gaussian(idx, X)
            log_probs.append(prior + likelihood)

        return np.array(log_probs).T  

    def predict(self, X):
        log_probs = self._predict_log_proba(X)
        class_indices = np.argmax(log_probs, axis=1)
        return self.classes[class_indices]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
