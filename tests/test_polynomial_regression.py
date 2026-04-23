import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from polynomial_regression import PolynomialRegression
from itertools import combinations_with_replacement

def random_polynomial_data(degree, n_features, n=10, noise_std=10.0):
    X = np.random.uniform(-2, 2, size=(n, n_features))

    terms = []
    for d in range(degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            terms.append(comb)

    coeffs = np.random.randn(len(terms))

    y = np.zeros(n)
    for coef, comb in zip(coeffs, terms):
        term_val = np.ones(n)
        for j in comb:
            term_val *= X[:, j]
        y += coef * term_val

    y += np.random.normal(0, noise_std, size=n)

    return X, y, coeffs, terms

def main():
    degree = 5
    features = 1
    X, y, coeffs, _ = random_polynomial_data(degree, features, 50, 2)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    model = PolynomialRegression(degree, 0)
     
    model.fit(X_train,y_train)
    
    X_vals = np.linspace(-5, 5, 100).reshape(-1, 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    for i in range(0, len(model.coeffs_history), 5):
        model.coeffs = model.coeffs_history[i]
        y_pred = model.predict(X_vals)

        ax1.clear()
        ax2.clear()

        ax1.scatter(X, y, color='black')
        ax1.plot(X_vals, y_pred, color='red')
        ax1.set_xlim(min(X) - 1, max(X) + 1)
        ax1.set_ylim(min(y) - 10, max(y) + 10)
        ax1.set_title(f"Iteration {i}, \nMSE: {round(model.loss_history[i], 2)}")

        ax2.plot(model.loss_history[:i+1])
        ax2.set_title("Training MSE")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("MSE")
        ax2.set_xlim(1, 1000)
        ax2.grid()

        plt.pause(0.05)


    input()

if __name__ == "__main__":
    main()

