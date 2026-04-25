import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]

    n, d = X.shape
    features = [np.ones(n)]  

    for deg in range(1, degree + 1):
        for comb in combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            for j in comb:
                feat *= X[:, j]
            features.append(feat)

    return np.vstack(features).T


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

    # --- Setup ---
    degree = 3
    features = 1
    X, y, coeffs, _ = random_polynomial_data(degree, features, 100, 2)
    X_poly = polynomial_features(X, degree) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    
    X_train_poly = polynomial_features(X_train, degree)
    X_test_poly = polynomial_features(X_test, degree)

    # --- Model ---
    model = LinearRegression(0)
    model.fit(X_train_poly, y_train)
    

    # --- History computations ---
    X_vals = np.linspace(-5, 5, 100).reshape(-1, 1)
    X_vals_poly = polynomial_features(X_vals, degree)
    frames = list(range(0, len(model.coeffs_history), 5))

    train_losses = []
    test_losses = []
    predictions = []

    for i in frames:
        model.coeffs = model.coeffs_history[i]

        y_pred = model.predict(X_vals_poly)
        predictions.append(y_pred)

        y_test_pred = model.predict(X_test_poly)
        test_losses.append(np.mean((y_test_pred - y_test) ** 2))

        train_losses.append(model.loss_history[i])

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    def update(idx):
        ax1.clear()
        ax2.clear()

        i = frames[idx]

        ax1.scatter(X_train, y_train, color='black', label="Train data")
        ax1.scatter(X_test, y_test, color='grey', label="Test data")
        ax1.set_xlim(X.min() - 2, X.max() + 2) 
        ax1.set_ylim(y.min() - 2, y.max() + 2)
        ax1.plot(X_vals, predictions[idx], color='red')
        ax1.set_title(f"Epoch {i}\n" f"Train MSE: {train_losses[idx]:.5f}\n" f"Test MSE: {test_losses[idx]:.5f}")
        ax1.legend()

        ax2.plot(frames[:idx+1], train_losses[:idx+1], label="train MSE")
        ax2.plot(frames[:idx+1], test_losses[:idx+1], label="test MSE")
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0, max(max(train_losses, test_losses)))
        ax2.legend()
        ax2.grid()

    anim = FuncAnimation(fig, update, frames=len(frames), interval=50)

    anim.save("../images/polynomial_regression_train.gif", writer=PillowWriter(fps=20))

    plt.show()





if __name__ == "__main__":
    main()

