import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from itertools import combinations_with_replacement

def random_linear_data(n_features, n=10, noise_std=10.0):
    X = np.random.uniform(-2, 2, size=(n, n_features))

    coeffs = 5*np.random.randn(n_features + 1)
    X_design = np.hstack([np.ones((n, 1)), X])

    y = X_design @ coeffs
    y += np.random.normal(0, noise_std, size=n)

    return X, y, coeffs


def main():

    # --- Setup ---
    degree = 1
    features = 1
    X, y, coeffs = random_linear_data(features, 100, 2)
    
    # --- Model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    
    model = LinearRegression(0)
    model.fit(X_train,y_train)
    
    # --- Plot ---
    X_vals = np.linspace(-5, 5, 100).reshape(-1, 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    mse_tests = []
    for i in range(0, len(model.coeffs_history), 5):
        model.coeffs = model.coeffs_history[i]
        y_test_pred = model.predict(X_test)
        mse_test = np.mean((y_test_pred - y_test)**2)
        mse_tests.append(mse_test)
        y_pred = model.predict(X_vals)

        ax1.clear()
        ax2.clear()
        
        ax1.scatter(X_train, y_train, color='black', label="Train data") 
        ax1.scatter(X_test, y_test, color='grey', label="Test data") 
        ax1.plot(X_vals, y_pred, color='red') # Model
        ax1.set_xlim(min(X) - 1, max(X) + 1)
        ax1.set_ylim(min(y) - 3, max(y) + 3)
        ax1.set_title(f"Epoch {i} \n Train MSE: {round(model.loss_history[i], 5)} \n Test MSE: {round(mse_test, 5)}")
        ax1.legend()

        ax2.plot(list(range(0, 5*len(mse_tests),5)), model.loss_history[0:i+1:5],label="train MSE")
        ax2.plot(list(range(0, 5*len(mse_tests), 5)), mse_tests[0:i+1],label="test MSE")
        ax2.set_xlim(1, 1000)
        ax2.set_ylim(0, 50)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE")
        ax2.legend()
        ax2.grid()

        plt.pause(0.05)


    plt.show()

if __name__ == "__main__":
    main()

