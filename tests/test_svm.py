import numpy as np
import matplotlib.pyplot as plt
from svm import SVM

def make_data():
    X = np.random.randn(200, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    return X, y


def main():
    # --- Setup ---
    X, y = make_data()
    
    # --- Model ---
    model = SVM()
    model.fit(X, y, lr=1e-3, epochs=500, C=1.0)

    preds = model.predict(X)
    acc = np.mean(preds == y)

    # --- Plot ---
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    x_vals = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
    line, = ax.plot([], [], 'r')
    for epoch in range(0, len(model.coeffs_history), 5):

        model.w = model.coeffs_history[epoch][0]
        model.b = model.coeffs_history[epoch][1]

        w = model.w
        b = model.b

        if abs(w[1]) > 1e-8:
            y_vals = -(w[0] * x_vals + b) / w[1]

            line.set_data(x_vals, y_vals)

        ax.set_title(f"SVM epoch {epoch}")

        plt.pause(0.05)

    plt.show()

if __name__ == "__main__":
    main()


