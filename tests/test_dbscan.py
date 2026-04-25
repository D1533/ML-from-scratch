import numpy as np
import matplotlib.pyplot as plt
from dbscan import DBSCAN


def random_data(n=300):
    n1 = n // 2
    n2 = n // 2

    # --- inner circle ---
    theta1 = np.random.rand(n1) * 2 * np.pi
    r1 = 1.0 + 0.1 * np.random.randn(n1)

    x1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]

    # --- outer circle ---
    theta2 = np.random.rand(n2) * 2 * np.pi
    r2 = 3.0 + 0.1 * np.random.randn(n2)

    x2 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]

    # --- combine ---
    X = np.vstack([x1, x2])

    # optional noise
    noise = np.random.uniform(-4, 4, (30, 2))
    X = np.vstack([X, noise])

    return X

def main():
    # --- Setup ---
    X = random_data()

    # --- Model ---
    model = DBSCAN(eps=0.8, min_pts=5)
    labels = model.fit_predict(X)

    # --- Plot ---
    cmap = plt.get_cmap("tab10")
    classes = np.unique(labels)
    for i, c in enumerate(classes):
        if c == -1:
            plt.scatter(X[labels == c, 0], X[labels == c, 1], color="black", marker="x", s=80, label="noise")
        else:
            plt.scatter(X[labels == c, 0], X[labels == c, 1], color=cmap(i), edgecolors="k", label=f"cluster {c}")

    plt.title("DBSCAN")
    plt.legend()
    plt.savefig("../images/dbscan.png")
    plt.show()

if __name__ == "__main__":
    main()
