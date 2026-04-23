import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans

def make_data():
    np.random.seed(0)

    c1 = np.random.randn(100, 2) + np.array([0, 0])
    c2 = np.random.randn(100, 2) + np.array([5, 5])
    c3 = np.random.randn(100, 2) + np.array([0, 5])

    return np.vstack([c1, c2, c3])

def main():
    # --- Setup ---
    X = make_data()
    
    # --- Model ---
    model = KMeans(n_clusters=3, max_iter=20, random_state=0)
    model.fit(X)  
    
    # --- Plot ---
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(len(model.centroids_history)):
        centroids = model.centroids_history[i]
        labels = model.labels_history[i]

        ax.clear()

        for k in range(model.n_clusters):
            pts = X[labels == k]
            ax.scatter(pts[:, 0], pts[:, 1])

        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X')
        ax.set_title(f"Iteration {i}, Inertia: {model.inertia_history[i]:.2f}")

        plt.pause(0.3)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
