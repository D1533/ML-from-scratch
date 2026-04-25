import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayes

def random_data(n=200):
        x0 = np.random.randn(n//2, 2) + np.array([-2, 0])
        x1 = np.random.randn(n//2, 2) + np.array([2, 0])

        X = np.vstack([x0, x1])
        y = np.array([0]*(n//2) + [1]*(n//2))

        noise_idx = np.random.choice(n, size=int(0.1*n), replace=False)
        y[noise_idx] = 1 - y[noise_idx]

        return X, y

def main():
    # --- Setup ---
    X, y = random_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Model ---
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = np.mean(y_pred == y_test)
    
    # --- Plot ---
    cmap = plt.get_cmap("tab10")  
    classes = np.unique(y)
    for i, c in enumerate(classes):
        plt.scatter(X_train[y_train == c, 0], X_train[y_train == c, 1], color=cmap(i), edgecolors="k", label=f"train class {c}")
    for i,c in enumerate(classes):
        plt.scatter(X_test[y_pred == c, 0], X_test[y_pred == c, 1], color=cmap(i), marker="x", s=100, label=f"test class {c}")

    plt.title(f"Naive Bayes Classification\n Accuracy: {acc}")
    plt.legend()
    plt.savefig("../images/naive_bayes_train.png")
    plt.show()

if __name__ == "__main__":
    main()



