
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from knn import KNN

def test_classification():
    def random_data(n=200):
        x0 = np.random.randn(n//2, 2) + np.array([-2, 0])
        x1 = np.random.randn(n//2, 2) + np.array([2, 0])

        X = np.vstack([x0, x1])
        y = np.array([0]*(n//2) + [1]*(n//2))

        # flip some labels (noise)
        noise_idx = np.random.choice(n, size=int(0.1*n), replace=False)
        y[noise_idx] = 1 - y[noise_idx]

        return X, y   
    
    X, y = random_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

    knn = KNN(10, "classification")
    knn.fit(X_train, y_train)

    print("score: ", knn.score(X_test, y_test))
    y_pred = knn.predict(X_test) 
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="bwr", marker="x", s=100, label="test (pred)")
    plt.show()

def test_regression():
    def random_data(n=200):
        X = np.random.rand(n, 2)

        y = (
            np.sin(2 * np.pi * X[:, 0]) +
            X[:, 1]**2 +
            0.1 * np.random.randn(n)
        )

        return X, y

    X, y = random_data()
    X_test, y_test = random_data(50)

    knn = KNN("regression")
    knn.fit(X, y)
    y_pred = knn.predict(X_test)

    mse = np.mean((y_test - y_pred) ** 2)
    print("MSE:", mse)

    # simple visualization (true vs predicted scatter)
    plt.scatter(y_test, y_pred)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("KNN Regression: True vs Predicted")
    plt.show()   

def main():
    test_classification()    
    #test_regression()

if __name__ == "__main__":
    main()


