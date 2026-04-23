import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

def test_1d():
    np.random.seed(42)

    # 1D input
    X = np.linspace(-5, 5, 200).reshape(-1, 1)

    # true underlying function (ground truth)
    true_w = 2.0
    true_b = -0.5

    probs = 1 / (1 + np.exp(-(true_w * X[:, 0] + true_b)))

    # sample labels
    y = (probs > np.random.rand(len(probs))).astype(int)

    model = LogisticRegression()
    model.fit(X, y, lr=0.1, epochs=2000)

    # predictions
    p_hat = model.predict_prob(X)

    print("accuracy:", np.mean(model.predict(X) == y))

    # plot
    plt.scatter(X[:, 0], y, label="data", alpha=0.5)

    plt.plot(X[:, 0], p_hat, label="learned p(x)", linewidth=2)
    plt.plot(X[:, 0], probs, label="true p(x)", linestyle="dashed")

    plt.legend()
    plt.title("Logistic Regression (1D)")
    plt.show()


if __name__ == "__main__":
    test_1d()
