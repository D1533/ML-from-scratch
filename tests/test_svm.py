import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.model_selection import train_test_split
from svm import SVM


def make_data():
    X = np.random.randn(200, 2)

    margin = 0.2 

    y = np.where(X[:, 0] + X[:, 1] > margin, 1,
        np.where(X[:, 0] + X[:, 1] < -margin, -1, 0)
    )

    mask = y != 0
    X = X[mask]
    y = y[mask]

    noise_ratio = 0.1
    n_noisy = int(noise_ratio * len(y))
    idx = np.random.choice(len(y), n_noisy, replace=False)
    y[idx] *= -1

    return X, y


def main():
    # --- Setup ---
    X, y = make_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # --- Model ---
    model = SVM()
    model.fit(X_train, y_train, lr=1e-3, epochs=500, C=1.0)

    # --- Plot ---
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    def update(epoch):
        ax.clear()

        model.w = model.coeffs_history[epoch][0]
        model.b = model.coeffs_history[epoch][1]

        w = model.w
        b = model.b

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_acc = np.mean(train_preds == y_train)
        test_acc = np.mean(test_preds == y_test)

        loss = model.loss_history[epoch]
        
        ax.scatter(X[:, 0],X[:, 1] , c=y, edgecolor='k')

        ax.set_xlim(X[:, 0].min()-1, X[:, 0].max()+1)
        ax.set_ylim(X[:, 1].min()-1, X[:, 1].max()+1)

        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, 'r')

        ax.set_title(f"Epoch {epoch} \n Loss: {loss:.5f}\nTrain acc: {train_acc:.2f}, Test acc: {test_acc:.2f}")


    anim = FuncAnimation(fig, update, frames=range(0, len(model.coeffs_history), 5), interval=50)
    anim.save("../images/svm_train.gif", writer=PillowWriter(fps=20))

    plt.show()


if __name__ == "__main__":
    main()
