import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from logistic_regression import LogisticRegression


def main():

    # --- Setup ---
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    true_w = 2.0
    true_b = -0.5

    probs = 1 / (1 + np.exp(-(true_w * X[:, 0] + true_b)))
    y = (probs > np.random.rand(len(probs))).astype(int)

    # --- Model ---
    model = LogisticRegression()
    model.fit(X, y, lr=0.1, epochs=200)


    frames = list(range(0, len(model.coeffs_history), 5))
    X_vals = X  
    predictions = []
    losses = []

    for i in frames:
        model.coeffs = model.coeffs_history[i]
        p_hat = model.predict_prob(X_vals)

        predictions.append(p_hat)
        losses.append(model.loss_history[i])

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    def update(idx):
        ax1.clear()
        ax2.clear()

        i = frames[idx]

        ax1.scatter(X[:, 0], y, alpha=0.4, label="data")
        ax1.plot(X_vals[:, 0], predictions[idx], linewidth=2, label="learned p(x)")
        ax1.plot(X_vals[:, 0], probs, linestyle="dashed", label="true p(x)")
        ax1.set_title(f"Epoch {i} \nLog Loss: {losses[idx]:.5f}")
        ax1.legend()

        ax2.plot(frames[:idx+1], losses[:idx+1])
        ax2.set_ylim(0, max(losses[:idx+1]))
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("log loss")
        ax2.set_xlim(0, max(frames))
        ax2.set_title("Log loss")
        ax2.grid()

    anim = FuncAnimation(fig, update, frames=len(frames), interval=50)

    anim.save("../images/logistic_regression_train.gif", writer=PillowWriter(fps=20))

    plt.show()

if __name__ == "__main__":
    main()
