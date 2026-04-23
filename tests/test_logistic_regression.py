import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

def test_1d():
    
    # --- Setup ---
    X = np.linspace(-5, 5, 200).reshape(-1, 1)
    true_w = 2.0
    true_b = -0.5

    probs = 1 / (1 + np.exp(-(true_w * X[:, 0] + true_b)))
    y = (probs > np.random.rand(len(probs))).astype(int)
    
    # --- Model ---
    model = LogisticRegression()
    model.fit(X, y, lr=0.1, epochs=200)
    

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    for i in range(0, len(model.coeffs_history), 5):

        ax1.clear()
        ax2.clear()

        model.coeffs = model.coeffs_history[i]
        p_hat = model.predict_prob(X)


        ax1.scatter(X[:, 0], y, label="data", alpha=0.4) # Data
        ax1.plot(X[:, 0], p_hat, label="learned p(x)", linewidth=2) # Model
        ax1.plot(X[:, 0], probs, label="true p(x)", linestyle="dashed") # Real p(x)
        ax1.legend()
        ax1.set_title(f"Logistic Regression (epoch {i}), Log Loss: {round(model.loss_history[i], 5)}")
        
        ax2.plot(model.loss_history[:i+1])
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("log loss")
        ax2.set_xlim(1, 1000)
        ax2.set_title("Log loss")
        ax2.grid()

        plt.pause(0.05)

    
    input()

def main():
    test_1d()

if __name__ == "__main__":
    main()
