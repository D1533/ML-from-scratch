



## Linear Regression

Given a dataset $X \in \mathbb{R}^{m \times n}$ we want to learn a linear function that predicts $y \in \mathbb{R}$ from $x \in \mathbb{R}^n$, such that

$$
\hat{y} = w^t x + b
$$

The Loss function is just the MSE
$$
L(w,b) = \frac{1}{m} \| Xw + \mathbb{1}b - y\|^2
$$
whose gradient is 
$$
\begin{aligned}
\nabla_w L &= \frac{2}{m}X^t(Xw + b\mathbb{1} - y), \\
\frac{\partial L}{\partial b} &= \frac{2}{m}\mathbb{1}^t(Xw + b\mathbb{1} - y),
\end{aligned}
$$

therefore 

$$
\begin{align*}
\nabla L &= \frac{2}{m}\left(X^t(Xw + b\mathbb{1} - y), \mathbb{1}^t(Xw + b\mathbb{1} - y)\right)^t
\end{align*}
$$


So the updates of the gradient descent algorithms are


---

### Vectorized Form

Let:
- \( X \in \mathbb{R}^{m \times n} \)
- \( y \in \mathbb{R}^m \)

\[
\hat{y} = Xw + b
\]

\[
\nabla_w L = \frac{2}{m} X^\top (Xw + b\mathbf{1} - y)
\]

\[
\frac{\partial L}{\partial b} = \frac{2}{m} \mathbf{1}^\top (Xw + b\mathbf{1} - y)
\]

---

### Optimization (Gradient Descent)

\[
w \leftarrow w - \eta \nabla_w L
\]

\[
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]
