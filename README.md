
## Linear Regression


## Linear Regression

Given a dataset $X \in \mathbb{R}^{m \times n}$ we want to learn a linear function that predicts $y \in \mathbb{R}$ from $x \in \mathbb{R}^n$, such that

$$
\hat{y} = w^T x + b
$$

The Loss function is just the MSE:

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




