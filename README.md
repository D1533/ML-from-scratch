



## Linear Regression

Given a dataset $X \in \mathbb{R}^{m \times n}$ we want to learn a linear function that predicts $y \in \mathbb{R}$ from $x \in \mathbb{R}^n$, such that

$$
\hat{y} = w^t x + b
$$

The Loss function is just the MSE
$$
L(w,b) = \frac{1}{m} \| Xw + b\mathbf{1} - y \|^2
$$
