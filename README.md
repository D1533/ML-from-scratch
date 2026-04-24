
## Mathematical Background

## Linear Regression

Given a dataset $X \in \mathbb{R}^{m \times n}$ we want to learn a linear function that predicts $y \in \mathbb{R}$ from $x \in \mathbb{R}^n$, such that

$$
\hat{y} = w^T x + b,
$$

where $w \in \mathbb{R}^n, b \in \mathbb{R}$.

We define the Loss function as the MSE

$$
L(w, b) = \frac{1}{m}\Vert Xw + b\mathbb{1} - y \Vert^2.
$$

In order to minimize that function, we compute the  gradient of $L$

$$
\begin{aligned}
\nabla_w L &= \frac{2}{m}X^T(Xw + b\mathbb{1} - y), \\
\frac{\partial L}{\partial b} &= \frac{2}{m}\mathbb{1}^T(Xw + b\mathbb{1} - y),
\end{aligned}
$$

therefore 

$$
\begin{align*}
\nabla L &= \frac{2}{m}\left(X^T(Xw + b\mathbb{1} - y), \mathbb{1}^T(Xw + b\mathbb{1} - y)\right)^T.
\end{align*}
$$

Finally, the iterations for the gradient descent algorithm are defined as

$$
(w_{i+1}, b_{i+1})^T = (w_i, b_i)^T -\eta \nabla L(w_i, b_i).
$$

---

## Logistic Regression

Given a dataset $X \in \mathbb{R}^{m \times n}$, we want to learn a linear function that predicts a probability $y \in \{0,1\}$ from $x \in \mathbb{R}^n$, such that

$$
\hat{y} = \sigma(w^T x + b),
$$

where $w \in \mathbb{R}^n$, $b \in \mathbb{R}$, and $\sigma$ is the sigmoid function defined as

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

We define the loss function as the binary cross-entropy (log loss)

$$
L(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right].
$$

Let

$$
z = Xw + b\mathbb{1}, \quad \hat{y} = \sigma(z).
$$

Then the gradient of the loss is given by

$$
\nabla_w L = \frac{1}{m} X^T (\hat{y} - y),
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{m} \mathbb{1}^T (\hat{y} - y),
$$

therefore

$$
\nabla L = \frac{1}{m}\left(X^T (\sigma(Xw + b\mathbb{1}) - y), \mathbb{1}^T (\sigma(Xw + b\mathbb{1}) - y)\right)^T,
$$

where $\sigma(Xw + b)$ is applied componentwise.


Finally, the gradient descent updates are

$$
(w_{i+1}, b_{i+1})^T = (w_i, b_i)^T - \eta \nabla L(w_i, b_i).
$$

---

## Support Vector Machine (SVM)

Given a dataset $X \in \mathbb{R}^{m \times n}$ with labels $y^{(i)} \in \{-1, 1\}$, we aim to learn a linear classifier of the form

$$
f(x) = w^T x + b,
$$

where $w \in \mathbb{R}^n$, $b \in \mathbb{R}$, and the predicted class is given by

$$
\hat{y} = \mathrm{sign}(w^T x + b).
$$

The SVM seeks a hyperplane that maximizes the margin while penalizing margin violations. This leads to the primal optimization problem

$$
\min_{w,b} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \max(0, 1 - y^{(i)}(w^T x^{(i)} + b)),
$$

where $C > 0$ is a regularization parameter and

$$
\max(0, 1 - y^{(i)}(w^T x^{(i)} + b))
$$

is the hinge loss.

Let

$$
\ell_i = \max(0, 1 - y^{(i)}(w^T x^{(i)} + b)).
$$

Then the objective function becomes

$$
L(w,b) = \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \ell_i.
$$

Since the hinge loss is not differentiable everywhere, we use a subgradient.

For each sample $i$:

- if $y^{(i)}(w^T x^{(i)} + b) \geq 1$, it does not contribute to the gradient,
- if $y^{(i)}(w^T x^{(i)} + b) < 1$, it contributes to the gradient.

Thus, the subgradients are

$$
\nabla_w L = w - C \sum_{i: y^{(i)}(w^T x^{(i)} + b) < 1} y^{(i)} x^{(i)},
$$

$$
\frac{\partial L}{\partial b} = - C \sum_{i: y^{(i)}(w^T x^{(i)} + b) < 1} y^{(i)}.
$$

Finally, the decision function is

$$
f(x) = w^T x + b, \quad \hat{y} = \mathrm{sign}(f(x)).
$$
