
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

Given a dataset $X \in \mathbb{R}^{m \times n}$ with labels $y^{(i)} \in \text{\{ -1, 1 \}}$ , we aim to learn a linear classifier of the form

$$
f(x) = w^T x + b,
$$

where $w \in \mathbb{R}^n$, $b \in \mathbb{R}$, and the predicted class is given by

$$
\hat{y} = \mathrm{sign}(w^T x + b).
$$

The SVM seeks a hyperplane that maximizes the margin while penalizing margin violations. This leads to the primal optimization problem

$$
\min_{w,b} \quad \frac{1}{2} \Vert w \Vert^2 + C \sum_{i=1}^m \max(0, 1 - y^{(i)}(w^T x^{(i)} + b)),
$$

where $C > 0$ is a regularization parameter. Therefore the loss function is

$$
L(w,b) = \frac{1}{2}\Vert w\Vert^2 + C\sum_{i=1}^m \max(0, 1 - y^{(i)}(w^T x^{(i)} + b)).
$$

Notice that $L$ is not diferentiable, however, the terms in the sumatory only affect the derivative when $y^{(i)}(w^Tx^{(i)} + b) < 1$.

Let $\mathcal{M} =  { i : y^{(i)}(w^Tx^(i) + b) < 1} $, then

$$
\nabla_w L = w - C \sum_{i \in \mathcal{M}} y^{(i)} x^{(i)},
$$

$$
\frac{\partial L}{\partial b} = - C \sum_{i \in \mathcal{M}} y^{(i)}.
$$

Finally, the decision function is

$$
f(x) = w^T x + b, \quad \hat{y} = \mathrm{sign}(f(x)).
$$
