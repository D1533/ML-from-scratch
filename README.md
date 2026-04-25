
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

## Polynomial Regression

Polynomial regression extends linear regression by mapping the input $x \in \mathbb{R}^n$ into a higher-dimensional feature space.

We define a feature map $\phi : \mathbb{R}^n \to \mathbb{R}^N$, where $\phi(x)$ contains all monomials of degree $d$, that is, all 
$x_1^{a_1}x_2^{a_2}\cdots x_n^{a_n}$ where $a_1 + a_2 + \cdots + a_n \leq d$.

Then we apply linear regression in this transformed space:

$$
\hat{y} = w^T \phi(x) + b.
$$

Notice that the model is still linear in the parameters $w$, even though it is nonlinear in the original input $x$.

Therefore, polynomial regression can be solved using the same optimization framework as linear regression by replacing $X$ with the transformed $\Phi(X)$:

The loss function becomes

$$
L(w, b) = \frac{1}{m}\left\Vert \Phi(X)w + b\mathbb{1} - y \right\Vert^2,
$$

which is identical in form to linear regression.

Thus, polynomial regression is a special case of linear regression applied in a higher-dimensional feature space.


## Logistic Regression

Given a dataset $X \in \mathbb{R}^{m \times n}$, we want to learn a linear function that predicts a probability $y \in (0, 1)$ from $x \in \mathbb{R}^n$, such that

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

where $\sigma(Xw + b\mathbb{1})$ is applied componentwise.


Finally, the gradient descent updates are

$$
(w_{i+1}, b_{i+1})^T = (w_i, b_i)^T - \eta \nabla L(w_i, b_i).
$$

---

## Support Vector Machine (SVM)

Given a dataset $X \in \mathbb{R}^{m \times n}$ with labels $y^{(i)} \in \lbrace -1, 1 \rbrace$ , we aim to learn a linear classifier of the form

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

Since the hinge loss is not differentiable, we use a subgradient. Defining the active set

$$
\mathcal{A} = \lbrace i : y^{(i)}(w^T x^{(i)} + b) < 1 \rbrace,
$$

a valid subgradient is

$$
\begin{aligned}
\partial_w L &= w - C \sum_{i \in \mathcal{A}} y^{(i)} x^{(i)}. \\
\partial_b L &= - C \sum_{i \in \mathcal{A}} y^{(i)}.
\end{aligned}
$$

therefore 

$$
\partial L = \left(w - C \sum_{i \in \mathcal{A}} y^{(i)} x^{(i)},  - C \sum_{i \in \mathcal{M}} y^{(i)}\\right)^T.
$$

Finally, the iterations for the gradient descent algorithm are

$$
(w_{i+1}, b_{i+1})^T = (w_i, b_i) - \eta\partial L
$$
