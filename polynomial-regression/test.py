import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from polynomial_regression import PolynomialRegression
from itertools import combinations_with_replacement

def random_polynomial_data(degree, n_features, n=10, noise_std=10.0):
    X = np.random.uniform(-5, 5, size=(n, n_features))

    terms = []
    for d in range(degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            terms.append(comb)

    coeffs = np.random.randn(len(terms))

    y = np.zeros(n)
    for coef, comb in zip(coeffs, terms):
        term_val = np.ones(n)
        for j in comb:
            term_val *= X[:, j]
        y += coef * term_val

    y += np.random.normal(0, noise_std, size=n)

    return X, y, coeffs, terms

def main():
    degree = 3
    features = 1
    X, y, coeffs, _ = random_polynomial_data(degree, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    model = PolynomialRegression(5, 0)
     
    model.fit(X_train,y_train)
    print("R2 score: ", model.score(X_test, y_test))
    
    X_vals = np.linspace(-5, 5, 100).reshape(-1,1)
    y_pred = model.predict(X_vals)
    plt.scatter(X, y)
    plt.plot(X_vals, y_pred)
    plt.show()

    '''
    x0_vals = np.linspace(-5, 5, 100)
    x1_vals = np.linspace(-5, 5, 100)
    X0, X1 = np.meshgrid(x0_vals, x1_vals)
    grid = np.c_[X0.ravel(), X1.ravel()]
    y_pred = model.predict(grid).reshape(X0.shape)
     
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0], X[:,1], y)   
    ax.plot_surface(X0, X1, y_pred, alpha=0.5)
    plt.show()
    '''

if __name__ == "__main__":
    main()

