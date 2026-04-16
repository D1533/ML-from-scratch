import numpy as np
from sklearn.datasets import make_classification
from logistic_regression import LogisticRegression

def test():
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )

    model = LogisticRegression()
    model.fit(X, y)
    pred = model.predict(X)
    print(np.mean(pred == y))

def main():
    test()




if __name__ == "__main__":
    main()
