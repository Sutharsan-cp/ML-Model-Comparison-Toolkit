import numpy as np

def train_elastic_linear(X, y, lr=0.01, epochs=1000, lamda=0.1, alpha=0.5):
    """
    alpha=0 -> Ridge, alpha=1 -> Lasso, in between -> Elastic Net
    """
    m, n = X.shape
    weights = np.zeros((n, 1))
    bias = 0

    for epoch in range(epochs):
        y_pred = np.dot(X, weights) + bias
        dw = (1/m) * np.dot(X.T, (y_pred - y)) \
             + alpha * (lamda/m) * np.sign(weights) \
             + (1-alpha) * (lamda/m) * weights
        db = (1/m) * np.sum(y_pred - y)
        weights -= lr * dw
        bias -= lr * db

    return weights, bias
