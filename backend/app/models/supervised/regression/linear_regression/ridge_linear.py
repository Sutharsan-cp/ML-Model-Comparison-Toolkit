import numpy as np

def train_ridge_linear(X, y, lr=0.01, epochs=1000, lamda=0.1):
    m, n = X.shape
    weights = np.zeros((n, 1))
    bias = 0

    for epoch in range(epochs):
        y_pred = np.dot(X, weights) + bias
        dw = (1/m) * np.dot(X.T, (y_pred - y)) + (lamda/m) * weights
        db = (1/m) * np.sum(y_pred - y)
        weights -= lr * dw
        bias -= lr * db

    return weights, bias
