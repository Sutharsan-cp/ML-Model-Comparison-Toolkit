import numpy as np
from .metrics import evaluate_model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def negative_log_likelihood(y, y_pred, weights, lamda):
    m = len(y)
    l2_term = (lamda / (2*m)) * np.sum(np.square(weights))
    return - (1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)) + l2_term

def train_ridge_logistic(X, y, lr=0.01, epochs=1000, lamda=0.1):
    m, n = X.shape
    weights = np.zeros((n, 1))
    bias = 0

    for epoch in range(epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (y_pred - y)) + (lamda/m) * weights
        db = (1/m) * np.sum(y_pred - y)
        weights -= lr * dw
        bias -= lr * db

        if epoch % 100 == 0:
            loss = negative_log_likelihood(y, y_pred, weights, lamda)
            #print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, bias

def predict(X, weights, bias):
    y_pred_probs = sigmoid(np.dot(X, weights) + bias)
    return (y_pred_probs >= 0.5).astype(int), y_pred_probs
