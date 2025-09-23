import numpy as np

def sigmoid(z):
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def negative_log_likelihood(y, y_pred):
    """Compute binary cross-entropy loss."""
    m = len(y)
    return - (1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    """Train logistic regression using gradient descent."""
    m, n = X.shape
    weights = np.zeros((n, 1))
    bias = 0

    for epoch in range(epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        weights -= lr * dw
        bias -= lr * db

        if epoch % 100 == 0:
            loss = negative_log_likelihood(y, y_pred)
            #print(f"Epoch {epoch}, Loss: {loss:.4f}")
            #print(weights, bias)
    return weights, bias


def predict(X, weights, bias):
    """
    Predict class labels and probabilities for input data X.
    Returns: predicted class labels (0 or 1), predicted probabilities
    """
    y_pred_probs = sigmoid(np.dot(X, weights) + bias)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    return y_pred_labels, y_pred_probs


