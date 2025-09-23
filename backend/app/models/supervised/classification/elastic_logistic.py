import numpy as np
from .metrics import evaluate_model

def sigmoid(z):
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))

def negative_log_likelihood(y, y_pred, weights, lamda, alpha):
    """
    Compute the negative log-likelihood loss with Elastic Net regularization.
    y: true labels
    y_pred: predicted probabilities
    weights: model weights
    lamda: regularization strength
    alpha: mixing parameter (0 = Ridge, 1 = Lasso)
    """
    m = len(y)
    l1_term = alpha * (lamda / m) * np.sum(np.abs(weights))      # L1 penalty
    l2_term = (1 - alpha) * (lamda / (2 * m)) * np.sum(np.square(weights))  # L2 penalty
    # Add small value (1e-15) to avoid log(0)
    return - (1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)) + l1_term + l2_term

def train_elastic_logistic(X, y, lr=0.01, epochs=1000, lamda=0.1, alpha=0.5):
    """
    Train logistic regression model with Elastic Net regularization using gradient descent.
    X: feature matrix
    y: target vector
    lr: learning rate
    epochs: number of iterations
    lamda: regularization strength
    alpha: mixing parameter (0 = Ridge, 1 = Lasso)
    Returns: trained weights and bias
    """
    m, n = X.shape
    weights = np.zeros((n, 1))  # Initialize weights
    bias = 0  # Initialize bias

    for epoch in range(epochs):
        z = np.dot(X, weights) + bias  # Linear combination
        y_pred = sigmoid(z)  # Predicted probabilities
        # Elastic Net gradient: alpha for L1, (1-alpha) for L2
        dw = (1/m) * np.dot(X.T, (y_pred - y)) \
             + alpha * (lamda/m) * np.sign(weights) \
             + (1 - alpha) * (lamda/m) * weights
        db = (1/m) * np.sum(y_pred - y)  # Gradient for bias
        weights -= lr * dw  # Update weights
        bias -= lr * db     # Update bias

        if epoch % 100 == 0:
            loss = negative_log_likelihood(y, y_pred, weights, lamda, alpha)
            # print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, bias

def predict(X, weights, bias):
    """
    Predict class labels and probabilities for input data X.
    Returns: predicted class labels (0 or 1), predicted probabilities
    """
    y_pred_probs = sigmoid(np.dot(X, weights) + bias)
    return (y_pred_probs >= 0.5).astype(int), y_pred_probs