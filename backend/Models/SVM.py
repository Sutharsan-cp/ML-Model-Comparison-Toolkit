import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Ensure labels are -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Only regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Hinge loss + regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.where(y <= 0, -1, 1))

X = np.array([
    [1.0, 2.1],
    [1.5, 1.8],
    [2.0, 2.0],
    [7.0, 8.0],
    [8.0, 8.5],
    [9.0, 9.0]
])

# First 3 = Class 0, last 3 = Class 1 → Convert to -1 and 1
y = np.array([0, 0, 0, 1, 1, 1])

svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

predictions = svm.predict(X)
print("Predictions:", predictions)
print("Accuracy:", svm.score(X, y))
print("Weights:", svm.w)
print("Bias:", svm.b)
