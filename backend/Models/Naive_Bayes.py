import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.classes = None
        self.class_priors = {}
        self.mean_var = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            self.mean_var[c] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0) + self.alpha
            }

    def _gaussian_likelihood(self, class_idx, x):
        mean = self.mean_var[class_idx]["mean"]
        var = self.mean_var[class_idx]["var"]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                likelihood = np.sum(np.log(self._gaussian_likelihood(c, x)))
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def optimize_alpha(X, y, alpha_values):
    best_alpha = None
    best_score = -1
    for alpha in alpha_values:
        model = NaiveBayes(alpha=alpha)
        model.fit(X, y)
        score = model.score(X, y)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha, best_score

X = np.array([
    [1.0, 2.1],
    [1.5, 1.8],
    [2.0, 2.0],
    [7.0, 8.0],
    [8.0, 8.5],
    [9.0, 9.0]
])

# Assume first 3 points = Class 0, last 3 points = Class 1
y = np.array([0, 0, 0, 1, 1, 1])

# Optimize alpha
best_alpha, best_score = optimize_alpha(X, y, alpha_values=np.linspace(0.1, 2, 10))
print(f"Best alpha: {best_alpha}, Score: {best_score:.2f}")

# Train with best alpha
nb = NaiveBayes(alpha=best_alpha)
nb.fit(X, y)
preds = nb.predict(X)

print("Predictions:", preds)
print("Accuracy:", nb.score(X, y))
