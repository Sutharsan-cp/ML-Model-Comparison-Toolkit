import numpy as np
import pandas as pd

class SimpleLightGBM:
    def __init__(self, n_estimators=50, learning_rate=0.1, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.models = []  # Each model = (feature_idx, threshold, left_val, right_val)

    def _best_split(self, X, residuals):
        """
        Find the best split for a single decision stump.
        """
        n_samples, n_features = X.shape
        best_feat, best_thresh, best_score = None, None, float("inf")
        best_left_val, best_right_val = None, None

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                left_val = np.mean(residuals[left_mask])
                right_val = np.mean(residuals[right_mask])

                # Mean squared error
                score = (
                    np.sum((residuals[left_mask] - left_val) ** 2) +
                    np.sum((residuals[right_mask] - right_val) ** 2)
                )

                if score < best_score:
                    best_feat = feat
                    best_thresh = t
                    best_score = score
                    best_left_val = left_val
                    best_right_val = right_val

        return best_feat, best_thresh, best_left_val, best_right_val

    def fit(self, X, y):
        # Initial prediction = mean of y
        self.initial_pred = np.mean(y)
        y_pred = np.full(y.shape, self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - y_pred

            feat, thresh, left_val, right_val = self._best_split(X, residuals)
            if feat is None:
                break

            # Store model parameters
            self.models.append((feat, thresh, left_val, right_val))

            # Update predictions
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            y_pred[left_mask] += self.learning_rate * left_val
            y_pred[right_mask] += self.learning_rate * right_val

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.initial_pred)
        for feat, thresh, left_val, right_val in self.models:
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            y_pred[left_mask] += self.learning_rate * left_val
            y_pred[right_mask] += self.learning_rate * right_val
        return np.round(y_pred)  # For classification

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

X = np.array([
    [1.0, 2.1],
    [1.5, 1.8],
    [2.0, 2.0],
    [7.0, 8.0],
    [8.0, 8.5],
    [9.0, 9.0]
])

y = np.array([0, 0, 0, 1, 1, 1])  # Binary classification

lgbm = SimpleLightGBM(n_estimators=10, learning_rate=0.3)
lgbm.fit(X, y)

preds = lgbm.predict(X)
print("Predictions:", preds)
print("Accuracy:", lgbm.score(X, y))