import numpy as np
import pandas as pd

class SimpleCatBoost:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.splits = []
        self.init_val = None

    def _target_encode(self, X, y):
        """
        Replace categorical columns with mean target encoding.
        """
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == "object" or X_encoded[col].dtype.name == "category":
                mapping = y.groupby(X_encoded[col]).mean()
                X_encoded[col] = X_encoded[col].map(mapping)
        return X_encoded.astype(float)

    def _build_oblivious_tree(self, X, residuals):
        """
        Build an oblivious tree (same feature and threshold for each level).
        """
        n_samples, n_features = X.shape
        best_feat, best_thresh, best_score = None, None, float("inf")

        for feat in range(n_features):
            thresholds = np.unique(X.iloc[:, feat])
            for t in thresholds:
                left_mask = X.iloc[:, feat] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                score = (
                    np.sum((residuals[left_mask] - residuals[left_mask].mean()) ** 2) +
                    np.sum((residuals[right_mask] - residuals[right_mask].mean()) ** 2)
                )
                if score < best_score:
                    best_feat, best_thresh, best_score = feat, t, score

        # For simplicity, we just store feature and threshold
        self.splits.append((best_feat, best_thresh))
        return best_feat, best_thresh

    def fit(self, X, y):
        X_encoded = self._target_encode(X, pd.Series(y))
        self.init_val = np.mean(y)
        y_pred = np.full(len(y), self.init_val)

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            feat, thresh = self._build_oblivious_tree(X_encoded, residuals)

            left_mask = X_encoded.iloc[:, feat] <= thresh
            right_mask = ~left_mask
            left_val = residuals[left_mask].mean()
            right_val = residuals[right_mask].mean()

            self.trees.append((feat, thresh, left_val, right_val))

            # Update predictions
            y_pred[left_mask] += self.learning_rate * left_val
            y_pred[right_mask] += self.learning_rate * right_val

    def predict(self, X):
        X_encoded = self._target_encode(X, pd.Series(np.zeros(len(X))))  # dummy y for encoding
        y_pred = np.full(len(X), self.init_val)
        for feat, thresh, left_val, right_val in self.trees:
            left_mask = X_encoded.iloc[:, feat] <= thresh
            right_mask = ~left_mask
            y_pred[left_mask] += self.learning_rate * left_val
            y_pred[right_mask] += self.learning_rate * right_val
        return np.round(y_pred)  # For classification

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

df = pd.DataFrame({
    "Feature1": [1.0, 1.5, 2.0, 7.0, 8.0, 9.0],
    "Feature2": ["low", "low", "low", "high", "high", "high"]  # categorical
})
y = np.array([0, 0, 0, 1, 1, 1])

model = SimpleCatBoost(n_estimators=10, learning_rate=0.3)
model.fit(df, y)

preds = model.predict(df)
print("Predictions:", preds)
print("Accuracy:", model.score(df, y))
