import numpy as np
import pandas as pd

class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def best_split(self, X, y):
        best_feat, best_thresh, best_gini = None, None, float("inf")
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                gini_split = (
                    (left_mask.sum() / len(y)) * self.gini(y[left_mask]) +
                    (right_mask.sum() / len(y)) * self.gini(y[right_mask])
                )
                if gini_split < best_gini:
                    best_feat, best_thresh, best_gini = feat, t, gini_split
        return best_feat, best_thresh

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]
        if self.max_depth and depth >= self.max_depth:
            return np.bincount(y).argmax()

        feat, thresh = self.best_split(X, y)
        if feat is None:
            return np.bincount(y).argmax()

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        return {
            "feature": feat,
            "threshold": thresh,
            "left": self.build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        if sample[tree["feature"]] <= tree["threshold"]:
            return self.predict_sample(sample, tree["left"])
        else:
            return self.predict_sample(sample, tree["right"])

    def predict(self, X):
        return np.array([self.predict_sample(row, self.tree) for row in X])

X = np.array([
    [1, 2],
    [1, 3],
    [2, 3],
    [3, 4],
    [4, 4],
    [5, 5]
])
y = np.array([0, 0, 0, 1, 1, 1])

cart = CARTDecisionTree()
cart.fit(X, y)
print("CART Tree:", cart.tree)
print("CART Predictions:", cart.predict(X))
