import numpy as np
import pandas as pd

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def information_gain(self, X_col, y):
        parent_entropy = self.entropy(y)
        values, counts = np.unique(X_col, return_counts=True)
        weighted_entropy = np.sum([
            (counts[i] / len(y)) * self.entropy(y[X_col == v])
            for i, v in enumerate(values)
        ])
        return parent_entropy - weighted_entropy

    def best_split(self, X, y):
        best_feat, best_gain = None, -1
        for feat in range(X.shape[1]):
            gain = self.information_gain(X[:, feat], y)
            if gain > best_gain:
                best_feat, best_gain = feat, gain
        return best_feat

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]
        if self.max_depth and depth >= self.max_depth:
            return np.bincount(y).argmax()

        feat = self.best_split(X, y)
        if feat is None:
            return np.bincount(y).argmax()

        tree = {feat: {}}
        for value in np.unique(X[:, feat]):
            subset_X = X[X[:, feat] == value]
            subset_y = y[X[:, feat] == value]
            tree[feat][value] = self.build_tree(subset_X, subset_y, depth + 1)
        return tree

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        feat = list(tree.keys())[0]
        value = sample[feat]
        return self.predict_sample(sample, tree[feat].get(value, 0))

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

id3 = ID3DecisionTree()
id3.fit(X, y)
print("ID3 Tree:", id3.tree)
print("ID3 Predictions:", id3.predict(X))
