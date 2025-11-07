import numpy as np
from collections import Counter

class DecisionTreeC45:
    def __init__(self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1, minGainRatio=0.01):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.minGainRatio = minGainRatio
        self.tree = None
        self.featureNames = None
    
    def entropy(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def splitInformation(self, splits):
        total = sum(len(split) for split in splits)
        if total == 0:
            return 0
        return -np.sum([(len(split) / total) * np.log2(len(split) / total) for split in splits if len(split) > 0])
    
    def gainRatio(self, y, splits):
        informationGain = self.entropy(y) - np.sum([(len(split) / len(y)) * self.entropy(split) for split in splits if len(split) > 0])
        splitInfo = self.splitInformation(splits)
        return informationGain / splitInfo if splitInfo > 0 else 0
    
    def bestSplit(self, X, y):
        bestGainRatio = -1
        bestFeature = None
        bestThreshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) < self.minSamplesLeaf or np.sum(rightMask) < self.minSamplesLeaf:
                    continue
                
                splits = [y[leftMask], y[rightMask]]
                gainRatio = self.gainRatio(y, splits)
                
                if gainRatio > bestGainRatio and gainRatio >= self.minGainRatio:
                    bestGainRatio = gainRatio
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestGainRatio
    
    def buildTree(self, X, y, depth=0):
        nSamples = X.shape[0]
        nClasses = len(np.unique(y))
        
        if (self.maxDepth is not None and depth >= self.maxDepth or 
            nSamples < self.minSamplesSplit or 
            nClasses == 1):
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold, gainRatio = self.bestSplit(X, y)
        
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        leftMask = X[:, feature] <= threshold
        rightMask = ~leftMask
        
        leftSubtree = self.buildTree(X[leftMask], y[leftMask], depth + 1)
        rightSubtree = self.buildTree(X[rightMask], y[rightMask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'gainRatio': gainRatio,
            'left': leftSubtree,
            'right': rightSubtree
        }
    
    def fit(self, X, y, featureNames=None):
        X = np.array(X)
        y = np.array(y)
        self.featureNames = featureNames
        self.tree = self.buildTree(X, y)
        return self
    
    def predictSingle(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node['feature']] <= node['threshold']:
            return self.predictSingle(x, node['left'])
        else:
            return self.predictSingle(x, node['right'])
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self.predictSingle(x, self.tree) for x in X])
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
