import numpy as np
from collections import Counter

class DecisionTreeID3:
    def __init__(self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.tree = None
        self.featureNames = None
    
    def entropy(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def informationGain(self, y, yLeft, yRight):
        p = len(yLeft) / len(y)
        return self.entropy(y) - p * self.entropy(yLeft) - (1 - p) * self.entropy(yRight)
    
    def bestSplit(self, X, y):
        bestGain = -1
        bestFeature = None
        bestThreshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) < self.minSamplesLeaf or np.sum(rightMask) < self.minSamplesLeaf:
                    continue
                
                gain = self.informationGain(y, y[leftMask], y[rightMask])
                if gain > bestGain:
                    bestGain = gain
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestGain
    
    def buildTree(self, X, y, depth=0):
        nSamples = X.shape[0]
        nClasses = len(np.unique(y))
        
        if (self.maxDepth is not None and depth >= self.maxDepth or 
            nSamples < self.minSamplesSplit or 
            nClasses == 1):
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold, gain = self.bestSplit(X, y)
        
        if feature is None or gain == 0:
            return Counter(y).most_common(1)[0][0]
        
        leftMask = X[:, feature] <= threshold
        rightMask = ~leftMask
        
        leftSubtree = self.buildTree(X[leftMask], y[leftMask], depth + 1)
        rightSubtree = self.buildTree(X[rightMask], y[rightMask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'gain': gain,
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
    
    def predictProba(self, X):
        X = np.array(X)
        allClasses = sorted(list(set(self.predict(X))))
        probas = []
        
        for x in X:
            pred = self.predictSingle(x, self.tree)
            proba = [1.0 if cls == pred else 0.0 for cls in allClasses]
            probas.append(proba)
        
        return np.array(probas)
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
