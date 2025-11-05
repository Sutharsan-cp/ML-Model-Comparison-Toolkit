import numpy as np
from collections import Counter
from scipy.stats import chi2

class DecisionTreeCHAID:
    def __init__(self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1, alpha=0.05):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.alpha = alpha
        self.tree = None
        self.featureNames = None
    
    def chiSquareTest(self, featureValues, y):
        contingencyTable = []
        uniqueFeatures = np.unique(featureValues)
        uniqueClasses = np.unique(y)
        
        for featVal in uniqueFeatures:
            row = []
            for cls in uniqueClasses:
                count = np.sum((featureValues == featVal) & (y == cls))
                row.append(count)
            contingencyTable.append(row)
        
        contingencyTable = np.array(contingencyTable)
        expected = np.outer(contingencyTable.sum(axis=1), contingencyTable.sum(axis=0)) / contingencyTable.sum()
        
        chi2Stat = np.sum((contingencyTable - expected) ** 2 / (expected + 1e-8))
        dof = (len(uniqueFeatures) - 1) * (len(uniqueClasses) - 1)
        pValue = 1 - chi2.cdf(chi2Stat, dof)
        
        return chi2Stat, pValue
    
    def bestSplit(self, X, y):
        bestPValue = 1
        bestFeature = None
        bestThreshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) < self.minSamplesLeaf or np.sum(rightMask) < self.minSamplesLeaf:
                    continue
                
                splitFeature = np.where(leftMask, 'left', 'right')
                chi2Stat, pValue = self.chiSquareTest(splitFeature, y)
                
                if pValue < bestPValue and pValue < self.alpha:
                    bestPValue = pValue
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestPValue
    
    def buildTree(self, X, y, depth=0):
        nSamples = X.shape[0]
        nClasses = len(np.unique(y))
        
        if (self.maxDepth is not None and depth >= self.maxDepth or 
            nSamples < self.minSamplesSplit or 
            nClasses == 1):
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold, pValue = self.bestSplit(X, y)
        
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        leftMask = X[:, feature] <= threshold
        rightMask = ~leftMask
        
        leftSubtree = self.buildTree(X[leftMask], y[leftMask], depth + 1)
        rightSubtree = self.buildTree(X[rightMask], y[rightMask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'pValue': pValue,
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