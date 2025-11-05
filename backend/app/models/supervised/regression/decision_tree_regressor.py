import numpy as np

class DecisionTreeRegressor:
    def __init__(self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1, minImpurityDecrease=0.0):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.minImpurityDecrease = minImpurityDecrease
        self.tree = None
        self.isFitted = False
    
    def mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def bestSplit(self, X, y):
        bestMseReduction = -1
        bestFeature = None
        bestThreshold = None
        
        currentMse = self.mse(y)
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) < self.minSamplesLeaf or np.sum(rightMask) < self.minSamplesLeaf:
                    continue
                
                leftMse = self.mse(y[leftMask])
                rightMse = self.mse(y[rightMask])
                
                nLeft = np.sum(leftMask)
                nRight = np.sum(rightMask)
                nTotal = len(y)
                
                mseReduction = currentMse - (nLeft/nTotal * leftMse + nRight/nTotal * rightMse)
                
                if mseReduction > bestMseReduction and mseReduction >= self.minImpurityDecrease:
                    bestMseReduction = mseReduction
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestMseReduction
    
    def buildTree(self, X, y, depth=0):
        nSamples = X.shape[0]
        
        if (self.maxDepth is not None and depth >= self.maxDepth or 
            nSamples < self.minSamplesSplit or 
            len(np.unique(y)) == 1):
            return np.mean(y)
        
        feature, threshold, mseReduction = self.bestSplit(X, y)
        
        if feature is None:
            return np.mean(y)
        
        leftMask = X[:, feature] <= threshold
        rightMask = ~leftMask
        
        leftSubtree = self.buildTree(X[leftMask], y[leftMask], depth + 1)
        rightSubtree = self.buildTree(X[rightMask], y[rightMask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'mseReduction': mseReduction,
            'left': leftSubtree,
            'right': rightSubtree
        }
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self.buildTree(X, y)
        self.isFitted = True
        return self
    
    def predictSingle(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node['feature']] <= node['threshold']:
            return self.predictSingle(x, node['left'])
        else:
            return self.predictSingle(x, node['right'])
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return np.array([self.predictSingle(x, self.tree) for x in X])
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def getParams(self):
        return {
            'maxDepth': self.maxDepth,
            'minSamplesSplit': self.minSamplesSplit,
            'minSamplesLeaf': self.minSamplesLeaf,
            'minImpurityDecrease': self.minImpurityDecrease
        }