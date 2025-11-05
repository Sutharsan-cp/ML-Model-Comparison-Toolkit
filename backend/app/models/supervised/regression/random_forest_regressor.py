import numpy as np

class DecisionTreeRegressor:
    def __init__(self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1, maxFeatures=None):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.maxFeatures = maxFeatures
        self.tree = None
        self.nFeatures = None
    
    def mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def bestSplit(self, X, y):
        bestMseReduction = -1
        bestFeature = None
        bestThreshold = None
        
        currentMse = self.mse(y)
        nFeatures = X.shape[1] if self.maxFeatures is None else min(self.maxFeatures, X.shape[1])
        features = np.random.choice(X.shape[1], nFeatures, replace=False)
        
        for feature in features:
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
                
                if mseReduction > bestMseReduction:
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
        self.nFeatures = X.shape[1]
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

class RandomForestRegressor:
    def __init__(self, nEstimators=100, maxDepth=None, minSamplesSplit=2, 
                 minSamplesLeaf=1, maxFeatures='auto', bootstrap=True, randomState=None):
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.maxFeatures = maxFeatures
        self.bootstrap = bootstrap
        self.randomState = randomState
        
        self.estimators = []
        self.featureImportance = None
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples, nFeatures = X.shape
        
        if self.maxFeatures == 'auto':
            maxFeatures = int(np.sqrt(nFeatures))
        elif self.maxFeatures == 'sqrt':
            maxFeatures = int(np.sqrt(nFeatures))
        elif self.maxFeatures == 'log2':
            maxFeatures = int(np.log2(nFeatures))
        else:
            maxFeatures = self.maxFeatures
        
        self.estimators = []
        
        for i in range(self.nEstimators):
            if self.bootstrap:
                indices = np.random.choice(nSamples, nSamples, replace=True)
                XBoot = X[indices]
                yBoot = y[indices]
            else:
                XBoot = X
                yBoot = y
            
            tree = DecisionTreeRegressor(
                maxDepth=self.maxDepth,
                minSamplesSplit=self.minSamplesSplit,
                minSamplesLeaf=self.minSamplesLeaf,
                maxFeatures=maxFeatures
            )
            
            tree.fit(XBoot, yBoot)
            self.estimators.append(tree)
        
        self.computeFeatureImportance(nFeatures)
        self.isFitted = True
        return self
    
    def computeFeatureImportance(self, nFeatures):
        self.featureImportance = np.zeros(nFeatures)
        
        for tree in self.estimators:
            self.traverseTree(tree.tree, tree.nFeatures)
        
        if np.sum(self.featureImportance) > 0:
            self.featureImportance /= np.sum(self.featureImportance)
    
    def traverseTree(self, node, nFeatures):
        if isinstance(node, dict):
            if node['feature'] < nFeatures:
                self.featureImportance[node['feature']] += node['mseReduction']
            self.traverseTree(node['left'], nFeatures)
            self.traverseTree(node['right'], nFeatures)
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.mean(predictions, axis=0)
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def getFeatureImportance(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.featureImportance
    
    def getParams(self):
        return {
            'nEstimators': self.nEstimators,
            'maxDepth': self.maxDepth,
            'minSamplesSplit': self.minSamplesSplit,
            'minSamplesLeaf': self.minSamplesLeaf,
            'maxFeatures': self.maxFeatures,
            'bootstrap': self.bootstrap,
            'featureImportance': self.featureImportance.tolist() if self.featureImportance is not None else None
        }