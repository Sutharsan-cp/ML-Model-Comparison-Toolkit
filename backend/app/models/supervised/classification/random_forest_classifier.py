import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1, maxFeatures=None):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.maxFeatures = maxFeatures
        self.tree = None
        self.nFeatures = None
        self.featureImportance = None
    
    def giniImpurity(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def informationGain(self, y, yLeft, yRight):
        p = len(yLeft) / len(y)
        return self.giniImpurity(y) - p * self.giniImpurity(yLeft) - (1 - p) * self.giniImpurity(yRight)
    
    def bestSplit(self, X, y):
        bestGain = -1
        bestFeature = None
        bestThreshold = None
        
        nFeatures = X.shape[1] if self.maxFeatures is None else min(self.maxFeatures, X.shape[1])
        features = np.random.choice(X.shape[1], nFeatures, replace=False)
        
        for feature in features:
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
        nSamples, nFeatures = X.shape
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
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.nFeatures = X.shape[1]
        self.tree = self.buildTree(X, y)
        self.featureImportance = np.zeros(self.nFeatures)
        self.computeFeatureImportance(self.tree)
        return self
    
    def computeFeatureImportance(self, node):
        if isinstance(node, dict):
            self.featureImportance[node['feature']] += node['gain']
            self.computeFeatureImportance(node['left'])
            self.computeFeatureImportance(node['right'])
    
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
    
    def predictProbaSingle(self, x, node):
        if not isinstance(node, dict):
            return {node: 1.0}
        
        if x[node['feature']] <= node['threshold']:
            return self.predictProbaSingle(x, node['left'])
        else:
            return self.predictProbaSingle(x, node['right'])
    
    def predictProba(self, X):
        X = np.array(X)
        allClasses = sorted(list(set(self.predict(X))))
        probas = []
        
        for x in X:
            pred = self.predictProbaSingle(x, self.tree)
            proba = [pred.get(cls, 0.0) for cls in allClasses]
            probas.append(proba)
        
        return np.array(probas)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

class RandomForestClassifier:
    def __init__(self, nEstimators=100, maxDepth=None, minSamplesSplit=2, 
                 minSamplesLeaf=1, maxFeatures='sqrt', bootstrap=True, randomState=None):
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.maxFeatures = maxFeatures
        self.bootstrap = bootstrap
        self.randomState = randomState
        self.estimators = []
        self.classes = None
        self.featureImportance = None
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        nSamples, nFeatures = X.shape
        
        if self.maxFeatures == 'sqrt':
            maxFeatures = int(np.sqrt(nFeatures))
        elif self.maxFeatures == 'log2':
            maxFeatures = int(np.log2(nFeatures))
        else:
            maxFeatures = self.maxFeatures
        
        self.estimators = []
        self.oobScores = []
        
        for i in range(self.nEstimators):
            if self.bootstrap:
                indices = np.random.choice(nSamples, nSamples, replace=True)
                XBoot = X[indices]
                yBoot = y[indices]
            else:
                XBoot = X
                yBoot = y
            
            tree = DecisionTreeClassifier(
                maxDepth=self.maxDepth,
                minSamplesSplit=self.minSamplesSplit,
                minSamplesLeaf=self.minSamplesLeaf,
                maxFeatures=maxFeatures
            )
            
            tree.fit(XBoot, yBoot)
            self.estimators.append(tree)
            
            if self.bootstrap:
                oobMask = ~np.isin(np.arange(nSamples), indices)
                if np.any(oobMask):
                    oobScore = tree.score(X[oobMask], y[oobMask])
                    self.oobScores.append(oobScore)
        
        self.computeFeatureImportance()
        self.isFitted = True
        return self
    
    def computeFeatureImportance(self):
        if not self.estimators:
            return
        
        nFeatures = self.estimators[0].nFeatures
        self.featureImportance = np.zeros(nFeatures)
        
        for tree in self.estimators:
            if tree.featureImportance is not None:
                self.featureImportance += tree.featureImportance
        
        if np.sum(self.featureImportance) > 0:
            self.featureImportance /= np.sum(self.featureImportance)
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
    
    def predictProba(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        allProbas = []
        
        for tree in self.estimators:
            treeProbas = tree.predictProba(X)
            allProbas.append(treeProbas)
        
        allProbas = np.array(allProbas)
        return np.mean(allProbas, axis=0)
    
    def predictLogProba(self, X):
        probabilities = self.predictProba(X)
        return np.log(probabilities + 1e-8)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getFeatureImportance(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.featureImportance
    
    def getOOBScore(self):
        if not self.bootstrap:
            return None
        return np.mean(self.oobScores) if self.oobScores else None
    
    def getParams(self):
        return {
            'nEstimators': self.nEstimators,
            'maxDepth': self.maxDepth,
            'minSamplesSplit': self.minSamplesSplit,
            'minSamplesLeaf': self.minSamplesLeaf,
            'maxFeatures': self.maxFeatures,
            'bootstrap': self.bootstrap,
            'randomState': self.randomState,
            'nClasses': len(self.classes) if self.classes is not None else 0,
            'featureImportance': self.featureImportance.tolist() if self.featureImportance is not None else None,
            'oobScore': self.getOOBScore()
        }