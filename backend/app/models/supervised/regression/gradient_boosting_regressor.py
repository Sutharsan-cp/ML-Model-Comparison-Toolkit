import numpy as np

class GradientBoostingBaseTree:
    def __init__(self, maxDepth=3, minSamplesSplit=2, minSamplesLeaf=1, minImpurityDecrease=0.0):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.minImpurityDecrease = minImpurityDecrease
        self.tree = None
    
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
        
        if (depth >= self.maxDepth or 
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

class GradientBoostingRegressor:
    def __init__(self, nEstimators=100, learningRate=0.1, maxDepth=3, minSamplesSplit=2, 
                 minSamplesLeaf=1, subsample=1.0, minImpurityDecrease=0.0, 
                 validationFraction=0.1, nIterNoChange=None, tol=1e-4, randomState=None):
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.subsample = subsample
        self.minImpurityDecrease = minImpurityDecrease
        self.validationFraction = validationFraction
        self.nIterNoChange = nIterNoChange
        self.tol = tol
        self.randomState = randomState
        
        self.estimators = []
        self.trainLoss = []
        self.validationLoss = []
        self.initialPrediction = None
        self.isFitted = False
        self.bestIteration = 0
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def createValidationSet(self, X, y, validationFraction):
        if validationFraction <= 0:
            return X, y, X, y
        
        nSamples = len(X)
        nValidation = int(validationFraction * nSamples)
        indices = np.random.permutation(nSamples)
        
        trainIndices = indices[nValidation:]
        valIndices = indices[:nValidation]
        
        XTrain = X[trainIndices]
        yTrain = y[trainIndices]
        XVal = X[valIndices]
        yVal = y[valIndices]
        
        return XTrain, yTrain, XVal, yVal
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        XTrain, yTrain, XVal, yVal = self.createValidationSet(X, y, self.validationFraction)
        
        nSamples = XTrain.shape[0]
        self.initialPrediction = np.mean(yTrain)
        pred = np.full(nSamples, self.initialPrediction)
        
        self.estimators = []
        self.trainLoss = []
        self.validationLoss = []
        
        bestValLoss = float('inf')
        noImprovementCount = 0
        
        for i in range(self.nEstimators):
            residuals = yTrain - pred
            
            if self.subsample < 1.0:
                sampleSize = int(self.subsample * nSamples)
                indices = np.random.choice(nSamples, sampleSize, replace=False)
                XSample = XTrain[indices]
                rSample = residuals[indices]
            else:
                XSample = XTrain
                rSample = residuals
            
            tree = GradientBoostingBaseTree(
                maxDepth=self.maxDepth,
                minSamplesSplit=self.minSamplesSplit,
                minSamplesLeaf=self.minSamplesLeaf,
                minImpurityDecrease=self.minImpurityDecrease
            )
            
            tree.fit(XSample, rSample)
            self.estimators.append(tree)
            
            update = tree.predict(XTrain)
            pred += self.learningRate * update
            
            trainMse = np.mean((yTrain - pred) ** 2)
            self.trainLoss.append(trainMse)
            
            if len(XVal) > 0:
                valPred = self.predict(XVal)
                valMse = np.mean((yVal - valPred) ** 2)
                self.validationLoss.append(valMse)
                
                if self.nIterNoChange is not None:
                    if valMse < bestValLoss - self.tol:
                        bestValLoss = valMse
                        noImprovementCount = 0
                        self.bestIteration = i
                    else:
                        noImprovementCount += 1
                    
                    if noImprovementCount >= self.nIterNoChange:
                        break
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        pred = np.full(X.shape[0], self.initialPrediction)
        
        for tree in self.estimators:
            pred += self.learningRate * tree.predict(X)
        
        return pred
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def stagedPredict(self, X):
        X = np.array(X)
        pred = np.full(X.shape[0], self.initialPrediction)
        
        for tree in self.estimators:
            pred += self.learningRate * tree.predict(X)
            yield pred.copy()
    
    def getLossHistory(self):
        return {
            'trainLoss': self.trainLoss,
            'validationLoss': self.validationLoss
        }
    
    def getParams(self):
        return {
            'nEstimators': len(self.estimators),
            'learningRate': self.learningRate,
            'maxDepth': self.maxDepth,
            'minSamplesSplit': self.minSamplesSplit,
            'minSamplesLeaf': self.minSamplesLeaf,
            'subsample': self.subsample,
            'initialPrediction': self.initialPrediction,
            'bestIteration': self.bestIteration,
            'finalTrainLoss': self.trainLoss[-1] if self.trainLoss else None,
            'finalValidationLoss': self.validationLoss[-1] if self.validationLoss else None
        }