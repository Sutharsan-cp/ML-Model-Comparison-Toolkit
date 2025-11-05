import numpy as np

class XGBoostTree:
    def __init__(self, maxDepth=6, minChildWeight=1, gamma=0, lambdaReg=1, learningRate=0.3):
        self.maxDepth = maxDepth
        self.minChildWeight = minChildWeight
        self.gamma = gamma
        self.lambdaReg = lambdaReg
        self.learningRate = learningRate
        self.tree = None
    
    def computeGradients(self, y, pred):
        return pred - y
    
    def computeHessians(self, y, pred):
        return np.ones_like(y)
    
    def calculateGain(self, gLeft, gRight, hLeft, hRight):
        gain = 0.5 * (
            (gLeft**2 / (hLeft + self.lambdaReg)) +
            (gRight**2 / (hRight + self.lambdaReg)) -
            ((gLeft + gRight)**2 / (hLeft + hRight + self.lambdaReg))
        ) - self.gamma
        return gain
    
    def findBestSplit(self, X, g, h):
        bestGain = -float('inf')
        bestFeature = None
        bestThreshold = None
        
        nSamples, nFeatures = X.shape
        
        for feature in range(nFeatures):
            uniqueValues = np.unique(X[:, feature])
            
            for threshold in uniqueValues:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) < 1 or np.sum(rightMask) < 1:
                    continue
                
                gLeft = np.sum(g[leftMask])
                gRight = np.sum(g[rightMask])
                hLeft = np.sum(h[leftMask])
                hRight = np.sum(h[rightMask])
                
                if hLeft < self.minChildWeight or hRight < self.minChildWeight:
                    continue
                
                gain = self.calculateGain(gLeft, gRight, hLeft, hRight)
                
                if gain > bestGain:
                    bestGain = gain
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestGain
    
    def buildTree(self, X, g, h, depth=0):
        nSamples = X.shape[0]
        
        if depth >= self.maxDepth or nSamples < 2:
            leafValue = -np.sum(g) / (np.sum(h) + self.lambdaReg)
            return leafValue * self.learningRate
        
        feature, threshold, gain = self.findBestSplit(X, g, h)
        
        if feature is None or gain < 0:
            leafValue = -np.sum(g) / (np.sum(h) + self.lambdaReg)
            return leafValue * self.learningRate
        
        leftMask = X[:, feature] <= threshold
        rightMask = ~leftMask
        
        leftSubtree = self.buildTree(X[leftMask], g[leftMask], h[leftMask], depth + 1)
        rightSubtree = self.buildTree(X[rightMask], g[rightMask], h[rightMask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'gain': gain,
            'left': leftSubtree,
            'right': rightSubtree
        }
    
    def fit(self, X, g, h):
        X = np.array(X)
        g = np.array(g)
        h = np.array(h)
        self.tree = self.buildTree(X, g, h)
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

class XGBoostRegressor:
    def __init__(self, nEstimators=100, learningRate=0.1, maxDepth=6, minChildWeight=1, 
                 gamma=0, lambdaReg=1, subsample=1.0, colsampleBytree=1.0, randomState=None):
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.minChildWeight = minChildWeight
        self.gamma = gamma
        self.lambdaReg = lambdaReg
        self.subsample = subsample
        self.colsampleBytree = colsampleBytree
        self.randomState = randomState
        
        self.estimators = []
        self.featureImportance = None
        self.initialPrediction = None
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def computeGradients(self, y, pred):
        return pred - y
    
    def computeHessians(self, y, pred):
        return np.ones_like(y)
    
    def fit(self, X, y, evalSet=None, earlyStoppingRounds=None, verbose=False):
        X = np.array(X)
        y = np.array(y)
        
        nSamples, nFeatures = X.shape
        
        self.estimators = []
        self.trainLoss = []
        self.valLoss = [] if evalSet else None
        
        self.initialPrediction = np.mean(y)
        pred = np.full(nSamples, self.initialPrediction)
        
        bestValLoss = float('inf')
        noImprovementCount = 0
        
        for epoch in range(self.nEstimators):
            gradients = self.computeGradients(y, pred)
            hessians = self.computeHessians(y, pred)
            
            if self.subsample < 1.0:
                sampleSize = int(self.subsample * nSamples)
                indices = np.random.choice(nSamples, sampleSize, replace=False)
                XSample = X[indices]
                gSample = gradients[indices]
                hSample = hessians[indices]
            else:
                XSample = X
                gSample = gradients
                hSample = hessians
            
            if self.colsampleBytree < 1.0:
                featureSize = int(self.colsampleBytree * nFeatures)
                featureIndices = np.random.choice(nFeatures, featureSize, replace=False)
                XSample = XSample[:, featureIndices]
                featureMap = featureIndices
            else:
                featureMap = np.arange(nFeatures)
            
            tree = XGBoostTree(
                maxDepth=self.maxDepth,
                minChildWeight=self.minChildWeight,
                gamma=self.gamma,
                lambdaReg=self.lambdaReg,
                learningRate=self.learningRate
            )
            
            tree.fit(XSample, gSample, hSample)
            self.estimators.append((tree, featureMap))
            
            predUpdate = tree.predict(XSample if self.colsampleBytree < 1.0 else X)
            pred += predUpdate
            
            trainLoss = np.mean((y - pred) ** 2)
            self.trainLoss.append(trainLoss)
            
            if evalSet:
                XVal, yVal = evalSet
                valPred = self.predict(XVal)
                valLoss = np.mean((yVal - valPred) ** 2)
                self.valLoss.append(valLoss)
                
                if verbose:
                    print(f"Epoch {epoch+1}: Train Loss = {trainLoss:.4f}, Val Loss = {valLoss:.4f}")
                
                if earlyStoppingRounds:
                    if valLoss < bestValLoss:
                        bestValLoss = valLoss
                        noImprovementCount = 0
                    else:
                        noImprovementCount += 1
                    
                    if noImprovementCount >= earlyStoppingRounds:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}: Train Loss = {trainLoss:.4f}")
        
        self.computeFeatureImportance(nFeatures)
        self.isFitted = True
        return self
    
    def computeFeatureImportance(self, nFeatures):
        self.featureImportance = np.zeros(nFeatures)
        
        for tree, featureMap in self.estimators:
            self.traverseTree(tree.tree, featureMap)
        
        if np.sum(self.featureImportance) > 0:
            self.featureImportance /= np.sum(self.featureImportance)
    
    def traverseTree(self, node, featureMap):
        if isinstance(node, dict):
            mappedFeature = featureMap[node['feature']]
            self.featureImportance[mappedFeature] += node['gain']
            self.traverseTree(node['left'], featureMap)
            self.traverseTree(node['right'], featureMap)
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        pred = np.full(X.shape[0], self.initialPrediction)
        
        for tree, featureMap in self.estimators:
            if self.colsampleBytree < 1.0:
                XSubset = X[:, featureMap]
            else:
                XSubset = X
            
            pred += tree.predict(XSubset)
        
        return pred
    
    def stagedPredict(self, X):
        X = np.array(X)
        pred = np.full(X.shape[0], self.initialPrediction)
        
        for tree, featureMap in self.estimators:
            if self.colsampleBytree < 1.0:
                XSubset = X[:, featureMap]
            else:
                XSubset = X
            
            pred += tree.predict(XSubset)
            yield pred.copy()
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def getFeatureImportance(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.featureImportance
    
    def getLossHistory(self):
        return {
            'trainLoss': self.trainLoss,
            'valLoss': self.valLoss
        }
    
    def getParams(self):
        return {
            'nEstimators': len(self.estimators),
            'learningRate': self.learningRate,
            'maxDepth': self.maxDepth,
            'minChildWeight': self.minChildWeight,
            'gamma': self.gamma,
            'lambdaReg': self.lambdaReg,
            'subsample': self.subsample,
            'colsampleBytree': self.colsampleBytree,
            'initialPrediction': self.initialPrediction,
            'featureImportance': self.featureImportance.tolist() if self.featureImportance is not None else None,
            'finalTrainLoss': self.trainLoss[-1] if self.trainLoss else None,
            'finalValLoss': self.valLoss[-1] if self.valLoss else None
        }