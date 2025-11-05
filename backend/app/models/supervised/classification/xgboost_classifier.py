import numpy as np
from collections import Counter

class XGBoostTree:
    def __init__(self, maxDepth=6, minChildWeight=1, gamma=0, lambdaReg=1, learningRate=0.3):
        self.maxDepth = maxDepth
        self.minChildWeight = minChildWeight
        self.gamma = gamma
        self.lambdaReg = lambdaReg
        self.learningRate = learningRate
        self.tree = None
    
    def computeGradients(self, y, pred):
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return pred - y
    
    def computeHessians(self, y, pred):
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return pred * (1 - pred)
    
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

class XGBoostClassifier:
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
        self.classes = None
        self.baseScore = 0.5
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def logLoss(self, yTrue, yPred):
        yPred = np.clip(yPred, 1e-8, 1 - 1e-8)
        return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))
    
    def fit(self, X, y, evalSet=None, earlyStoppingRounds=None, verbose=False):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("XGBoostClassifier supports only binary classification")
        
        yBinary = np.where(y == self.classes[1], 1, 0)
        nSamples, nFeatures = X.shape
        
        self.estimators = []
        self.trainLoss = []
        self.valLoss = [] if evalSet else None
        
        pred = np.full(nSamples, self.baseScore)
        bestValLoss = float('inf')
        noImprovementCount = 0
        
        for epoch in range(self.nEstimators):
            gradients = self.computeGradients(yBinary, pred)
            hessians = self.computeHessians(yBinary, pred)
            
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
            
            trainLoss = self.logLoss(yBinary, self.sigmoid(pred))
            self.trainLoss.append(trainLoss)
            
            if evalSet:
                XVal, yVal = evalSet
                yValBinary = np.where(yVal == self.classes[1], 1, 0)
                valPred = self.predictProbaRaw(XVal)
                valLoss = self.logLoss(yValBinary, self.sigmoid(valPred))
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
    
    def computeGradients(self, y, pred):
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return pred - y
    
    def computeHessians(self, y, pred):
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return pred * (1 - pred)
    
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
    
    def predictProbaRaw(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        pred = np.full(X.shape[0], self.baseScore)
        
        for tree, featureMap in self.estimators:
            if self.colsampleBytree < 1.0:
                XSubset = X[:, featureMap]
            else:
                XSubset = X
            
            pred += tree.predict(XSubset)
        
        return pred
    
    def predictProba(self, X):
        rawPred = self.predictProbaRaw(X)
        probabilities = self.sigmoid(rawPred)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X):
        probabilities = self.predictProba(X)
        return self.classes[(probabilities[:, 1] > 0.5).astype(int)]
    
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
            'baseScore': self.baseScore,
            'featureImportance': self.featureImportance.tolist() if self.featureImportance is not None else None,
            'finalTrainLoss': self.trainLoss[-1] if self.trainLoss else None,
            'finalValLoss': self.valLoss[-1] if self.valLoss else None
        }