import numpy as np
from collections import Counter

class GradientBoostingBaseTree:
    def __init__(self, maxDepth=3, minSamplesSplit=2, minSamplesLeaf=1, minImpurityDecrease=0.0):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.minImpurityDecrease = minImpurityDecrease
        self.tree = None
        self.isRegression = False
    
    def computeLeafValue(self, gradients, hessians, lambdaReg=1.0):
        if self.isRegression:
            return np.mean(gradients)
        else:
            numerator = -np.sum(gradients)
            denominator = np.sum(hessians) + lambdaReg
            return numerator / denominator if denominator != 0 else 0.0
    
    def mseSplit(self, gradients, leftGradients, rightGradients):
        totalVar = np.var(gradients)
        leftVar = np.var(leftGradients) if len(leftGradients) > 0 else 0
        rightVar = np.var(rightGradients) if len(rightGradients) > 0 else 0
        pLeft = len(leftGradients) / len(gradients)
        pRight = len(rightGradients) / len(gradients)
        return totalVar - pLeft * leftVar - pRight * rightVar
    
    def bestSplit(self, X, gradients, hessians):
        bestGain = -float('inf')
        bestFeature = None
        bestThreshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) < self.minSamplesLeaf or np.sum(rightMask) < self.minSamplesLeaf:
                    continue
                
                if self.isRegression:
                    gain = self.mseSplit(gradients, gradients[leftMask], gradients[rightMask])
                else:
                    gain = self.mseSplit(gradients, gradients[leftMask], gradients[rightMask])
                
                if gain > bestGain and gain >= self.minImpurityDecrease:
                    bestGain = gain
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestGain
    
    def buildTree(self, X, gradients, hessians, depth=0, lambdaReg=1.0):
        nSamples = X.shape[0]
        
        if (depth >= self.maxDepth or 
            nSamples < self.minSamplesSplit or 
            len(np.unique(gradients)) == 1):
            return self.computeLeafValue(gradients, hessians, lambdaReg)
        
        feature, threshold, gain = self.bestSplit(X, gradients, hessians)
        
        if feature is None:
            return self.computeLeafValue(gradients, hessians, lambdaReg)
        
        leftMask = X[:, feature] <= threshold
        rightMask = ~leftMask
        
        leftSubtree = self.buildTree(X[leftMask], gradients[leftMask], hessians[leftMask], depth + 1, lambdaReg)
        rightSubtree = self.buildTree(X[rightMask], gradients[rightMask], hessians[rightMask], depth + 1, lambdaReg)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'gain': gain,
            'left': leftSubtree,
            'right': rightSubtree
        }
    
    def fit(self, X, gradients, hessians, lambdaReg=1.0):
        X = np.array(X)
        gradients = np.array(gradients)
        hessians = np.array(hessians)
        self.tree = self.buildTree(X, gradients, hessians, lambdaReg=lambdaReg)
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

class GradientBoostingClassifier:
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
        self.trainScore = []
        self.validationScore = []
        self.classes = None
        self.initialPrediction = None
        self.isFitted = False
        self.bestIteration = 0
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def logLoss(self, yTrue, yPred):
        yPred = np.clip(yPred, 1e-8, 1 - 1e-8)
        return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))
    
    def computeGradients(self, y, pred):
        probabilities = self.sigmoid(pred)
        return probabilities - y
    
    def computeHessians(self, y, pred):
        probabilities = self.sigmoid(pred)
        return probabilities * (1 - probabilities)
    
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
        
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("GradientBoostingClassifier supports only binary classification")
        
        yBinary = np.where(y == self.classes[1], 1, 0)
        
        XTrain, yTrain, XVal, yVal = self.createValidationSet(X, yBinary, self.validationFraction)
        
        nSamples = XTrain.shape[0]
        self.initialPrediction = np.log(np.mean(yTrain) / (1 - np.mean(yTrain) + 1e-8))
        pred = np.full(nSamples, self.initialPrediction)
        
        self.estimators = []
        self.trainScore = []
        self.validationScore = []
        
        bestValLoss = float('inf')
        noImprovementCount = 0
        
        for i in range(self.nEstimators):
            gradients = self.computeGradients(yTrain, pred)
            hessians = self.computeHessians(yTrain, pred)
            
            if self.subsample < 1.0:
                sampleSize = int(self.subsample * nSamples)
                indices = np.random.choice(nSamples, sampleSize, replace=False)
                XSample = XTrain[indices]
                gSample = gradients[indices]
                hSample = hessians[indices]
            else:
                XSample = XTrain
                gSample = gradients
                hSample = hessians
            
            tree = GradientBoostingBaseTree(
                maxDepth=self.maxDepth,
                minSamplesSplit=self.minSamplesSplit,
                minSamplesLeaf=self.minSamplesLeaf,
                minImpurityDecrease=self.minImpurityDecrease
            )
            
            tree.fit(XSample, gSample, hSample)
            self.estimators.append(tree)
            
            update = tree.predict(XTrain)
            pred += self.learningRate * update
            
            trainLoss = self.logLoss(yTrain, self.sigmoid(pred))
            self.trainScore.append(trainLoss)
            
            if len(XVal) > 0:
                valPred = self.predictRaw(XVal)
                valLoss = self.logLoss(yVal, self.sigmoid(valPred))
                self.validationScore.append(valLoss)
                
                if self.nIterNoChange is not None:
                    if valLoss < bestValLoss - self.tol:
                        bestValLoss = valLoss
                        noImprovementCount = 0
                        self.bestIteration = i
                    else:
                        noImprovementCount += 1
                    
                    if noImprovementCount >= self.nIterNoChange:
                        print(f"Early stopping at iteration {i+1}")
                        break
        
        # ✅ Ensure model is marked as fitted (fix)
        self.isFitted = True
        return self
    
    def predictRaw(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        pred = np.full(X.shape[0], self.initialPrediction)
        
        for tree in self.estimators:
            pred += self.learningRate * tree.predict(X)
        
        return pred
    
    def predictProba(self, X):
        rawPred = self.predictRaw(X)
        probabilities = self.sigmoid(rawPred)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X):
        probabilities = self.predictProba(X)
        return self.classes[(probabilities[:, 1] > 0.5).astype(int)]
    
    def predictLogProba(self, X):
        probabilities = self.predictProba(X)
        return np.log(probabilities + 1e-8)
    
    def stagedPredictProba(self, X):
        X = np.array(X)
        pred = np.full(X.shape[0], self.initialPrediction)
        
        for tree in self.estimators:
            pred += self.learningRate * tree.predict(X)
            yield self.sigmoid(pred)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getFeatureImportance(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        
        nFeatures = self.estimators[0].tree['feature'] if self.estimators else 0
        if isinstance(nFeatures, int):
            nFeatures = max(tree.tree['feature'] for tree in self.estimators if isinstance(tree.tree, dict)) + 1
        else:
            nFeatures = 1
        
        importance = np.zeros(nFeatures)
        
        for tree in self.estimators:
            self.accumulateImportance(tree.tree, importance)
        
        if np.sum(importance) > 0:
            importance /= np.sum(importance)
        
        return importance
    
    def accumulateImportance(self, node, importance):
        if isinstance(node, dict):
            importance[node['feature']] += node['gain']
            self.accumulateImportance(node['left'], importance)
            self.accumulateImportance(node['right'], importance)
    
    def getLossHistory(self):
        return {
            'trainScore': self.trainScore,
            'validationScore': self.validationScore
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
            'finalTrainLoss': self.trainScore[-1] if self.trainScore else None,
            'finalValidationLoss': self.validationScore[-1] if self.validationScore else None
        }
