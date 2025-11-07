import numpy as np

class CatBoostTree:
    def __init__(self, depth=6, learningRate=0.03, randomStrength=1, borderCount=32):
        self.depth = depth
        self.learningRate = learningRate
        self.randomStrength = randomStrength
        self.borderCount = borderCount
        self.tree = None
        self.borders = {}
    
    def computeObliviousSplit(self, X, gradients, sampleWeight):
        nSamples, nFeatures = X.shape
        bestFeature = None
        bestBorder = None
        bestScore = -float('inf')
        
        for feature in range(nFeatures):
            if feature not in self.borders:
                self.borders[feature] = self.calcBorders(X[:, feature])
            
            for border in self.borders[feature]:
                leftMask = X[:, feature] <= border
                rightMask = ~leftMask
                
                if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
                    continue
                
                leftGrad = np.sum(gradients[leftMask] * sampleWeight[leftMask])
                leftHess = np.sum(sampleWeight[leftMask])
                rightGrad = np.sum(gradients[rightMask] * sampleWeight[rightMask])
                rightHess = np.sum(sampleWeight[rightMask])
                
                score = (leftGrad**2 / (leftHess + 1)) + (rightGrad**2 / (rightHess + 1))
                
                if score > bestScore:
                    bestScore = score
                    bestFeature = feature
                    bestBorder = border
        
        return bestFeature, bestBorder
    
    def calcBorders(self, featureValues):
        uniqueVals = np.unique(featureValues)
        if len(uniqueVals) <= self.borderCount:
            return uniqueVals[:-1]
        
        percentiles = np.linspace(0, 100, self.borderCount + 2)[1:-1]
        borders = np.percentile(featureValues, percentiles)
        return np.unique(borders)
    
    def buildObliviousTree(self, X, gradients, sampleWeight, depth=0):
        if depth >= self.depth:
            leafValue = -np.sum(gradients * sampleWeight) / (np.sum(sampleWeight) + 1)
            return leafValue * self.learningRate
        
        feature, border = self.computeObliviousSplit(X, gradients, sampleWeight)
        
        if feature is None:
            leafValue = -np.sum(gradients * sampleWeight) / (np.sum(sampleWeight) + 1)
            return leafValue * self.learningRate
        
        leftMask = X[:, feature] <= border
        rightMask = ~leftMask
        
        leftSubtree = self.buildObliviousTree(X, gradients, sampleWeight, depth + 1)
        rightSubtree = self.buildObliviousTree(X, gradients, sampleWeight, depth + 1)
        
        return {
            'feature': feature,
            'border': border,
            'left': leftSubtree,
            'right': rightSubtree
        }
    
    def fit(self, X, gradients, sampleWeight):
        X = np.array(X)
        gradients = np.array(gradients)
        sampleWeight = np.array(sampleWeight)
        self.tree = self.buildObliviousTree(X, gradients, sampleWeight)
        return self
    
    def predictSingle(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node['feature']] <= node['border']:
            return self.predictSingle(x, node['left'])
        else:
            return self.predictSingle(x, node['right'])
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self.predictSingle(x, self.tree) for x in X])

class CatBoostRegressor:
    def __init__(self, iterations=1000, learningRate=0.03, depth=6, l2LeafReg=3, 
                 borderCount=32, randomStrength=1, useBestModel=False, 
                 earlyStoppingRounds=None, verbose=False, randomState=None):
        self.iterations = iterations
        self.learningRate = learningRate
        self.depth = depth
        self.l2LeafReg = l2LeafReg
        self.borderCount = borderCount
        self.randomStrength = randomStrength
        self.useBestModel = useBestModel
        self.earlyStoppingRounds = earlyStoppingRounds
        self.verbose = verbose
        self.randomState = randomState
        
        self.estimators = []
        self.featureImportance = None
        self.bestIteration = 0
        self.isFitted = False
        self.catFeatureIndices = []
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def computeGradients(self, y, pred):
        return pred - y
    
    def computeHessians(self, y, pred):
        return np.ones_like(y)
    
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
    
    def fit(self, X, y, catFeatures=None, evalSet=None):
        X = np.array(X)
        y = np.array(y)
        
        if evalSet is None:
            XTrain, yTrain, XVal, yVal = self.createValidationSet(X, y, 0.1)
        else:
            XTrain, yTrain = X, y
            XVal, yVal = evalSet
        
        nSamples, nFeatures = XTrain.shape
        
        if catFeatures is not None:
            self.catFeatureIndices = catFeatures
        else:
            self.catFeatureIndices = []
        
        self.estimators = []
        self.trainLoss = []
        self.valLoss = []
        
        pred = np.full(nSamples, np.mean(yTrain))
        bestValLoss = float('inf')
        noImprovementCount = 0
        
        for i in range(self.iterations):
            gradients = self.computeGradients(yTrain, pred)
            hessians = self.computeHessians(yTrain, pred)
            
            sampleWeight = np.ones(nSamples)
            
            tree = CatBoostTree(
                depth=self.depth,
                learningRate=self.learningRate,
                randomStrength=self.randomStrength,
                borderCount=self.borderCount
            )
            
            tree.fit(XTrain, gradients, sampleWeight)
            self.estimators.append(tree)
            
            update = tree.predict(XTrain)
            pred += update
            
            trainLoss = np.mean((yTrain - pred) ** 2)
            self.trainLoss.append(trainLoss)
            
            if len(XVal) > 0:
                valPred = self.predict(XVal)
                valLoss = np.mean((yVal - valPred) ** 2)
                self.valLoss.append(valLoss)
                
                if self.verbose and i % 100 == 0:
                    print(f"Iteration {i}: Train Loss = {trainLoss:.4f}, Val Loss = {valLoss:.4f}")
                
                if self.useBestModel and valLoss < bestValLoss:
                    bestValLoss = valLoss
                    self.bestIteration = i
                    noImprovementCount = 0
                else:
                    noImprovementCount += 1
                
                if self.earlyStoppingRounds and noImprovementCount >= self.earlyStoppingRounds:
                    if self.verbose:
                        print(f"Early stopping at iteration {i+1}")
                    break
            else:
                if self.verbose and i % 100 == 0:
                    print(f"Iteration {i+1}: Train Loss = {trainLoss:.4f}")
        
        self.computeFeatureImportance(nFeatures)
        # ✅ FIX: ensure model marked as fitted
        self.isFitted = True
        return self
    
    def computeFeatureImportance(self, nFeatures):
        self.featureImportance = np.zeros(nFeatures)
        for tree in self.estimators:
            self.traverseTreeImportance(tree.tree)
        if np.sum(self.featureImportance) > 0:
            self.featureImportance /= np.sum(self.featureImportance)
    
    def traverseTreeImportance(self, node):
        if isinstance(node, dict):
            self.featureImportance[node['feature']] += 1
            self.traverseTreeImportance(node['left'])
            self.traverseTreeImportance(node['right'])
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        pred = np.full(X.shape[0], 0.0)
        
        if self.useBestModel:
            estimatorsToUse = self.estimators[:self.bestIteration + 1]
        else:
            estimatorsToUse = self.estimators
        
        for tree in estimatorsToUse:
            pred += tree.predict(X)
        
        return pred
    
    def stagedPredict(self, X):
        X = np.array(X)
        pred = np.zeros(X.shape[0])
        for tree in self.estimators:
            pred += tree.predict(X)
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
    
    def getBestIteration(self):
        return self.bestIteration
    
    def getLossHistory(self):
        return {
            'trainLoss': self.trainLoss,
            'valLoss': self.valLoss
        }
    
    def getParams(self):
        return {
            'iterations': len(self.estimators),
            'learningRate': self.learningRate,
            'depth': self.depth,
            'l2LeafReg': self.l2LeafReg,
            'borderCount': self.borderCount,
            'randomStrength': self.randomStrength,
            'useBestModel': self.useBestModel,
            'bestIteration': self.bestIteration,
            'catFeatureIndices': self.catFeatureIndices,
            'featureImportance': self.featureImportance.tolist() if self.featureImportance is not None else None,
            'finalTrainLoss': self.trainLoss[-1] if self.trainLoss else None,
            'finalValLoss': self.valLoss[-1] if self.valLoss else None
        }
