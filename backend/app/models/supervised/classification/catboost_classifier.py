import numpy as np
from collections import Counter

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

class CatBoostClassifier:
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
        self.classes = None
        self.bestIteration = 0
        self.isFitted = False
        self.catFeatureIndices = []
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def logLoss(self, yTrue, yPred):
        yPred = np.clip(yPred, 1e-8, 1 - 1e-8)
        return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))
    
    def calcGradients(self, y, pred):
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return pred - y
    
    def orderedBoosting(self, X, y, permutations):
        nSamples = X.shape[0]
        predictions = np.full(nSamples, 0.5)
        allGradients = []
        
        for i in range(self.iterations):
            gradients = self.calcGradients(y, predictions)
            
            if i < len(permutations):
                perm = permutations[i]
            else:
                perm = np.random.permutation(nSamples)
            
            treeGradients = np.zeros(nSamples)
            sampleWeight = np.ones(nSamples)
            
            for j in range(nSamples):
                idx = perm[j]
                treeGradients[idx] = gradients[idx]
                
                tree = CatBoostTree(
                    depth=self.depth,
                    learningRate=self.learningRate,
                    randomStrength=self.randomStrength,
                    borderCount=self.borderCount
                )
                
                tree.fit(X[:j+1], treeGradients[:j+1], sampleWeight[:j+1])
                update = tree.predict(X)[idx]
                predictions[idx] += update
            
            allGradients.append(gradients.copy())
            
            if self.verbose and i % 100 == 0:
                loss = self.logLoss(y, predictions)
                print(f"Iteration {i}: LogLoss = {loss:.4f}")
        
        return predictions, allGradients
    
    def fit(self, X, y, catFeatures=None, evalSet=None):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("CatBoostClassifier supports only binary classification")
        
        yBinary = np.where(y == self.classes[1], 1, 0)
        nSamples, nFeatures = X.shape
        
        if catFeatures is not None:
            self.catFeatureIndices = catFeatures
        else:
            self.catFeatureIndices = []
        
        self.estimators = []
        self.trainLoss = []
        self.valLoss = [] if evalSet else None
        
        pred = np.full(nSamples, 0.5)
        bestValLoss = float('inf')
        noImprovementCount = 0
        
        permutations = [np.random.permutation(nSamples) for _ in range(min(self.iterations, 10))]
        
        for i in range(self.iterations):
            gradients = self.calcGradients(yBinary, pred)
            
            if i < len(permutations):
                sampleIndices = permutations[i]
            else:
                sampleIndices = np.random.permutation(nSamples)
            
            sampleWeight = np.ones(nSamples)
            
            tree = CatBoostTree(
                depth=self.depth,
                learningRate=self.learningRate,
                randomStrength=self.randomStrength,
                borderCount=self.borderCount
            )
            
            tree.fit(X, gradients, sampleWeight)
            self.estimators.append(tree)
            
            update = tree.predict(X)
            pred += update
            
            trainLoss = self.logLoss(yBinary, self.sigmoid(pred))
            self.trainLoss.append(trainLoss)
            
            if evalSet:
                XVal, yVal = evalSet
                yValBinary = np.where(yVal == self.classes[1], 1, 0)
                valPred = self.predictProbaRaw(XVal)
                valLoss = self.logLoss(yValBinary, self.sigmoid(valPred))
                self.valLoss.append(valLoss)
                
                if self.verbose and i % 10 == 0:
                    print(f"Iteration {i+1}: Train Loss = {trainLoss:.4f}, Val Loss = {valLoss:.4f}")
                
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
    
    def predictProbaRaw(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        pred = np.full(X.shape[0], 0.5)
        
        if self.useBestModel:
            estimatorsToUse = self.estimators[:self.bestIteration + 1]
        else:
            estimatorsToUse = self.estimators
        
        for tree in estimatorsToUse:
            pred += tree.predict(X)
        
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
    
    def stagedPredictProba(self, X):
        X = np.array(X)
        pred = np.full(X.shape[0], 0.5)
        
        for tree in self.estimators:
            pred += tree.predict(X)
            yield self.sigmoid(pred)
    
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