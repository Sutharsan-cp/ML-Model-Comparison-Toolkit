import numpy as np


class DecisionStumpRegressor:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.leftValue = None
        self.rightValue = None
        self.isFitted = False
    
    def findBestSplit(self, X, y, sampleWeight):
        bestError = float('inf')
        bestFeature = None
        bestThreshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
                    continue
                
                leftValue = np.average(y[leftMask], weights=sampleWeight[leftMask])
                rightValue = np.average(y[rightMask], weights=sampleWeight[rightMask])
                
                predictions = np.where(leftMask, leftValue, rightValue)
                error = np.average((y - predictions) ** 2, weights=sampleWeight)
                
                if error < bestError:
                    bestError = error
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestError
    
    def fit(self, X, y, sampleWeight):
        X = np.array(X)
        y = np.array(y)
        sampleWeight = np.array(sampleWeight)
        
        self.feature, self.threshold, error = self.findBestSplit(X, y, sampleWeight)
        
        if self.feature is not None:
            leftMask = X[:, self.feature] <= self.threshold
            self.leftValue = np.average(y[leftMask], weights=sampleWeight[leftMask])
            self.rightValue = np.average(y[~leftMask], weights=sampleWeight[~leftMask])
        else:
            self.leftValue = np.average(y, weights=sampleWeight)
            self.rightValue = self.leftValue
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        if self.feature is None:
            return np.full(X.shape[0], self.leftValue)
        
        return np.where(X[:, self.feature] <= self.threshold, self.leftValue, self.rightValue)


class AdaBoostRegressor:
    def __init__(self, nEstimators=50, learningRate=1.0, loss='linear', randomState=None):
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.loss = loss
        self.randomState = randomState
        
        self.estimators = []
        self.estimatorWeights = []
        self.estimatorErrors = []
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def computeError(self, yTrue, yPred, sampleWeight):
        absoluteErrors = np.abs(yTrue - yPred)
        
        if self.loss == 'linear':
            errors = absoluteErrors
        elif self.loss == 'square':
            errors = (yTrue - yPred) ** 2
        elif self.loss == 'exponential':
            errors = 1 - np.exp(-absoluteErrors)
        else:
            raise ValueError("Loss must be 'linear', 'square', or 'exponential'")
        
        return np.average(errors, weights=sampleWeight)
    
    def computeBeta(self, error):
        # Prevent division by zero or infinity
        return max(1e-10, min(error / (1 - error + 1e-10), 1e10))
    
    def updateWeights(self, yTrue, yPred, sampleWeight, beta):
        absoluteErrors = np.abs(yTrue - yPred)
        
        if self.loss == 'linear':
            multiplier = 1 + absoluteErrors
        elif self.loss == 'square':
            multiplier = 1 + absoluteErrors ** 2
        elif self.loss == 'exponential':
            multiplier = np.exp(absoluteErrors)
        
        multiplier = np.power(multiplier, beta)
        newWeights = sampleWeight * multiplier
        return newWeights / np.sum(newWeights)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples = X.shape[0]
        sampleWeight = np.ones(nSamples) / nSamples
        
        self.estimators = []
        self.estimatorWeights = []
        self.estimatorErrors = []
        
        for i in range(self.nEstimators):
            stump = DecisionStumpRegressor()
            stump.fit(X, y, sampleWeight)
            yPred = stump.predict(X)
            
            error = self.computeError(y, yPred, sampleWeight)
            
            # Prevent weak learner rejection too early
            if error >= 0.99:  # Only reject if truly random
                if len(self.estimators) == 0:
                    raise ValueError("Base estimator too weak")
                break
            
            beta = self.computeBeta(error) * self.learningRate
            
            self.estimators.append(stump)
            self.estimatorWeights.append(beta)
            self.estimatorErrors.append(error)
            
            if error < 1e-10:
                break
            
            sampleWeight = self.updateWeights(y, yPred, sampleWeight, beta)
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        predictions = np.zeros(X.shape[0])
        totalWeight = 0
        
        for stump, beta in zip(self.estimators, self.estimatorWeights):
            weight = np.log(1.0 / (beta + 1e-10)) if beta > 0 else 1.0
            predictions += weight * stump.predict(X)
            totalWeight += weight
        
        return predictions / totalWeight if totalWeight != 0 else predictions
    
    def stagedPredict(self, X):
        X = np.array(X)
        predictions = np.zeros(X.shape[0])
        totalWeight = 0
        
        for stump, beta in zip(self.estimators, self.estimatorWeights):
            weight = np.log(1.0 / (beta + 1e-10)) if beta > 0 else 1.0
            predictions += weight * stump.predict(X)
            totalWeight += weight
            yield predictions / totalWeight if totalWeight != 0 else predictions
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def getEstimatorWeights(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.estimatorWeights
    
    def getEstimatorErrors(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.estimatorErrors
    
    def getParams(self):
        return {
            'nEstimators': len(self.estimators),
            'learningRate': self.learningRate,
            'loss': self.loss,
            'estimatorWeights': self.estimatorWeights,
            'estimatorErrors': self.estimatorErrors
        }
