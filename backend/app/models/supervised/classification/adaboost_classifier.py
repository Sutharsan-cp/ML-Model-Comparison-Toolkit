import numpy as np
from collections import Counter
from .decision_stump import DecisionStump


class AdaBoostClassifier:
    def __init__(self, nEstimators=50, learningRate=1.0, algorithm='SAMME', randomState=None):
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.algorithm = algorithm
        self.randomState = randomState
        
        self.estimators = []
        self.estimatorWeights = []
        self.estimatorErrors = []
        self.classes = None
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def computeError(self, yTrue, yPred, sampleWeight):
        incorrect = yTrue != yPred
        return np.sum(sampleWeight * incorrect) / np.sum(sampleWeight)
    
    def computeAlpha(self, error):
        if error >= 1 - 1e-10:
            return 0
        elif error <= 1e-10:
            return 1e10
        
        if self.algorithm == 'SAMME':
            return np.log((1 - error) / error)
        else:  # SAMME.R
            return np.log((1 - error) / error) + np.log(len(self.classes) - 1)
    
    def updateWeights(self, yTrue, yPred, sampleWeight, alpha):
        incorrect = yTrue != yPred
        
        if self.algorithm == 'SAMME':
            multiplier = np.exp(alpha * incorrect)
        else:  # SAMME.R
            multiplier = np.exp(-alpha * (2 * (yTrue == yPred) - 1))
        
        newWeights = sampleWeight * multiplier
        return newWeights / np.sum(newWeights)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        nClasses = len(self.classes)
        
        if nClasses < 2:
            raise ValueError("AdaBoost requires at least 2 classes")
        
        nSamples = X.shape[0]
        sampleWeight = np.ones(nSamples) / nSamples
        
        self.estimators = []
        self.estimatorWeights = []
        self.estimatorErrors = []
        
        for i in range(self.nEstimators):
            if self.algorithm == 'SAMME.R':
                stump = DecisionStump(criterion='entropy')
            else:
                stump = DecisionStump(criterion='gini')
            
            stump.fit(X, y)
            yPred = stump.predict(X)
            
            error = self.computeError(y, yPred, sampleWeight)
            
            if error > 0.5:
                if len(self.estimators) == 0:
                    raise ValueError("Base estimator too weak")
                break
            
            if error < 1e-10:
                alpha = 1e10
            else:
                alpha = self.computeAlpha(error)
            
            alpha *= self.learningRate
            
            self.estimators.append(stump)
            self.estimatorWeights.append(alpha)
            self.estimatorErrors.append(error)
            
            if error < 1e-10:
                break
            
            sampleWeight = self.updateWeights(y, yPred, sampleWeight, alpha)
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        
        if self.algorithm == 'SAMME.R':
            return self.predictSAMMER(X)
        else:
            return self.predictSAMME(X)
    
    def predictSAMME(self, X):
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        classScores = np.zeros((nSamples, nClasses))
        
        for stump, alpha in zip(self.estimators, self.estimatorWeights):
            predictions = stump.predict(X)
            
            for i, cls in enumerate(self.classes):
                mask = predictions == cls
                classScores[mask, i] += alpha
        
        return self.classes[np.argmax(classScores, axis=1)]
    
    def predictSAMMER(self, X):
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        classScores = np.zeros((nSamples, nClasses))
        
        for stump, alpha in zip(self.estimators, self.estimatorWeights):
            predictions = stump.predict(X)
            
            for i, cls in enumerate(self.classes):
                correct = predictions == cls
                classScores[correct, i] += alpha
        
        return self.classes[np.argmax(classScores, axis=1)]
    
    def predictProba(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        if self.algorithm == 'SAMME.R':
            return self.predictProbaSAMMER(X)
        else:
            return self.predictProbaSAMME(X)
    
    def predictProbaSAMME(self, X):
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        classScores = np.zeros((nSamples, nClasses))
        
        for stump, alpha in zip(self.estimators, self.estimatorWeights):
            predictions = stump.predict(X)
            
            for i, cls in enumerate(self.classes):
                mask = predictions == cls
                classScores[mask, i] += alpha
        
        expScores = np.exp(classScores - np.max(classScores, axis=1, keepdims=True))
        return expScores / np.sum(expScores, axis=1, keepdims=True)
    
    def predictProbaSAMMER(self, X):
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        classScores = np.zeros((nSamples, nClasses))
        
        for stump, alpha in zip(self.estimators, self.estimatorWeights):
            predictions = stump.predict(X)
            
            for i, cls in enumerate(self.classes):
                correct = predictions == cls
                classScores[correct, i] += alpha
        
        expScores = np.exp(classScores - np.max(classScores, axis=1, keepdims=True))
        return expScores / np.sum(expScores, axis=1, keepdims=True)
    
    def predictLogProba(self, X):
        probabilities = self.predictProba(X)
        return np.log(probabilities + 1e-8)
    
    def stagedPredict(self, X):
        X = np.array(X)
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        predictions = np.zeros((len(self.estimators), nSamples), dtype=self.classes.dtype)
        
        for i, (stump, alpha) in enumerate(zip(self.estimators, self.estimatorWeights)):
            if self.algorithm == 'SAMME.R':
                classScores = np.zeros((nSamples, nClasses))
                
                for j, (stump_j, alpha_j) in enumerate(zip(self.estimators[:i+1], self.estimatorWeights[:i+1])):
                    pred_j = stump_j.predict(X)
                    for k, cls in enumerate(self.classes):
                        correct = pred_j == cls
                        classScores[correct, k] += alpha_j
                
                predictions[i] = self.classes[np.argmax(classScores, axis=1)]
            else:
                classScores = np.zeros((nSamples, nClasses))
                
                for j, (stump_j, alpha_j) in enumerate(zip(self.estimators[:i+1], self.estimatorWeights[:i+1])):
                    pred_j = stump_j.predict(X)
                    for k, cls in enumerate(self.classes):
                        mask = pred_j == cls
                        classScores[mask, k] += alpha_j
                
                predictions[i] = self.classes[np.argmax(classScores, axis=1)]
            
            yield predictions[i]
    
    def stagedPredictProba(self, X):
        X = np.array(X)
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        for i in range(len(self.estimators)):
            if self.algorithm == 'SAMME.R':
                classScores = np.zeros((nSamples, nClasses))
                
                for j, (stump, alpha) in enumerate(zip(self.estimators[:i+1], self.estimatorWeights[:i+1])):
                    pred = stump.predict(X)
                    for k, cls in enumerate(self.classes):
                        correct = pred == cls
                        classScores[correct, k] += alpha
            else:
                classScores = np.zeros((nSamples, nClasses))
                
                for j, (stump, alpha) in enumerate(zip(self.estimators[:i+1], self.estimatorWeights[:i+1])):
                    pred = stump.predict(X)
                    for k, cls in enumerate(self.classes):
                        mask = pred == cls
                        classScores[mask, k] += alpha
            
            expScores = np.exp(classScores - np.max(classScores, axis=1, keepdims=True))
            probabilities = expScores / np.sum(expScores, axis=1, keepdims=True)
            yield probabilities
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getEstimatorWeights(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.estimatorWeights
    
    def getEstimatorErrors(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.estimatorErrors
    
    def getFeatureImportance(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        
        nFeatures = self.estimators[0].feature + 1 if self.estimators else 0
        importance = np.zeros(nFeatures)
        
        for stump, weight in zip(self.estimators, self.estimatorWeights):
            if stump.feature is not None:
                importance[stump.feature] += weight
        
        if np.sum(importance) > 0:
            importance /= np.sum(importance)
        
        return importance
    
    def getParams(self):
        return {
            'nEstimators': len(self.estimators),
            'learningRate': self.learningRate,
            'algorithm': self.algorithm,
            'estimatorWeights': self.estimatorWeights,
            'estimatorErrors': self.estimatorErrors,
            'nClasses': len(self.classes) if self.classes is not None else 0,
            'featureImportance': self.getFeatureImportance().tolist() if self.isFitted else None
        }