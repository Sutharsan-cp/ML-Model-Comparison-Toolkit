import numpy as np
from collections import Counter

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0, fitPrior=True, classPrior=None):
        self.alpha = alpha
        self.fitPrior = fitPrior
        self.classPrior = classPrior
        
        self.classes = None
        self.classCounts = None
        self.featureCounts = None
        self.classLogPrior = None
        self.featureLogProb = None
        self.isFitted = False
    
    def computeClassPrior(self, y):
        if self.classPrior is not None:
            return np.log(self.classPrior)
        
        if self.fitPrior:
            classCounts = self.classCounts
            return np.log(classCounts) - np.log(np.sum(classCounts))
        else:
            nClasses = len(self.classes)
            return np.full(nClasses, -np.log(nClasses))
    
    def computeFeatureLikelihood(self, X, y):
        nSamples, nFeatures = X.shape
        nClasses = len(self.classes)
        
        featureCounts = np.zeros((nClasses, nFeatures))
        
        for i, cls in enumerate(self.classes):
            classMask = y == cls
            if np.any(classMask):
                featureCounts[i] = np.sum(X[classMask], axis=0)
        
        # Add Laplace smoothing
        featureCounts += self.alpha
        
        # Normalize to get probabilities
        classTotals = featureCounts.sum(axis=1, keepdims=True)
        featureProb = featureCounts / classTotals
        
        return np.log(featureProb)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # --- PATCH START ---
        # Auto-handle negative values safely (instead of raising error)
        if np.any(X < 0):
            X = np.clip(X, 0, None)
        # --- PATCH END ---
        
        self.classes = np.unique(y)
        nClasses = len(self.classes)
        
        # Count class occurrences
        self.classCounts = np.zeros(nClasses)
        for i, cls in enumerate(self.classes):
            self.classCounts[i] = np.sum(y == cls)
        
        # Compute class priors
        self.classLogPrior = self.computeClassPrior(y)
        
        # Compute feature likelihoods
        self.featureLogProb = self.computeFeatureLikelihood(X, y)
        
        self.isFitted = True
        return self
    
    def predictLogProba(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        nSamples, nFeatures = X.shape
        nClasses = len(self.classes)
        
        # Initialize log probabilities with class priors
        logProba = np.zeros((nSamples, nClasses))
        logProba += self.classLogPrior
        
        # Add feature log probabilities
        for i in range(nSamples):
            for j in range(nClasses):
                featureMask = X[i] > 0
                if np.any(featureMask):
                    logProba[i, j] += np.sum(self.featureLogProb[j][featureMask] * X[i][featureMask])
        
        return logProba
    
    def predictProba(self, X):
        logProba = self.predictLogProba(X)
        maxLogProba = np.max(logProba, axis=1, keepdims=True)
        expLogProba = np.exp(logProba - maxLogProba)
        return expLogProba / np.sum(expLogProba, axis=1, keepdims=True)
    
    def predict(self, X):
        logProba = self.predictLogProba(X)
        return self.classes[np.argmax(logProba, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getFeatureLogProb(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.featureLogProb
    
    def getClassLogPrior(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.classLogPrior
    
    def getParams(self):
        return {
            'alpha': self.alpha,
            'fitPrior': self.fitPrior,
            'classPrior': self.classPrior,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'classCounts': self.classCounts.tolist() if self.classCounts is not None else None,
            'nFeatures': self.featureLogProb.shape[1] if self.featureLogProb is not None else 0
        }

# ------------------------------------------------------------
# Gaussian, Multinomial, and Bernoulli remain unchanged
# ------------------------------------------------------------

class GaussianNaiveBayes:
    def __init__(self, priors=None, varSmoothing=1e-9):
        self.priors = priors
        self.varSmoothing = varSmoothing
        self.classes = None
        self.classCounts = None
        self.theta = None
        self.sigma = None
        self.classPrior = None
        self.isFitted = False
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        nClasses = len(self.classes)
        nFeatures = X.shape[1]
        self.theta = np.zeros((nClasses, nFeatures))
        self.sigma = np.zeros((nClasses, nFeatures))
        self.classCounts = np.zeros(nClasses)
        
        for i, cls in enumerate(self.classes):
            classMask = y == cls
            self.classCounts[i] = np.sum(classMask)
            if self.classCounts[i] > 0:
                XClass = X[classMask]
                self.theta[i] = np.mean(XClass, axis=0)
                self.sigma[i] = np.var(XClass, axis=0)
        
        self.sigma += self.varSmoothing
        
        if self.priors is not None:
            self.classPrior = np.log(self.priors)
        else:
            self.classPrior = np.log(self.classCounts / len(y))
        
        self.isFitted = True
        return self
    
    def predictLogProba(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        X = np.array(X)
        nSamples, nFeatures = X.shape
        nClasses = len(self.classes)
        logProba = np.zeros((nSamples, nClasses))
        
        for i in range(nClasses):
            diff = X - self.theta[i]
            logVar = np.log(2 * np.pi * self.sigma[i])
            logProba[:, i] = -0.5 * np.sum(logVar + (diff ** 2) / self.sigma[i], axis=1)
            logProba[:, i] += self.classPrior[i]
        return logProba
    
    def predictProba(self, X):
        logProba = self.predictLogProba(X)
        maxLogProba = np.max(logProba, axis=1, keepdims=True)
        expLogProba = np.exp(logProba - maxLogProba)
        return expLogProba / np.sum(expLogProba, axis=1, keepdims=True)
    
    def predict(self, X):
        logProba = self.predictLogProba(X)
        return self.classes[np.argmax(logProba, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

class MultinomialNaiveBayes(NaiveBayesClassifier):
    pass  # Inherits everything

class BernoulliNaiveBayes(NaiveBayesClassifier):
    def __init__(self, alpha=1.0, binarize=0.0, fitPrior=True, classPrior=None):
        super().__init__(alpha, fitPrior, classPrior)
        self.binarize = binarize
    
    def fit(self, X, y):
        X = np.array(X)
        if self.binarize is not None:
            X = (X > self.binarize).astype(int)
        return super().fit(X, y)
