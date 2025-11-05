import numpy as np
from scipy.optimize import minimize

class RidgeClassifier:
    def __init__(self, alpha=1.0, fitIntercept=True, maxIter=1000, tol=1e-4, solver='auto'):
        self.alpha = alpha
        self.fitIntercept = fitIntercept
        self.maxIter = maxIter
        self.tol = tol
        self.solver = solver
        
        self.weights = None
        self.bias = 0.0
        self.classes = None
        self.isFitted = False
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def computeCost(self, weights, X, y):
        z = X @ weights
        predictions = self.sigmoid(z)
        
        logLoss = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
        ridgePenalty = 0.5 * self.alpha * np.sum(weights ** 2)
        
        return logLoss + ridgePenalty
    
    def computeGradient(self, weights, X, y):
        z = X @ weights
        predictions = self.sigmoid(z)
        error = predictions - y
        
        gradient = (X.T @ error) / len(y)
        gradient += self.alpha * weights
        
        return gradient
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("RidgeClassifier supports only binary classification")
        
        yBinary = np.where(y == self.classes[1], 1, 0)
        XWithIntercept = self.addIntercept(X)
        
        nFeatures = XWithIntercept.shape[1]
        initialWeights = np.zeros(nFeatures)
        
        result = minimize(
            fun=self.computeCost,
            x0=initialWeights,
            args=(XWithIntercept, yBinary),
            method='L-BFGS-B',
            jac=self.computeGradient,
            options={'maxiter': self.maxIter, 'ftol': self.tol, 'disp': False}
        )
        
        if self.fitIntercept:
            self.bias = result.x[0]
            self.weights = result.x[1:]
        else:
            self.bias = 0.0
            self.weights = result.x
        
        self.isFitted = True
        return self
    
    def decisionFunction(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def predictProba(self, X):
        decisionValues = self.decisionFunction(X)
        probabilities = self.sigmoid(decisionValues)
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
    
    def getCoefficients(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.weights
    
    def getIntercept(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.bias
    
    def getParams(self):
        return {
            'alpha': self.alpha,
            'fitIntercept': self.fitIntercept,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias,
            'classes': self.classes.tolist() if self.classes is not None else None
        }