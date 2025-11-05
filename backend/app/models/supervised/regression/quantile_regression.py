import numpy as np

class QuantileRegression:
    def __init__(self, quantile=0.5, fitIntercept=True, maxIter=1000, tol=1e-4):
        self.quantile = quantile
        self.fitIntercept = fitIntercept
        self.maxIter = maxIter
        self.tol = tol
        
        self.weights = None
        self.bias = 0.0
        self.isFitted = False
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def quantileLoss(self, residuals):
        return np.sum(np.where(residuals >= 0, self.quantile * residuals, (self.quantile - 1) * residuals))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        XWithIntercept = self.addIntercept(X)
        nSamples, nFeatures = XWithIntercept.shape
        
        weights = np.zeros(nFeatures)
        
        for iteration in range(self.maxIter):
            weightsOld = weights.copy()
            predictions = XWithIntercept @ weights
            residuals = y - predictions
            
            for j in range(nFeatures):
                Xj = XWithIntercept[:, j]
                weights[j] = 0
                
                currentResiduals = y - XWithIntercept @ weights
                indicator = (currentResiduals <= 0).astype(float)
                gradient = np.mean(Xj * (self.quantile - indicator))
                
                learningRate = 1.0 / (iteration + 1)
                weights[j] = weightsOld[j] - learningRate * gradient
            
            if np.max(np.abs(weights - weightsOld)) < self.tol:
                break
        
        if self.fitIntercept:
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.bias = 0.0
            self.weights = weights
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        predictions = self.predict(X)
        residuals = y - predictions
        quantileLoss = self.quantileLoss(residuals)
        return -quantileLoss
    
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
            'quantile': self.quantile,
            'fitIntercept': self.fitIntercept,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias
        }