import numpy as np

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1Ratio=0.5, fitIntercept=True, maxIter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1Ratio = l1Ratio
        self.fitIntercept = fitIntercept
        self.maxIter = maxIter
        self.tol = tol
        
        self.weights = None
        self.bias = 0.0
        self.isFitted = False
    
    def softThreshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.fitIntercept:
            X = X - np.mean(X, axis=0)
            y = y - np.mean(y)
        
        nSamples, nFeatures = X.shape
        weights = np.zeros(nFeatures)
        
        for iteration in range(self.maxIter):
            weightsOld = weights.copy()
            
            for j in range(nFeatures):
                Xj = X[:, j]
                yPred = X @ weights
                weights[j] = 0
                
                residual = y - yPred + Xj * weightsOld[j]
                numerator = Xj @ residual
                denominator = Xj @ Xj + self.alpha * (1 - self.l1Ratio)
                
                if denominator == 0:
                    continue
                
                update = numerator / denominator
                threshold = self.alpha * self.l1Ratio / denominator
                weights[j] = self.softThreshold(update, threshold)
            
            if np.max(np.abs(weights - weightsOld)) < self.tol:
                break
        
        self.weights = weights
        
        if self.fitIntercept:
            self.bias = np.mean(y) - np.mean(X @ weights)
        else:
            self.bias = 0.0
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def getCoefficients(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.weights
    
    def getIntercept(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.bias
    
    def getSparsity(self):
        if not self.isFitted or self.weights is None:
            return 0
        return np.mean(self.weights == 0)
    
    def getParams(self):
        return {
            'alpha': self.alpha,
            'l1Ratio': self.l1Ratio,
            'fitIntercept': self.fitIntercept,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias,
            'sparsity': self.getSparsity()
        }