import numpy as np

class HuberRegressor:
    def __init__(self, epsilon=1.35, fitIntercept=True, maxIter=100, tol=1e-3, alpha=0.0001):
        self.epsilon = epsilon
        self.fitIntercept = fitIntercept
        self.maxIter = maxIter
        self.tol = tol
        self.alpha = alpha
        
        self.weights = None
        self.bias = 0.0
        self.isFitted = False
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def huberLoss(self, residuals):
        absResiduals = np.abs(residuals)
        quadratic = 0.5 * residuals ** 2
        linear = self.epsilon * (absResiduals - 0.5 * self.epsilon)
        return np.where(absResiduals <= self.epsilon, quadratic, linear)
    
    def huberGradient(self, residuals, X):
        absResiduals = np.abs(residuals)
        weights = np.where(absResiduals <= self.epsilon, 1.0, self.epsilon / absResiduals)
        return (X.T @ (weights * residuals)) / len(residuals) + self.alpha * np.concatenate([[0], self.weights]) if self.fitIntercept else self.alpha * self.weights
    
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
            
            gradient = self.huberGradient(residuals, XWithIntercept)
            
            learningRate = 1.0 / (iteration + 1)
            weights = weights - learningRate * gradient
            
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
    
    def getParams(self):
        return {
            'epsilon': self.epsilon,
            'fitIntercept': self.fitIntercept,
            'alpha': self.alpha,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias
        }