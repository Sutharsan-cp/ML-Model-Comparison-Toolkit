import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0, fitIntercept=True):
        self.alpha = alpha
        self.fitIntercept = fitIntercept
        
        self.weights = None
        self.bias = 0.0
        self.isFitted = False
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        XWithIntercept = self.addIntercept(X)
        nFeatures = XWithIntercept.shape[1]
        
        identityMatrix = np.eye(nFeatures)
        if self.fitIntercept:
            identityMatrix[0, 0] = 0
        
        try:
            weights = np.linalg.inv(XWithIntercept.T @ XWithIntercept + self.alpha * identityMatrix) @ XWithIntercept.T @ y
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(XWithIntercept.T @ XWithIntercept + self.alpha * identityMatrix) @ XWithIntercept.T @ y
        
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
            'alpha': self.alpha,
            'fitIntercept': self.fitIntercept,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias
        }