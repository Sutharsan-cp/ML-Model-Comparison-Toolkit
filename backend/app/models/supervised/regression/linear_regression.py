import numpy as np

class LinearRegression:
    def __init__(self, fitIntercept=True, method='normal'):
        self.fitIntercept = fitIntercept
        self.method = method
        
        self.weights = None
        self.bias = 0.0
        self.isFitted = False
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def fitNormal(self, X, y):
        XWithIntercept = self.addIntercept(X)
        
        try:
            weights = np.linalg.inv(XWithIntercept.T @ XWithIntercept) @ XWithIntercept.T @ y
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(XWithIntercept.T @ XWithIntercept) @ XWithIntercept.T @ y
        
        if self.fitIntercept:
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.bias = 0.0
            self.weights = weights
    
    def fitGradientDescent(self, X, y, learningRate=0.01, maxIter=1000, tol=1e-6):
        nSamples, nFeatures = X.shape
        if self.fitIntercept:
            nFeatures += 1
            XWithIntercept = self.addIntercept(X)
        else:
            XWithIntercept = X
        
        weights = np.zeros(nFeatures)
        
        for iteration in range(maxIter):
            predictions = XWithIntercept @ weights
            errors = predictions - y
            
            gradient = (XWithIntercept.T @ errors) / nSamples
            
            weightsNew = weights - learningRate * gradient
            
            if np.max(np.abs(weightsNew - weights)) < tol:
                break
            
            weights = weightsNew
        
        if self.fitIntercept:
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.bias = 0.0
            self.weights = weights
    
    def fit(self, X, y, learningRate=0.01, maxIter=1000, tol=1e-6):
        X = np.array(X)
        y = np.array(y)
        
        if self.method == 'normal':
            self.fitNormal(X, y)
        elif self.method == 'gradient':
            self.fitGradientDescent(X, y, learningRate, maxIter, tol)
        else:
            raise ValueError("Method must be 'normal' or 'gradient'")
        
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
            'fitIntercept': self.fitIntercept,
            'method': self.method,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias
        }