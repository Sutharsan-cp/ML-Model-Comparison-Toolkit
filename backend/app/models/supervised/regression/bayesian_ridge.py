import numpy as np

class BayesianRidge:
    def __init__(self, nIter=300, tol=1e-3, alpha1=1e-6, alpha2=1e-6, lambda1=1e-6, lambda2=1e-6, fitIntercept=True):
        self.nIter = nIter
        self.tol = tol
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.fitIntercept = fitIntercept
        
        self.weights = None
        self.bias = 0.0
        self.alpha = None
        self.lambdaReg = None
        self.sigma = None
        self.isFitted = False
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        XWithIntercept = self.addIntercept(X)
        nSamples, nFeatures = XWithIntercept.shape
        
        alpha = 1.0
        lambdaReg = 1.0
        
        weights = np.zeros(nFeatures)
        sigma = np.eye(nFeatures)
        
        for iteration in range(self.nIter):
            alphaOld = alpha
            lambdaOld = lambdaReg
            
            sigma = np.linalg.inv(alpha * np.eye(nFeatures) + lambdaReg * XWithIntercept.T @ XWithIntercept)
            weights = lambdaReg * sigma @ XWithIntercept.T @ y
            
            gamma = np.sum(1 - alpha * np.diag(sigma))
            residuals = y - XWithIntercept @ weights
            
            alpha = (gamma + 2 * self.alpha1) / (np.sum(weights ** 2) + 2 * self.alpha2)
            lambdaReg = (nSamples - gamma + 2 * self.lambda1) / (np.sum(residuals ** 2) + 2 * self.lambda2)
            
            if abs(alpha - alphaOld) < self.tol and abs(lambdaReg - lambdaOld) < self.tol:
                break
        
        self.alpha = alpha
        self.lambdaReg = lambdaReg
        self.sigma = sigma
        
        if self.fitIntercept:
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.bias = 0.0
            self.weights = weights
        
        self.isFitted = True
        return self
    
    def predict(self, X, returnStd=False):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        
        if self.fitIntercept:
            XWithIntercept = np.column_stack([np.ones(X.shape[0]), X])
        else:
            XWithIntercept = X
        
        predictions = XWithIntercept @ np.concatenate([[self.bias], self.weights]) if self.fitIntercept else X @ self.weights
        
        if returnStd:
            var = 1.0 / self.alpha + np.sum(XWithIntercept @ self.sigma * XWithIntercept, axis=1)
            std = np.sqrt(var)
            return predictions, std
        else:
            return predictions
    
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
    
    def getUncertainty(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return np.sqrt(np.diag(self.sigma))
    
    def getParams(self):
        return {
            'alpha': self.alpha,
            'lambdaReg': self.lambdaReg,
            'fitIntercept': self.fitIntercept,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias,
            'uncertainty': np.sqrt(np.diag(self.sigma)).tolist() if self.sigma is not None else None
        }