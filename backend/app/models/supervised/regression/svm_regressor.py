import numpy as np

class SVMRegressor:
    def __init__(self, kernel='linear', C=1.0, epsilon=0.1, gamma='scale', degree=3, 
                 coef0=0.0, tol=1e-3, maxIter=1000, randomState=None):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.maxIter = maxIter
        self.randomState = randomState
        
        self.supportVectors = None
        self.dualCoef = None
        self.intercept = 0.0
        self.weights = None
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def computeKernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (X1.shape[1] * np.var(X1)) if len(X1) > 0 else 1.0
            elif self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1] if X1.shape[1] > 0 else 1.0
            else:
                gamma = self.gamma
            
            normSquared = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-gamma * normSquared)
        elif self.kernel == 'poly':
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples, nFeatures = X.shape
        
        if self.gamma == 'scale':
            self.gamma = 1.0 / (nFeatures * np.var(X)) if nFeatures > 0 else 1.0
        elif self.gamma == 'auto':
            self.gamma = 1.0 / nFeatures if nFeatures > 0 else 1.0
        
        K = self.computeKernel(X, X)
        
        alpha = np.zeros(nSamples)
        alphaStar = np.zeros(nSamples)
        
        for iteration in range(self.maxIter):
            alphaOld = alpha.copy()
            alphaStarOld = alphaStar.copy()
            
            for i in range(nSamples):
                error = (K[i] @ (alpha - alphaStar)) - y[i]
                
                if (error - self.epsilon > 0 and alpha[i] < self.C) or (error + self.epsilon < 0 and alphaStar[i] < self.C):
                    alpha[i] = min(max(alpha[i] + 1 - error, 0), self.C)
                    alphaStar[i] = min(max(alphaStar[i] + 1 + error, 0), self.C)
            
            if np.max(np.abs(alpha - alphaOld)) < self.tol and np.max(np.abs(alphaStar - alphaStarOld)) < self.tol:
                break
        
        supportVectorMask = (alpha > 1e-5) | (alphaStar > 1e-5)
        self.supportVectors = X[supportVectorMask]
        self.dualCoef = alpha[supportVectorMask] - alphaStar[supportVectorMask]
        
        if self.kernel == 'linear':
            self.weights = np.sum(self.dualCoef.reshape(-1, 1) * self.supportVectors, axis=0)
        
        K_sv = self.computeKernel(self.supportVectors, self.supportVectors)
        self.intercept = np.mean(y[supportVectorMask] - K_sv @ self.dualCoef)
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        
        if self.kernel == 'linear' and self.weights is not None:
            return X @ self.weights + self.intercept
        else:
            K = self.computeKernel(X, self.supportVectors)
            return K @ self.dualCoef + self.intercept
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def getSupportVectors(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.supportVectors
    
    def getWeights(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.weights
    
    def getIntercept(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.intercept
    
    def getParams(self):
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'nSupportVectors': len(self.supportVectors) if self.supportVectors is not None else 0,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'intercept': self.intercept
        }