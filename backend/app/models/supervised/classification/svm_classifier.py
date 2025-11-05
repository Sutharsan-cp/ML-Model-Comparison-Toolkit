import numpy as np
from scipy.optimize import minimize

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, coef0=0.0, 
                 tol=1e-3, maxIter=1000, randomState=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.maxIter = maxIter
        self.randomState = randomState
        
        self.supportVectors = None
        self.supportVectorLabels = None
        self.dualCoef = None
        self.intercept = 0.0
        self.classes = None
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def computeKernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (X1.shape[1] * np.var(X1)) if len(X1) > 0 else 1.0
            elif self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1] if X1.shape[1] > 0 else 1.0
            else:
                gamma = self.gamma
            
            normSquared = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-gamma * normSquared)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def computeObjective(self, alpha, K, y):
        return 0.5 * alpha.T @ (K * np.outer(y, y)) @ alpha - np.sum(alpha)
    
    def computeGradient(self, alpha, K, y):
        return (K * np.outer(y, y)) @ alpha - np.ones(len(alpha))
    
    def computeConstraints(self, alpha, y):
        return np.array([np.sum(alpha * y)])
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("SVMClassifier supports only binary classification")
        
        yBinary = np.where(y == self.classes[0], -1, 1)
        nSamples, nFeatures = X.shape
        
        if self.gamma == 'scale':
            self.gamma = 1.0 / (nFeatures * np.var(X)) if nFeatures > 0 else 1.0
        elif self.gamma == 'auto':
            self.gamma = 1.0 / nFeatures if nFeatures > 0 else 1.0
        
        K = self.computeKernel(X, X)
        
        bounds = [(0, self.C) for _ in range(nSamples)]
        constraints = [{'type': 'eq', 'fun': lambda alpha: self.computeConstraints(alpha, yBinary)}]
        
        alphaInitial = np.zeros(nSamples)
        
        result = minimize(
            fun=self.computeObjective,
            x0=alphaInitial,
            args=(K, yBinary),
            method='SLSQP',
            jac=self.computeGradient,
            constraints=constraints,
            bounds=bounds,
            options={'ftol': self.tol, 'maxiter': self.maxIter, 'disp': False}
        )
        
        alpha = result.x
        supportVectorMask = alpha > 1e-5
        
        self.supportVectors = X[supportVectorMask]
        self.supportVectorLabels = yBinary[supportVectorMask]
        self.dualCoef = alpha[supportVectorMask] * self.supportVectorLabels
        
        supportVectorIndices = np.where(supportVectorMask)[0]
        if len(supportVectorIndices) > 0:
            if self.kernel == 'linear':
                self.weights = np.sum(self.dualCoef.reshape(-1, 1) * self.supportVectors, axis=0)
                self.intercept = np.mean(
                    self.supportVectorLabels - 
                    np.sum(self.dualCoef.reshape(-1, 1) * 
                          self.computeKernel(self.supportVectors, self.supportVectors), axis=1)
                )
            else:
                self.weights = None
                K_sv = self.computeKernel(self.supportVectors, self.supportVectors)
                self.intercept = np.mean(
                    self.supportVectorLabels - 
                    np.sum(self.dualCoef.reshape(-1, 1) * K_sv, axis=1)
                )
        else:
            self.weights = np.zeros(nFeatures)
            self.intercept = 0.0
        
        self.isFitted = True
        return self
    
    def decisionFunction(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        
        if self.kernel == 'linear' and self.weights is not None:
            return X @ self.weights + self.intercept
        else:
            K = self.computeKernel(X, self.supportVectors)
            return K @ self.dualCoef + self.intercept
    
    def predict(self, X):
        decisionValues = self.decisionFunction(X)
        return np.where(decisionValues >= 0, self.classes[1], self.classes[0])
    
    def predictProba(self, X):
        decisionValues = self.decisionFunction(X)
        probabilities = 1 / (1 + np.exp(-decisionValues))
        return np.column_stack([1 - probabilities, probabilities])
    
    def predictLogProba(self, X):
        probabilities = self.predictProba(X)
        return np.log(probabilities + 1e-8)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getSupportVectors(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.supportVectors
    
    def getSupportVectorIndices(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        indices = []
        for i, x in enumerate(X):
            for j, sv in enumerate(self.supportVectors):
                if np.allclose(x, sv):
                    indices.append(j)
                    break
            else:
                indices.append(-1)
        return np.array(indices)
    
    def getNumSupportVectors(self):
        if not self.isFitted:
            return 0
        return len(self.supportVectors)
    
    def getWeights(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.weights
    
    def getIntercept(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.intercept
    
    def getDualCoef(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.dualCoef
    
    def computeMargin(self):
        if not self.isFitted or self.kernel != 'linear':
            return None
        
        if self.weights is not None:
            normWeights = np.linalg.norm(self.weights)
            return 2.0 / normWeights if normWeights > 0 else 0.0
        return None
    
    def getParams(self):
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'nSupportVectors': self.getNumSupportVectors(),
            'nFeatures': self.supportVectors.shape[1] if self.supportVectors is not None else 0,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'intercept': self.intercept,
            'margin': self.computeMargin(),
            'classes': self.classes.tolist() if self.classes is not None else None
        }