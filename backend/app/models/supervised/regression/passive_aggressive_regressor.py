import numpy as np

class PassiveAggressiveRegressor:
    def __init__(self, C=1.0, epsilon=0.1, fitIntercept=True, maxIter=1000, tol=1e-3, shuffle=True, randomState=None):
        self.C = C
        self.epsilon = epsilon
        self.fitIntercept = fitIntercept
        self.maxIter = maxIter
        self.tol = tol
        self.shuffle = shuffle
        self.randomState = randomState
        
        self.weights = None
        self.bias = 0.0
        self.isFitted = False
        
        if randomState is not None:
            np.random.seed(randomState)
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def computeLoss(self, prediction, yTrue):
        error = np.abs(prediction - yTrue)
        return max(0, error - self.epsilon)
    
    def computeStepSize(self, loss, norm):
        if norm == 0:
            return 0
        step = loss / norm
        return min(self.C, step)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.fitIntercept:
            XWithIntercept = self.addIntercept(X)
            nFeatures = XWithIntercept.shape[1]
            weights = np.zeros(nFeatures)
        else:
            XWithIntercept = X
            nFeatures = X.shape[1]
            weights = np.zeros(nFeatures)
            self.bias = 0.0
        
        nSamples = XWithIntercept.shape[0]
        indices = np.arange(nSamples)
        
        for iteration in range(self.maxIter):
            if self.shuffle:
                np.random.shuffle(indices)
            
            weightsOld = weights.copy()
            
            for i in indices:
                xi = XWithIntercept[i]
                yi = y[i]
                
                prediction = xi @ weights
                loss = self.computeLoss(prediction, yi)
                
                if loss > 0:
                    norm = np.sum(xi ** 2)
                    if norm > 0:
                        step = self.computeStepSize(loss, norm)
                        sign = np.sign(yi - prediction)
                        weights += step * sign * xi
            
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
    
    def partialFit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if not self.isFitted:
            return self.fit(X, y)
        
        if self.fitIntercept:
            XWithIntercept = self.addIntercept(X)
            weights = np.concatenate([[self.bias], self.weights])
        else:
            XWithIntercept = X
            weights = self.weights
        
        nSamples = XWithIntercept.shape[0]
        
        for i in range(nSamples):
            xi = XWithIntercept[i]
            yi = y[i]
            
            prediction = xi @ weights
            loss = self.computeLoss(prediction, yi)
            
            if loss > 0:
                norm = np.sum(xi ** 2)
                if norm > 0:
                    step = self.computeStepSize(loss, norm)
                    sign = np.sign(yi - prediction)
                    weights += step * sign * xi
        
        if self.fitIntercept:
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.weights = weights
        
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
            'C': self.C,
            'epsilon': self.epsilon,
            'fitIntercept': self.fitIntercept,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias
        }