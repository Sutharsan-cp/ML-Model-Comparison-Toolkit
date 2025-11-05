import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2, fitIntercept=True, includeBias=True):
        self.degree = degree
        self.fitIntercept = fitIntercept
        self.includeBias = includeBias
        
        self.linearModel = None
        self.polyFeatures = None
        self.isFitted = False
    
    def createPolynomialFeatures(self, X):
        nSamples, nFeatures = X.shape
        features = []
        
        if self.includeBias:
            features.append(np.ones(nSamples))
        
        for d in range(1, self.degree + 1):
            for i in range(nFeatures):
                features.append(X[:, i] ** d)
        
        return np.column_stack(features)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.polyFeatures = self.createPolynomialFeatures(X)
        
        if self.fitIntercept:
            XWithIntercept = np.column_stack([np.ones(self.polyFeatures.shape[0]), self.polyFeatures])
        else:
            XWithIntercept = self.polyFeatures
        
        try:
            weights = np.linalg.inv(XWithIntercept.T @ XWithIntercept) @ XWithIntercept.T @ y
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(XWithIntercept.T @ XWithIntercept) @ XWithIntercept.T @ y
        
        if self.fitIntercept:
            self.linearModel = weights
        else:
            self.linearModel = weights
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        polyFeatures = self.createPolynomialFeatures(X)
        
        if self.fitIntercept:
            XWithIntercept = np.column_stack([np.ones(polyFeatures.shape[0]), polyFeatures])
        else:
            XWithIntercept = polyFeatures
        
        return XWithIntercept @ self.linearModel
    
    def score(self, X, y):
        predictions = self.predict(X)
        ssRes = np.sum((y - predictions) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssRes / ssTot) if ssTot != 0 else 0.0
    
    def getCoefficients(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.linearModel
    
    def getParams(self):
        return {
            'degree': self.degree,
            'fitIntercept': self.fitIntercept,
            'includeBias': self.includeBias,
            'coefficients': self.linearModel.tolist() if self.linearModel is not None else None
        }