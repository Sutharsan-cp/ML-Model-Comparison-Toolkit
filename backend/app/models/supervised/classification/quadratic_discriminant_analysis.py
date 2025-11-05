import numpy as np

class QuadraticDiscriminantAnalysis:
    def __init__(self, priors=None, regParam=0.0, tol=1e-4):
        self.priors = priors
        self.regParam = regParam
        self.tol = tol
        
        self.classes = None
        self.priors = None
        self.means = None
        self.covariances = None
        self.covarianceInvs = None
        self.covarianceLogDets = None
        self.isFitted = False
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        nClasses = len(self.classes)
        nSamples, nFeatures = X.shape
        
        if nClasses < 2:
            raise ValueError("QDA requires at least 2 classes")
        
        if self.priors is None:
            self.priors = np.array([np.sum(y == cls) / nSamples for cls in self.classes])
        else:
            if len(self.priors) != nClasses:
                raise ValueError("Number of priors must match number of classes")
            self.priors = np.array(self.priors)
        
        self.means = np.zeros((nClasses, nFeatures))
        self.covariances = np.zeros((nClasses, nFeatures, nFeatures))
        
        for i, cls in enumerate(self.classes):
            classMask = y == cls
            XClass = X[classMask]
            nClass = np.sum(classMask)
            
            self.means[i] = np.mean(XClass, axis=0)
            
            XCentered = XClass - self.means[i]
            cov = (XCentered.T @ XCentered) / (nClass - 1)
            
            if self.regParam > 0:
                cov += self.regParam * np.eye(nFeatures)
            
            self.covariances[i] = cov
        
        self.covarianceInvs = np.zeros_like(self.covariances)
        self.covarianceLogDets = np.zeros(nClasses)
        
        for i in range(nClasses):
            try:
                self.covarianceInvs[i] = np.linalg.inv(self.covariances[i])
                sign, logdet = np.linalg.slogdet(self.covariances[i])
                self.covarianceLogDets[i] = sign * logdet
            except np.linalg.LinAlgError:
                U, s, Vt = np.linalg.svd(self.covariances[i])
                s[s < self.tol] = self.tol
                self.covarianceInvs[i] = U @ np.diag(1.0 / s) @ Vt
                self.covarianceLogDets[i] = np.sum(np.log(s))
        
        self.isFitted = True
        return self
    
    def mahalanobisDistance(self, X, mean, covInv):
        diff = X - mean
        return np.sum(diff @ covInv * diff, axis=1)
    
    def discriminantFunction(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        nSamples = X.shape[0]
        nClasses = len(self.classes)
        
        discriminant = np.zeros((nSamples, nClasses))
        
        for i in range(nClasses):
            mahalanobis = self.mahalanobisDistance(X, self.means[i], self.covarianceInvs[i])
            discriminant[:, i] = -0.5 * (self.covarianceLogDets[i] + mahalanobis)
            discriminant[:, i] += np.log(self.priors[i])
        
        return discriminant
    
    def predictLogProba(self, X):
        discriminant = self.discriminantFunction(X)
        maxDiscriminant = np.max(discriminant, axis=1, keepdims=True)
        logProba = discriminant - maxDiscriminant - np.log(np.sum(np.exp(discriminant - maxDiscriminant), axis=1, keepdims=True))
        return logProba
    
    def predictProba(self, X):
        logProba = self.predictLogProba(X)
        return np.exp(logProba)
    
    def predict(self, X):
        discriminant = self.discriminantFunction(X)
        return self.classes[np.argmax(discriminant, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getMeans(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.means
    
    def getCovariances(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.covariances
    
    def getPriors(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.priors
    
    def getParams(self):
        return {
            'priors': self.priors.tolist() if self.priors is not None else None,
            'regParam': self.regParam,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'nFeatures': self.means.shape[1] if self.means is not None else 0,
            'nClasses': len(self.classes) if self.classes is not None else 0
        }