import numpy as np
from scipy.linalg import eigh

class LinearDiscriminantAnalysis:
    def __init__(self, solver='svd', shrinkage=None, priors=None, nComponents=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.nComponents = nComponents
        
        self.classes = None
        self.priors = None
        self.means = None
        self.covariance = None
        self.scalings = None
        self.xbar = None
        self.coef = None
        self.intercept = None
        self.explainedVarianceRatio = None
        self.isFitted = False
    
    def computeClassMeans(self, X, y):
        nFeatures = X.shape[1]
        nClasses = len(self.classes)
        
        means = np.zeros((nClasses, nFeatures))
        for i, cls in enumerate(self.classes):
            classMask = y == cls
            means[i] = np.mean(X[classMask], axis=0)
        
        return means
    
    def computePooledCovariance(self, X, y):
        nSamples, nFeatures = X.shape
        nClasses = len(self.classes)
        
        if self.solver == 'svd':
            # Use SVD-based covariance estimation
            cov = np.zeros((nFeatures, nFeatures))
            for i, cls in enumerate(self.classes):
                classMask = y == cls
                nClass = np.sum(classMask)
                XClass = X[classMask] - self.means[i]
                cov += (nClass - 1) * np.cov(XClass, rowvar=False, bias=True)
            
            cov /= (nSamples - nClasses)
            
        elif self.solver == 'eigen':
            # Use eigenvalue decomposition
            overallMean = np.mean(X, axis=0)
            cov = np.zeros((nFeatures, nFeatures))
            
            for i, cls in enumerate(self.classes):
                classMask = y == cls
                nClass = np.sum(classMask)
                XClass = X[classMask] - self.means[i]
                cov += (nClass - 1) * np.cov(XClass, rowvar=False, bias=True)
            
            cov /= (nSamples - nClasses)
            
            # Apply shrinkage if specified
            if self.shrinkage is not None:
                if self.shrinkage == 'auto':
                    # Ledoit-Wolf shrinkage
                    shrinkage = self.computeLedoitWolfShrinkage(X, cov)
                else:
                    shrinkage = self.shrinkage
                
                # Shrink towards diagonal
                diagCov = np.diag(np.diag(cov))
                cov = (1 - shrinkage) * cov + shrinkage * diagCov
        
        return cov
    
    def computeLedoitWolfShrinkage(self, X, cov):
        # Simplified Ledoit-Wolf shrinkage estimation
        nSamples, nFeatures = X.shape
        
        # Sample covariance matrix
        S = cov
        
        # Target: diagonal matrix with average variance
        mu = np.trace(S) / nFeatures
        T = mu * np.eye(nFeatures)
        
        # Estimate shrinkage parameter
        delta = np.linalg.norm(S - T, 'fro') ** 2
        beta = np.sum([np.linalg.norm(np.outer(X[i], X[i]) - S, 'fro') ** 2 for i in range(nSamples)]) / (nSamples ** 2)
        
        shrinkage = beta / delta if delta > 0 else 0
        return max(0, min(1, shrinkage))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        nClasses = len(self.classes)
        nSamples, nFeatures = X.shape
        
        if nClasses < 2:
            raise ValueError("LDA requires at least 2 classes")
        
        # Compute class priors
        if self.priors is None:
            self.priors = np.array([np.sum(y == cls) / nSamples for cls in self.classes])
        else:
            if len(self.priors) != nClasses:
                raise ValueError("Number of priors must match number of classes")
            self.priors = np.array(self.priors)
        
        # Compute class means
        self.means = self.computeClassMeans(X, y)
        
        # Compute overall mean
        self.xbar = np.mean(X, axis=0)
        
        # Compute pooled covariance matrix
        self.covariance = self.computePooledCovariance(X, y)
        
        # Compute between-class scatter matrix
        Sb = np.zeros((nFeatures, nFeatures))
        for i, cls in enumerate(self.classes):
            nClass = np.sum(y == cls)
            diff = (self.means[i] - self.xbar).reshape(-1, 1)
            Sb += nClass * (diff @ diff.T)
        
        # Solve generalized eigenvalue problem: Sb * w = λ * Sw * w
        if self.solver == 'svd':
            # SVD approach
            U, s, Vt = np.linalg.svd(self.covariance, full_matrices=False)
            rank = np.sum(s > 1e-8)
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank]
            
            # Whitening transform
            scaling = U @ np.diag(1.0 / np.sqrt(s)) @ Vt
            SbStar = scaling.T @ Sb @ scaling
            
            # Eigen decomposition of transformed Sb
            eigenVals, eigenVecs = eigh(SbStar)
            
            # Sort by descending eigenvalues
            idx = np.argsort(eigenVals)[::-1]
            eigenVals = eigenVals[idx]
            eigenVecs = eigenVecs[:, idx]
            
            # Transform back to original space
            self.scalings = scaling @ eigenVecs
            
        else:  # eigen solver
            # Direct generalized eigenvalue decomposition
            eigenVals, eigenVecs = eigh(Sb, self.covariance)
            
            # Sort by descending eigenvalues
            idx = np.argsort(eigenVals)[::-1]
            eigenVals = eigenVals[idx]
            self.scalings = eigenVecs[:, idx]
        
        # Select number of components
        if self.nComponents is None:
            self.nComponents = min(nClasses - 1, nFeatures)
        else:
            self.nComponents = min(self.nComponents, nClasses - 1, nFeatures)
        
        self.scalings = self.scalings[:, :self.nComponents]
        
        # Compute explained variance ratio
        totalVariance = np.sum(eigenVals)
        self.explainedVarianceRatio = eigenVals[:self.nComponents] / totalVariance
        
        # Compute coefficients and intercept for linear decision function
        self.coef = (self.scalings.T @ self.covariance @ self.scalings) @ self.scalings.T @ self.means.T
        self.coef = self.coef.T
        
        self.intercept = -0.5 * np.diag(self.means @ self.scalings @ self.scalings.T @ self.means.T)
        self.intercept += np.log(self.priors)
        
        self.isFitted = True
        return self
    
    def transform(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before transformation")
        
        X = np.array(X)
        return X @ self.scalings
    
    def decisionFunction(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return X @ self.scalings @ self.scalings.T @ self.means.T + self.intercept
    
    def predictLogProba(self, X):
        decisionValues = self.decisionFunction(X)
        
        # Use log-sum-exp trick for numerical stability
        maxDecision = np.max(decisionValues, axis=1, keepdims=True)
        expDecision = np.exp(decisionValues - maxDecision)
        logProba = decisionValues - maxDecision - np.log(np.sum(expDecision, axis=1, keepdims=True))
        
        return logProba
    
    def predictProba(self, X):
        logProba = self.predictLogProba(X)
        return np.exp(logProba)
    
    def predict(self, X):
        proba = self.predictProba(X)
        return self.classes[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getMeans(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.means
    
    def getCovariance(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.covariance
    
    def getScalings(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.scalings
    
    def getExplainedVarianceRatio(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.explainedVarianceRatio
    
    def getCoefficients(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.coef, self.intercept
    
    def getParams(self):
        return {
            'solver': self.solver,
            'shrinkage': self.shrinkage,
            'priors': self.priors.tolist() if self.priors is not None else None,
            'nComponents': self.nComponents,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'explainedVarianceRatio': self.explainedVarianceRatio.tolist() if self.explainedVarianceRatio is not None else None,
            'nFeatures': self.means.shape[1] if self.means is not None else 0,
            'nClasses': len(self.classes) if self.classes is not None else 0
        }
