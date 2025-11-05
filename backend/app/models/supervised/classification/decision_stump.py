import numpy as np
from collections import Counter

class DecisionStump:
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.feature = None
        self.threshold = None
        self.leftClass = None
        self.rightClass = None
        self.isFitted = False
    
    def gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def entropy(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def computeImpurity(self, y):
        if self.criterion == 'gini':
            return self.gini(y)
        elif self.criterion == 'entropy':
            return self.entropy(y)
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")
    
    def findBestSplit(self, X, y):
        bestImpurity = float('inf')
        bestFeature = None
        bestThreshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                leftMask = X[:, feature] <= threshold
                rightMask = ~leftMask
                
                if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
                    continue
                
                leftImpurity = self.computeImpurity(y[leftMask])
                rightImpurity = self.computeImpurity(y[rightMask])
                totalImpurity = (np.sum(leftMask) * leftImpurity + np.sum(rightMask) * rightImpurity) / len(y)
                
                if totalImpurity < bestImpurity:
                    bestImpurity = totalImpurity
                    bestFeature = feature
                    bestThreshold = threshold
        
        return bestFeature, bestThreshold, bestImpurity
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.feature, self.threshold, impurity = self.findBestSplit(X, y)
        
        if self.feature is not None:
            leftMask = X[:, self.feature] <= self.threshold
            self.leftClass = Counter(y[leftMask]).most_common(1)[0][0]
            self.rightClass = Counter(y[~leftMask]).most_common(1)[0][0]
        else:
            self.leftClass = Counter(y).most_common(1)[0][0]
            self.rightClass = self.leftClass
        
        self.isFitted = True
        return self
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        if self.feature is None:
            return np.full(X.shape[0], self.leftClass)
        
        predictions = np.where(
            X[:, self.feature] <= self.threshold,
            self.leftClass,
            self.rightClass
        )
        return predictions
    
    def predictProba(self, X):
        predictions = self.predict(X)
        allClasses = np.unique(predictions)
        probas = []
        
        for pred in predictions:
            proba = [1.0 if cls == pred else 0.0 for cls in allClasses]
            probas.append(proba)
        
        return np.array(probas)
    
    def getParams(self):
        return {
            'feature': self.feature,
            'threshold': self.threshold,
            'leftClass': self.leftClass,
            'rightClass': self.rightClass,
            'criterion': self.criterion
        }