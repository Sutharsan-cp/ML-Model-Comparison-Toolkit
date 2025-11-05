import numpy as np
from scipy.optimize import minimize

class LogisticRegression:
    def __init__(self, learningRate=0.01, maxIter=1000, fitIntercept=True, regularization=None, alpha=1.0, tol=1e-4):
        self.learningRate = learningRate
        self.maxIter = maxIter
        self.fitIntercept = fitIntercept
        self.regularization = regularization
        self.alpha = alpha
        self.tol = tol
        self.weights = None
        self.bias = 0.0
        self.classes = None
        self.isFitted = False
        self.lossHistory = []
    
    def addIntercept(self, X):
        if self.fitIntercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def negativeLogLikelihood(self, weights, X, y):
        m = X.shape[0]
        z = X @ weights
        probabilities = self.sigmoid(z)
        
        # Negative log likelihood
        nll = -np.sum(y * np.log(probabilities + 1e-8) + (1 - y) * np.log(1 - probabilities + 1e-8))
        
        # Regularization
        if self.regularization == 'l1':
            nll += self.alpha * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            nll += 0.5 * self.alpha * np.sum(weights ** 2)
        elif self.regularization == 'elasticnet':
            l1Ratio = 0.5  # Can be made parameter
            nll += self.alpha * (l1Ratio * np.sum(np.abs(weights)) + 
                               0.5 * (1 - l1Ratio) * np.sum(weights ** 2))
        
        return nll
    
    def gradientNLL(self, weights, X, y):
        m = X.shape[0]
        z = X @ weights
        probabilities = self.sigmoid(z)
        error = probabilities - y
        
        gradient = X.T @ error
        
        # Regularization gradients
        if self.regularization == 'l1':
            gradient += self.alpha * np.sign(weights)
        elif self.regularization == 'l2':
            gradient += self.alpha * weights
        elif self.regularization == 'elasticnet':
            l1Ratio = 0.5
            gradient += self.alpha * (l1Ratio * np.sign(weights) + (1 - l1Ratio) * weights)
        
        return gradient
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("This implementation supports only binary classification")
        
        yBinary = np.where(y == self.classes[1], 1, 0)
        XWithIntercept = self.addIntercept(X)
        
        nFeatures = XWithIntercept.shape[1]
        initialWeights = np.random.normal(0, 0.01, nFeatures)
        
        # Store loss history
        self.lossHistory = []
        
        def callback(weights):
            loss = self.negativeLogLikelihood(weights, XWithIntercept, yBinary)
            self.lossHistory.append(loss)
        
        result = minimize(
            fun=self.negativeLogLikelihood,
            x0=initialWeights,
            args=(XWithIntercept, yBinary),
            method='L-BFGS-B',
            jac=self.gradientNLL,
            callback=callback,
            options={
                'maxiter': self.maxIter,
                'disp': False,
                'gtol': self.tol
            }
        )
        
        if self.fitIntercept:
            self.bias = result.x[0]
            self.weights = result.x[1:]
        else:
            self.bias = 0.0
            self.weights = result.x
        
        self.isFitted = True
        self.finalLoss = result.fun
        return self
    
    def predictProba(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        XWithIntercept = self.addIntercept(X)
        
        if self.fitIntercept:
            fullWeights = np.concatenate([[self.bias], self.weights])
        else:
            fullWeights = self.weights
        
        z = XWithIntercept @ fullWeights
        probabilities = self.sigmoid(z)
        
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X):
        probabilities = self.predictProba(X)
        return self.classes[(probabilities[:, 1] > 0.5).astype(int)]
    
    def predictLogProba(self, X):
        probabilities = self.predictProba(X)
        return np.log(probabilities + 1e-8)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getCoefficients(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.weights
    
    def getIntercept(self):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        return self.bias
    
    def getLossHistory(self):
        return self.lossHistory
    
    def getParams(self):
        return {
            'weights': self.weights,
            'bias': self.bias,
            'classes': self.classes,
            'learningRate': self.learningRate,
            'maxIterations': self.maxIter,
            'fitIntercept': self.fitIntercept,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'finalLoss': self.finalLoss if self.isFitted else None
        }