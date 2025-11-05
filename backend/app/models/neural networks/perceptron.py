import numpy as np

class Perceptron:
    def __init__(self, learningRate=0.01, maxEpochs=1000, randomState=None):
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.randomState = randomState
        self.weights = None
        self.bias = None
        self.epochWeights = []
        self.epochBiases = []
        self.converged = False
        
    def initializeParameters(self, nFeatures):
        if self.randomState is not None:
            np.random.seed(self.randomState)
        self.weights = np.random.normal(0, 0.01, nFeatures)
        self.bias = 0.0
        
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predictSingle(self, x):
        linearOutput = np.dot(x, self.weights) + self.bias
        return self.activation(linearOutput)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples, nFeatures = X.shape
        self.initializeParameters(nFeatures)
        
        self.epochWeights.append(self.weights.copy())
        self.epochBiases.append(self.bias)
        
        for epoch in range(self.maxEpochs):
            errors = 0
            for i in range(nSamples):
                prediction = self.predictSingle(X[i])
                error = y[i] - prediction
                
                if error != 0:
                    self.weights += self.learningRate * error * X[i]
                    self.bias += self.learningRate * error
                    errors += 1
            
            self.epochWeights.append(self.weights.copy())
            self.epochBiases.append(self.bias)
            
            if errors == 0:
                self.converged = True
                break
                
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.predictSingle(X[i]))
        return np.array(predictions)
    
    def predictProba(self, X):
        X = np.array(X)
        linearOutput = np.dot(X, self.weights) + self.bias
        probPositive = 1 / (1 + np.exp(-linearOutput))
        probNegative = 1 - probPositive
        return np.column_stack([probNegative, probPositive])
    
    def decisionFunction(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def getWeights(self):
        return self.weights, self.bias