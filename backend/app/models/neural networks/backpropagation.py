import numpy as np

class Backpropagation:
    def __init__(self, layers=[64, 32, 10], activation='sigmoid', learningRate=0.01,
                 maxEpochs=100, batchSize=32, randomState=None):
        self.layers = layers
        self.activation = activation
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.zValues = []
        self.lossHistory = []
        self.accuracyHistory = []
        self.gradientNorms = []
        
    def initializeParameters(self, inputDim, outputDim):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        layerDims = [inputDim] + self.layers + [outputDim]
        self.weights = []
        self.biases = []
        
        for i in range(len(layerDims) - 1):
            w = np.random.randn(layerDims[i], layerDims[i + 1]) * 0.01
            b = np.zeros((1, layerDims[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def activationFunction(self, z, derivative=False):
        if self.activation == 'sigmoid':
            if derivative:
                s = 1 / (1 + np.exp(-z))
                return s * (1 - s)
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(z)**2
            return np.tanh(z)
        elif self.activation == 'relu':
            if derivative:
                return np.where(z > 0, 1, 0)
            return np.maximum(0, z)
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.zValues = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.activationFunction(z)
            self.zValues.append(z)
            self.activations.append(a)
        
        zOutput = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        aOutput = self.softmax(zOutput)
        self.zValues.append(zOutput)
        self.activations.append(aOutput)
        
        return aOutput
    
    def computeLoss(self, yPred, yTrue):
        m = yTrue.shape[0]
        logLikelihood = -np.log(yPred[range(m), yTrue])
        return np.sum(logLikelihood) / m
    
    def backward(self, X, y):
        m = X.shape[0]
        gradientsW = [np.zeros_like(w) for w in self.weights]
        gradientsB = [np.zeros_like(b) for b in self.biases]
        
        dz = self.activations[-1]
        dz[range(m), y] -= 1
        dz /= m
        
        gradientsW[-1] = np.dot(self.activations[-2].T, dz)
        gradientsB[-1] = np.sum(dz, axis=0, keepdims=True)
        
        for l in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[l + 1].T)
            dz = da * self.activationFunction(self.zValues[l], derivative=True)
            
            gradientsW[l] = np.dot(self.activations[l].T, dz)
            gradientsB[l] = np.sum(dz, axis=0, keepdims=True)
        
        gradientNorm = np.sqrt(sum([np.sum(g**2) for g in gradientsW] + [np.sum(g**2) for g in gradientsB]))
        self.gradientNorms.append(gradientNorm)
        
        return gradientsW, gradientsB
    
    def updateParameters(self, gradientsW, gradientsB):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learningRate * gradientsW[i]
            self.biases[i] -= self.learningRate * gradientsB[i]
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples, nFeatures = X.shape
        nClasses = len(np.unique(y))
        
        self.initializeParameters(nFeatures, nClasses)
        
        for epoch in range(self.maxEpochs):
            indices = np.random.permutation(nSamples)
            XShuffled = X[indices]
            yShuffled = y[indices]
            
            epochLoss = 0
            correctPredictions = 0
            
            for i in range(0, nSamples, self.batchSize):
                XBatch = XShuffled[i:i + self.batchSize]
                yBatch = yShuffled[i:i + self.batchSize]
                
                yPred = self.forward(XBatch)
                loss = self.computeLoss(yPred, yBatch)
                epochLoss += loss * len(XBatch)
                
                batchPreds = np.argmax(yPred, axis=1)
                correctPredictions += np.sum(batchPreds == yBatch)
                
                gradientsW, gradientsB = self.backward(XBatch, yBatch)
                self.updateParameters(gradientsW, gradientsB)
            
            avgLoss = epochLoss / nSamples
            accuracy = correctPredictions / nSamples
            
            self.lossHistory.append(avgLoss)
            self.accuracyHistory.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avgLoss:.4f}, Accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        yPred = self.forward(X)
        return np.argmax(yPred, axis=1)
    
    def predictProba(self, X):
        X = np.array(X)
        return self.forward(X)
    
    def getGradientFlow(self):
        return self.gradientNorms
    
    def getParameterStatistics(self):
        weightStats = []
        for i, w in enumerate(self.weights):
            weightStats.append({
                'layer': i,
                'mean': np.mean(w),
                'std': np.std(w),
                'min': np.min(w),
                'max': np.max(w)
            })
        return weightStats