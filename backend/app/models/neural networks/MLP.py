import numpy as np

class MLP:
    def __init__(self, hiddenLayers=[100, 50], activation='relu', learningRate=0.01,
                 maxEpochs=100, batchSize=32, randomState=None):
        self.hiddenLayers = hiddenLayers
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
        
    def initializeParameters(self, inputDim, outputDim):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        layerDims = [inputDim] + self.hiddenLayers + [outputDim]
        self.weights = []
        self.biases = []
        
        for i in range(len(layerDims) - 1):
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / layerDims[i])
            else:
                scale = np.sqrt(1.0 / layerDims[i])
                
            w = np.random.randn(layerDims[i], layerDims[i + 1]) * scale
            b = np.zeros((1, layerDims[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def activationFunction(self, z, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return np.where(z > 0, 1, 0)
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            if derivative:
                s = 1 / (1 + np.exp(-z))
                return s * (1 - s)
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(z)**2
            return np.tanh(z)
    
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
        
        if len(self.weights[-1][0]) > 1:
            aOutput = self.softmax(zOutput)
        else:
            aOutput = self.activationFunction(zOutput)
            
        self.zValues.append(zOutput)
        self.activations.append(aOutput)
        
        return aOutput
    
    def computeLoss(self, yPred, yTrue):
        if len(yPred[0]) > 1:
            m = yTrue.shape[0]
            logLikelihood = -np.log(yPred[range(m), yTrue])
            loss = np.sum(logLikelihood) / m
        else:
            loss = np.mean((yPred.flatten() - yTrue) ** 2)
        return loss
    
    def backward(self, X, y):
        m = X.shape[0]
        gradientsW = [np.zeros_like(w) for w in self.weights]
        gradientsB = [np.zeros_like(b) for b in self.biases]
        
        if len(self.weights[-1][0]) > 1:
            dz = self.activations[-1]
            dz[range(m), y] -= 1
            dz /= m
        else:
            dz = (self.activations[-1].flatten() - y).reshape(-1, 1) / m
        
        gradientsW[-1] = np.dot(self.activations[-2].T, dz)
        gradientsB[-1] = np.sum(dz, axis=0, keepdims=True)
        
        for l in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[l + 1].T)
            dz = da * self.activationFunction(self.zValues[l], derivative=True)
            
            gradientsW[l] = np.dot(self.activations[l].T, dz)
            gradientsB[l] = np.sum(dz, axis=0, keepdims=True)
        
        return gradientsW, gradientsB
    
    def updateParameters(self, gradientsW, gradientsB):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learningRate * gradientsW[i]
            self.biases[i] -= self.learningRate * gradientsB[i]
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples, nFeatures = X.shape
        
        if len(np.unique(y)) > 2:
            outputDim = len(np.unique(y))
        else:
            outputDim = 1
        
        self.initializeParameters(nFeatures, outputDim)
        
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
                
                if outputDim > 1:
                    preds = np.argmax(yPred, axis=1)
                    correctPredictions += np.sum(preds == yBatch)
                else:
                    preds = (yPred.flatten() > 0.5).astype(int)
                    correctPredictions += np.sum(preds == yBatch)
                
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
        
        if len(yPred[0]) > 1:
            return np.argmax(yPred, axis=1)
        else:
            return (yPred.flatten() > 0.5).astype(int)
    
    def predictProba(self, X):
        X = np.array(X)
        yPred = self.forward(X)
        
        if len(yPred[0]) > 1:
            return yPred
        else:
            probPositive = 1 / (1 + np.exp(-yPred.flatten()))
            probNegative = 1 - probPositive
            return np.column_stack([probNegative, probPositive])