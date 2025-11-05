import numpy as np

class CNN:
    def __init__(self, learningRate=0.001, maxEpochs=50, batchSize=32, randomState=None):
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.filters = []
        self.biases = []
        self.weights = []
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputShape, numClasses):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.filters = [np.random.randn(3, 3, 1, 8) * 0.1]
        self.biases = [np.zeros(8)]
        
        convOutputSize = (inputShape[0] - 2) * (inputShape[1] - 2) * 8 // 4
        self.weights = [np.random.randn(convOutputSize, 128) * 0.1,
                       np.random.randn(128, numClasses) * 0.1]
        self.biases.extend([np.zeros(128), np.zeros(numClasses)])
    
    def conv2d(self, X, filters, bias):
        batchSize, height, width, channels = X.shape
        filterSize = filters.shape[0]
        outputHeight = height - filterSize + 1
        outputWidth = width - filterSize + 1
        output = np.zeros((batchSize, outputHeight, outputWidth, filters.shape[3]))
        
        for i in range(outputHeight):
            for j in range(outputWidth):
                region = X[:, i:i+filterSize, j:j+filterSize, :]
                output[:, i, j, :] = np.tensordot(region, filters, axes=([1,2,3], [0,1,2])) + bias
        
        return output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def maxPool2d(self, X, poolSize=2):
        batchSize, height, width, channels = X.shape
        outputHeight = height // poolSize
        outputWidth = width // poolSize
        output = np.zeros((batchSize, outputHeight, outputWidth, channels))
        
        for i in range(outputHeight):
            for j in range(outputWidth):
                region = X[:, i*poolSize:(i+1)*poolSize, j*poolSize:(j+1)*poolSize, :]
                output[:, i, j, :] = np.max(region, axis=(1,2))
        
        return output
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        
        conv = self.conv2d(X, self.filters[0], self.biases[0])
        relu1 = self.relu(conv)
        pool = self.maxPool2d(relu1)
        
        self.activations.extend([conv, relu1, pool])
        
        flattened = pool.reshape(pool.shape[0], -1)
        self.activations.append(flattened)
        
        fc1 = np.dot(flattened, self.weights[0]) + self.biases[1]
        relu2 = self.relu(fc1)
        
        self.activations.extend([fc1, relu2])
        
        fc2 = np.dot(relu2, self.weights[1]) + self.biases[2]
        output = self.softmax(fc2)
        
        self.activations.append(fc2)
        self.activations.append(output)
        
        return output
    
    def computeLoss(self, yPred, yTrue):
        m = yTrue.shape[0]
        logLikelihood = -np.log(yPred[range(m), yTrue])
        return np.sum(logLikelihood) / m
    
    def backward(self, X, y):
        m = X.shape[0]
        gradients = {}
        
        dZ3 = self.activations[-1]
        dZ3[range(m), y] -= 1
        dZ3 /= m
        
        gradients['W2'] = np.dot(self.activations[6].T, dZ3)
        gradients['b2'] = np.sum(dZ3, axis=0)
        
        dA2 = np.dot(dZ3, self.weights[1].T)
        dZ2 = dA2 * (self.activations[6] > 0)
        
        gradients['W1'] = np.dot(self.activations[4].T, dZ2)
        gradients['b1'] = np.sum(dZ2, axis=0)
        
        return gradients
    
    def updateParameters(self, gradients):
        self.weights[1] -= self.learningRate * gradients['W2']
        self.biases[2] -= self.learningRate * gradients['b2']
        self.weights[0] -= self.learningRate * gradients['W1']
        self.biases[1] -= self.learningRate * gradients['b1']
    
    def fit(self, X, y):
        X = np.array(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        y = np.array(y)
        
        nSamples = X.shape[0]
        numClasses = len(np.unique(y))
        
        self.initializeParameters(X.shape[1:3], numClasses)
        
        for epoch in range(self.maxEpochs):
            indices = np.random.permutation(nSamples)
            epochLoss = 0
            correctPredictions = 0
            
            for i in range(0, nSamples, self.batchSize):
                XBatch = X[indices[i:i + self.batchSize]]
                yBatch = y[indices[i:i + self.batchSize]]
                
                yPred = self.forward(XBatch)
                loss = self.computeLoss(yPred, yBatch)
                epochLoss += loss * len(XBatch)
                
                preds = np.argmax(yPred, axis=1)
                correctPredictions += np.sum(preds == yBatch)
                
                gradients = self.backward(XBatch, yBatch)
                self.updateParameters(gradients)
            
            avgLoss = epochLoss / nSamples
            accuracy = correctPredictions / nSamples
            
            self.lossHistory.append(avgLoss)
            self.accuracyHistory.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avgLoss:.4f}, Accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        yPred = self.forward(X)
        return np.argmax(yPred, axis=1)
    
    def predictProba(self, X):
        X = np.array(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return self.forward(X)