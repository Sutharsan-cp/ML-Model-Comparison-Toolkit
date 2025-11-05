import numpy as np

class DenseNet:
    def __init__(self, growthRate=32, blockConfig=[6, 12, 24, 16], 
                 learningRate=0.001, maxEpochs=50, batchSize=32, randomState=None):
        self.growthRate = growthRate
        self.blockConfig = blockConfig
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.convLayers = []
        self.denseBlocks = []
        self.transitionLayers = []
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputShape, numClasses):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.convLayers = []
        self.denseBlocks = []
        self.transitionLayers = []
        
        inChannels = inputShape[2] if len(inputShape) == 3 else 1
        
        initialConv = {
            'filters': np.random.randn(7, 7, inChannels, 64) * 0.01,
            'bias': np.zeros(64),
            'stride': 2
        }
        self.convLayers.append(initialConv)
        inChannels = 64
        
        for i, numLayers in enumerate(self.blockConfig):
            denseBlock = []
            for j in range(numLayers):
                conv1 = {
                    'filters': np.random.randn(1, 1, inChannels, 4 * self.growthRate) * 0.01,
                    'bias': np.zeros(4 * self.growthRate)
                }
                conv2 = {
                    'filters': np.random.randn(3, 3, 4 * self.growthRate, self.growthRate) * 0.01,
                    'bias': np.zeros(self.growthRate)
                }
                denseBlock.append({'conv1': conv1, 'conv2': conv2})
                inChannels += self.growthRate
            
            self.denseBlocks.append(denseBlock)
            
            if i != len(self.blockConfig) - 1:
                transition = {
                    'filters': np.random.randn(1, 1, inChannels, inChannels // 2) * 0.01,
                    'bias': np.zeros(inChannels // 2)
                }
                self.transitionLayers.append(transition)
                inChannels = inChannels // 2
        
        self.fcWeights = np.random.randn(inChannels, numClasses) * 0.01
        self.fcBias = np.zeros(numClasses)
    
    def conv2d(self, X, filters, bias, stride=1, padding=0):
        batchSize, height, width, channels = X.shape
        filterSize = filters.shape[0]
        outputHeight = (height + 2 * padding - filterSize) // stride + 1
        outputWidth = (width + 2 * padding - filterSize) // stride + 1
        output = np.zeros((batchSize, outputHeight, outputWidth, filters.shape[3]))
        
        if padding > 0:
            X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        
        for i in range(outputHeight):
            for j in range(outputWidth):
                region = X[:, i*stride:i*stride+filterSize, j*stride:j*stride+filterSize, :]
                output[:, i, j, :] = np.tensordot(region, filters, axes=([1,2,3], [0,1,2])) + bias
        
        return output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def batchNorm(self, x):
        return (x - np.mean(x, axis=(0,1,2), keepdims=True)) / (np.std(x, axis=(0,1,2), keepdims=True) + 1e-8)
    
    def denseLayer(self, X, layerParams):
        conv1 = self.conv2d(X, layerParams['conv1']['filters'], layerParams['conv1']['bias'])
        conv1 = self.batchNorm(conv1)
        conv1 = self.relu(conv1)
        
        conv2 = self.conv2d(conv1, layerParams['conv2']['filters'], layerParams['conv2']['bias'], padding=1)
        conv2 = self.batchNorm(conv2)
        conv2 = self.relu(conv2)
        
        return conv2
    
    def denseBlock(self, X, blockLayers):
        features = [X]
        current = X
        
        for layer in blockLayers:
            newFeatures = self.denseLayer(current, layer)
            features.append(newFeatures)
            current = np.concatenate(features, axis=3)
        
        return current
    
    def avgPool2d(self, X, poolSize=2, stride=2):
        batchSize, height, width, channels = X.shape
        outputHeight = height // poolSize
        outputWidth = width // poolSize
        output = np.zeros((batchSize, outputHeight, outputWidth, channels))
        
        for i in range(outputHeight):
            for j in range(outputWidth):
                region = X[:, i*poolSize:(i+1)*poolSize, j*poolSize:(j+1)*poolSize, :]
                output[:, i, j, :] = np.mean(region, axis=(1,2))
        
        return output
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        current = X
        
        for convLayer in self.convLayers:
            current = self.conv2d(current, convLayer['filters'], convLayer['bias'], 
                                convLayer['stride'], padding=3)
            current = self.batchNorm(current)
            current = self.relu(current)
            current = self.avgPool2d(current, poolSize=3, stride=2)
            self.activations.append(current)
        
        for i, denseBlock in enumerate(self.denseBlocks):
            current = self.denseBlock(current, denseBlock)
            self.activations.append(current)
            
            if i < len(self.transitionLayers):
                transition = self.transitionLayers[i]
                current = self.conv2d(current, transition['filters'], transition['bias'])
                current = self.batchNorm(current)
                current = self.relu(current)
                current = self.avgPool2d(current)
                self.activations.append(current)
        
        current = np.mean(current, axis=(1,2))
        self.activations.append(current)
        
        output = np.dot(current, self.fcWeights) + self.fcBias
        output = self.softmax(output)
        self.activations.append(output)
        
        return output
    
    def computeLoss(self, yPred, yTrue):
        m = yTrue.shape[0]
        logLikelihood = -np.log(yPred[range(m), yTrue])
        return np.sum(logLikelihood) / m
    
    def backward(self, X, y):
        m = X.shape[0]
        
        dZ = self.activations[-1]
        dZ[range(m), y] -= 1
        dZ /= m
        
        dW = np.dot(self.activations[-3].T, dZ)
        db = np.sum(dZ, axis=0)
        
        return dW, db
    
    def updateParameters(self, dW, db):
        self.fcWeights -= self.learningRate * dW
        self.fcBias -= self.learningRate * db
    
    def fit(self, X, y):
        X = np.array(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        y = np.array(y)
        
        nSamples = X.shape[0]
        numClasses = len(np.unique(y))
        
        self.initializeParameters(X.shape[1:], numClasses)
        
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
                
                dW, db = self.backward(XBatch, yBatch)
                self.updateParameters(dW, db)
            
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