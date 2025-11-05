import numpy as np

class ResNet:
    def __init__(self, layers=[2, 2, 2, 2], learningRate=0.001, maxEpochs=50, 
                 batchSize=32, randomState=None):
        self.layers = layers
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.convLayers = []
        self.residualLayers = []
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputShape, numClasses):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.convLayers = []
        self.residualLayers = []
        
        inChannels = inputShape[2] if len(inputShape) == 3 else 1
        
        initialConv = {
            'filters': np.random.randn(7, 7, inChannels, 64) * 0.01,
            'bias': np.zeros(64),
            'stride': 2
        }
        self.convLayers.append(initialConv)
        inChannels = 64
        
        channels = [64, 128, 256, 512]
        
        for i, numBlocks in enumerate(self.layers):
            stride = 1 if i == 0 else 2
            
            for j in range(numBlocks):
                residualBlock = []
                
                conv1 = {
                    'filters': np.random.randn(3, 3, inChannels, channels[i]) * 0.01,
                    'bias': np.zeros(channels[i]),
                    'stride': stride if j == 0 else 1
                }
                residualBlock.append(conv1)
                
                conv2 = {
                    'filters': np.random.randn(3, 3, channels[i], channels[i]) * 0.01,
                    'bias': np.zeros(channels[i]),
                    'stride': 1
                }
                residualBlock.append(conv2)
                
                if inChannels != channels[i] or stride > 1:
                    shortcut = {
                        'filters': np.random.randn(1, 1, inChannels, channels[i]) * 0.01,
                        'bias': np.zeros(channels[i]),
                        'stride': stride
                    }
                    residualBlock.append(shortcut)
                else:
                    residualBlock.append(None)
                
                self.residualLayers.append(residualBlock)
                inChannels = channels[i]
        
        self.fcWeights = np.random.randn(inChannels, numClasses) * 0.01
        self.fcBias = np.zeros(numClasses)
    
    def conv2d(self, X, filters, bias, stride=1, padding=1):
        batchSize, height, width, channels = X.shape
        filterSize = filters.shape[0]
        outputHeight = (height + 2 * padding - filterSize) // stride + 1
        outputWidth = (width + 2 * padding - filterSize) // stride + 1
        output = np.zeros((batchSize, outputHeight, outputWidth, filters.shape[3]))
        
        XPad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        
        for i in range(outputHeight):
            for j in range(outputWidth):
                region = XPad[:, i*stride:i*stride+filterSize, j*stride:j*stride+filterSize, :]
                output[:, i, j, :] = np.tensordot(region, filters, axes=([1,2,3], [0,1,2])) + bias
        
        return output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def batchNorm(self, x):
        return (x - np.mean(x, axis=(0,1,2), keepdims=True)) / (np.std(x, axis=(0,1,2), keepdims=True) + 1e-8)
    
    def residualBlock(self, X, blockLayers):
        conv1, conv2, shortcut = blockLayers
        
        residual = self.conv2d(X, conv1['filters'], conv1['bias'], conv1['stride'])
        residual = self.batchNorm(residual)
        residual = self.relu(residual)
        
        residual = self.conv2d(residual, conv2['filters'], conv2['bias'], conv2['stride'])
        residual = self.batchNorm(residual)
        
        if shortcut is not None:
            X = self.conv2d(X, shortcut['filters'], shortcut['bias'], shortcut['stride'], padding=0)
            X = self.batchNorm(X)
        
        output = self.relu(residual + X)
        return output
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        current = X
        
        for convLayer in self.convLayers:
            current = self.conv2d(current, convLayer['filters'], convLayer['bias'], convLayer['stride'])
            current = self.batchNorm(current)
            current = self.relu(current)
            self.activations.append(current)
        
        for residualBlock in self.residualLayers:
            current = self.residualBlock(current, residualBlock)
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