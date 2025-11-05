import numpy as np

class Inception:
    def __init__(self, learningRate=0.001, maxEpochs=50, batchSize=32, randomState=None):
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.convLayers = []
        self.inceptionLayers = []
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputShape, numClasses):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.convLayers = []
        self.inceptionLayers = []
        
        inChannels = inputShape[2] if len(inputShape) == 3 else 1
        
        initialConv = {
            'filters': np.random.randn(7, 7, inChannels, 64) * 0.01,
            'bias': np.zeros(64),
            'stride': 2
        }
        self.convLayers.append(initialConv)
        inChannels = 64
        
        for i in range(3):
            inceptionModule = self.createInceptionModule(inChannels)
            self.inceptionLayers.append(inceptionModule)
            inChannels = inceptionModule['outputChannels']
        
        self.fcWeights = np.random.randn(inChannels, numClasses) * 0.01
        self.fcBias = np.zeros(numClasses)
    
    def createInceptionModule(self, inChannels):
        branch1x1 = {
            'filters': np.random.randn(1, 1, inChannels, 64) * 0.01,
            'bias': np.zeros(64)
        }
        
        branch3x3 = {
            'reduce': {
                'filters': np.random.randn(1, 1, inChannels, 96) * 0.01,
                'bias': np.zeros(96)
            },
            'expand': {
                'filters': np.random.randn(3, 3, 96, 128) * 0.01,
                'bias': np.zeros(128)
            }
        }
        
        branch5x5 = {
            'reduce': {
                'filters': np.random.randn(1, 1, inChannels, 16) * 0.01,
                'bias': np.zeros(16)
            },
            'expand': {
                'filters': np.random.randn(5, 5, 16, 32) * 0.01,
                'bias': np.zeros(32)
            }
        }
        
        branchPool = {
            'filters': np.random.randn(1, 1, inChannels, 32) * 0.01,
            'bias': np.zeros(32)
        }
        
        outputChannels = 64 + 128 + 32 + 32
        
        return {
            'branch1x1': branch1x1,
            'branch3x3': branch3x3,
            'branch5x5': branch5x5,
            'branchPool': branchPool,
            'outputChannels': outputChannels
        }
    
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
    
    def maxPool2d(self, X, poolSize=3, stride=1, padding=1):
        batchSize, height, width, channels = X.shape
        outputHeight = (height + 2 * padding - poolSize) // stride + 1
        outputWidth = (width + 2 * padding - poolSize) // stride + 1
        output = np.zeros((batchSize, outputHeight, outputWidth, channels))
        
        if padding > 0:
            X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        
        for i in range(outputHeight):
            for j in range(outputWidth):
                region = X[:, i*stride:i*stride+poolSize, j*stride:j*stride+poolSize, :]
                output[:, i, j, :] = np.max(region, axis=(1,2))
        
        return output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def inceptionForward(self, X, inceptionModule):
        branch1 = self.conv2d(X, inceptionModule['branch1x1']['filters'], 
                            inceptionModule['branch1x1']['bias'])
        branch1 = self.relu(branch1)
        
        branch2 = self.conv2d(X, inceptionModule['branch3x3']['reduce']['filters'],
                            inceptionModule['branch3x3']['reduce']['bias'])
        branch2 = self.relu(branch2)
        branch2 = self.conv2d(branch2, inceptionModule['branch3x3']['expand']['filters'],
                            inceptionModule['branch3x3']['expand']['bias'], padding=1)
        branch2 = self.relu(branch2)
        
        branch3 = self.conv2d(X, inceptionModule['branch5x5']['reduce']['filters'],
                            inceptionModule['branch5x5']['reduce']['bias'])
        branch3 = self.relu(branch3)
        branch3 = self.conv2d(branch3, inceptionModule['branch5x5']['expand']['filters'],
                            inceptionModule['branch5x5']['expand']['bias'], padding=2)
        branch3 = self.relu(branch3)
        
        branch4 = self.maxPool2d(X)
        branch4 = self.conv2d(branch4, inceptionModule['branchPool']['filters'],
                            inceptionModule['branchPool']['bias'])
        branch4 = self.relu(branch4)
        
        output = np.concatenate([branch1, branch2, branch3, branch4], axis=3)
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
            current = self.relu(current)
            current = self.maxPool2d(current, poolSize=3, stride=2, padding=1)
            self.activations.append(current)
        
        for inceptionModule in self.inceptionLayers:
            current = self.inceptionForward(current, inceptionModule)
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