import numpy as np

class MobileNet:
    def __init__(self, alpha=1.0, learningRate=0.001, maxEpochs=50, 
                 batchSize=32, randomState=None):
        self.alpha = alpha
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.depthwiseLayers = []
        self.pointwiseLayers = []
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputShape, numClasses):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.depthwiseLayers = []
        self.pointwiseLayers = []
        
        baseFilters = int(32 * self.alpha)
        config = [
            (64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
            (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
            (1024, 2), (1024, 1)
        ]
        
        inChannels = inputShape[2] if len(inputShape) == 3 else 1
        
        for filters, stride in config:
            actualFilters = int(filters * self.alpha)
            
            depthwiseFilters = np.random.randn(3, 3, inChannels, 1) * 0.01
            depthwiseBias = np.zeros(inChannels)
            self.depthwiseLayers.append({
                'filters': depthwiseFilters, 
                'bias': depthwiseBias, 
                'stride': stride
            })
            
            pointwiseFilters = np.random.randn(1, 1, inChannels, actualFilters) * 0.01
            pointwiseBias = np.zeros(actualFilters)
            self.pointwiseLayers.append({
                'filters': pointwiseFilters,
                'bias': pointwiseBias
            })
            
            inChannels = actualFilters
        
        self.fcWeights = np.random.randn(inChannels, numClasses) * 0.01
        self.fcBias = np.zeros(numClasses)
    
    def depthwiseConv2d(self, X, filters, bias, stride=1):
        batchSize, height, width, channels = X.shape
        filterSize = filters.shape[0]
        outputHeight = (height - filterSize) // stride + 1
        outputWidth = (width - filterSize) // stride + 1
        output = np.zeros((batchSize, outputHeight, outputWidth, channels))
        
        for c in range(channels):
            for i in range(outputHeight):
                for j in range(outputWidth):
                    region = X[:, i*stride:i*stride+filterSize, j*stride:j*stride+filterSize, c]
                    output[:, i, j, c] = np.sum(region * filters[:, :, c, 0], axis=(1,2)) + bias[c]
        
        return output
    
    def pointwiseConv2d(self, X, filters, bias):
        batchSize, height, width, inChannels = X.shape
        outChannels = filters.shape[3]
        output = np.zeros((batchSize, height, width, outChannels))
        
        for i in range(height):
            for j in range(width):
                region = X[:, i, j, :]
                output[:, i, j, :] = np.dot(region, filters[0, 0, :, :]) + bias
        
        return output
    
    def relu6(self, x):
        return np.minimum(np.maximum(0, x), 6)
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        current = X
        
        for i in range(len(self.depthwiseLayers)):
            depthwiseLayer = self.depthwiseLayers[i]
            pointwiseLayer = self.pointwiseLayers[i]
            
            current = self.depthwiseConv2d(current, depthwiseLayer['filters'], 
                                         depthwiseLayer['bias'], depthwiseLayer['stride'])
            current = self.relu6(current)
            self.activations.append(current)
            
            current = self.pointwiseConv2d(current, pointwiseLayer['filters'], 
                                         pointwiseLayer['bias'])
            current = self.relu6(current)
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