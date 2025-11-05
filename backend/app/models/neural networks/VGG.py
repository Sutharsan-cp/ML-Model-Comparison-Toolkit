import numpy as np

class VGG:
    def __init__(self, architecture='vgg16', learningRate=0.001, maxEpochs=50, 
                 batchSize=32, randomState=None):
        self.architecture = architecture
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.convLayers = []
        self.fcLayers = []
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputShape, numClasses):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        if self.architecture == 'vgg16':
            convConfig = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
                         512, 512, 512, 'M', 512, 512, 512, 'M']
        else:
            convConfig = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
                         512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        
        self.convLayers = []
        inChannels = inputShape[2] if len(inputShape) == 3 else 1
        height, width = inputShape[0], inputShape[1]
        
        for config in convConfig:
            if config == 'M':
                height //= 2
                width //= 2
            else:
                filters = np.random.randn(3, 3, inChannels, config) * 0.01
                bias = np.zeros(config)
                self.convLayers.append({'filters': filters, 'bias': bias, 'type': 'conv'})
                inChannels = config
        
        fcInputSize = height * width * inChannels
        self.fcLayers = [
            {'weights': np.random.randn(fcInputSize, 4096) * 0.01, 'bias': np.zeros(4096)},
            {'weights': np.random.randn(4096, 4096) * 0.01, 'bias': np.zeros(4096)},
            {'weights': np.random.randn(4096, numClasses) * 0.01, 'bias': np.zeros(numClasses)}
        ]
    
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
    
    def maxPool2d(self, X, poolSize=2, stride=2):
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
        current = X
        
        for layer in self.convLayers:
            if layer['type'] == 'conv':
                current = self.conv2d(current, layer['filters'], layer['bias'])
                current = self.relu(current)
                self.activations.append(current)
            else:
                current = self.maxPool2d(current)
                self.activations.append(current)
        
        flattened = current.reshape(current.shape[0], -1)
        self.activations.append(flattened)
        
        for layer in self.fcLayers:
            current = np.dot(self.activations[-1], layer['weights']) + layer['bias']
            current = self.relu(current) if layer is not self.fcLayers[-1] else current
            self.activations.append(current)
        
        if self.fcLayers:
            output = self.softmax(current)
            self.activations.append(output)
            return output
        return current
    
    def computeLoss(self, yPred, yTrue):
        m = yTrue.shape[0]
        logLikelihood = -np.log(yPred[range(m), yTrue])
        return np.sum(logLikelihood) / m
    
    def backward(self, X, y):
        m = X.shape[0]
        gradients = {}
        
        dZ = self.activations[-1]
        dZ[range(m), y] -= 1
        dZ /= m
        
        for i in reversed(range(len(self.fcLayers))):
            gradients[f'W_fc_{i}'] = np.dot(self.activations[-3-i].T, dZ)
            gradients[f'b_fc_{i}'] = np.sum(dZ, axis=0)
            
            if i > 0:
                dA = np.dot(dZ, self.fcLayers[i]['weights'].T)
                dZ = dA * (self.activations[-3-i] > 0)
        
        return gradients
    
    def updateParameters(self, gradients):
        for i in range(len(self.fcLayers)):
            self.fcLayers[i]['weights'] -= self.learningRate * gradients[f'W_fc_{i}']
            self.fcLayers[i]['bias'] -= self.learningRate * gradients[f'b_fc_{i}']
    
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