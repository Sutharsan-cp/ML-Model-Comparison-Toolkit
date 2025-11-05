import numpy as np

class EfficientNet:
    def __init__(self, compoundCoeff=0, learningRate=0.001, maxEpochs=50, 
                 batchSize=32, randomState=None):
        self.compoundCoeff = compoundCoeff
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.mbconvLayers = []
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputShape, numClasses):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.mbconvLayers = []
        
        widthCoeff, depthCoeff, resolution = self.getCompoundCoefficients()
        
        baseConfig = [
            (32, 16, 1, 1, 3, 1),
            (16, 24, 6, 2, 3, 2),
            (24, 40, 6, 2, 5, 2),
            (40, 80, 6, 2, 3, 3),
            (80, 112, 6, 1, 5, 3),
            (112, 192, 6, 2, 5, 4),
            (192, 320, 6, 1, 3, 1)
        ]
        
        inChannels = inputShape[2] if len(inputShape) == 3 else 1
        
        for i, (inCh, outCh, expandRatio, stride, kernelSize, repeats) in enumerate(baseConfig):
            actualInCh = self.roundFilters(inCh * widthCoeff)
            actualOutCh = self.roundFilters(outCh * widthCoeff)
            actualRepeats = self.roundRepeats(repeats * depthCoeff)
            
            for j in range(actualRepeats):
                actualStride = stride if j == 0 else 1
                
                mbconvBlock = {
                    'expandFilters': np.random.randn(1, 1, inChannels, actualInCh * expandRatio) * 0.01,
                    'expandBias': np.zeros(actualInCh * expandRatio),
                    'depthwiseFilters': np.random.randn(kernelSize, kernelSize, actualInCh * expandRatio, 1) * 0.01,
                    'depthwiseBias': np.zeros(actualInCh * expandRatio),
                    'seReduceFilters': np.random.randn(1, 1, actualInCh * expandRatio, actualInCh // 4) * 0.01,
                    'seReduceBias': np.zeros(actualInCh // 4),
                    'seExpandFilters': np.random.randn(1, 1, actualInCh // 4, actualInCh * expandRatio) * 0.01,
                    'seExpandBias': np.zeros(actualInCh * expandRatio),
                    'projectFilters': np.random.randn(1, 1, actualInCh * expandRatio, actualOutCh) * 0.01,
                    'projectBias': np.zeros(actualOutCh),
                    'stride': actualStride,
                    'kernelSize': kernelSize,
                    'expandRatio': expandRatio
                }
                self.mbconvLayers.append(mbconvBlock)
                inChannels = actualOutCh
        
        self.fcWeights = np.random.randn(inChannels, numClasses) * 0.01
        self.fcBias = np.zeros(numClasses)
    
    def getCompoundCoefficients(self):
        coefficients = {
            0: (1.0, 1.0, 224), 1: (1.0, 1.1, 240), 2: (1.1, 1.2, 260),
            3: (1.2, 1.4, 300), 4: (1.4, 1.8, 380), 5: (1.6, 2.2, 456),
            6: (1.8, 2.6, 528), 7: (2.0, 3.1, 600)
        }
        return coefficients.get(self.compoundCoeff, (1.0, 1.0, 224))
    
    def roundFilters(self, filters):
        divisor = 8
        return int(np.round(filters / divisor) * divisor)
    
    def roundRepeats(self, repeats):
        return int(np.ceil(repeats))
    
    def swish(self, x):
        return x * (1 / (1 + np.exp(-x)))
    
    def conv2d(self, X, filters, bias, stride=1):
        batchSize, height, width, inChannels = X.shape
        filterSize = filters.shape[0]
        outChannels = filters.shape[3]
        outputHeight = (height - filterSize) // stride + 1
        outputWidth = (width - filterSize) // stride + 1
        output = np.zeros((batchSize, outputHeight, outputWidth, outChannels))
        
        for i in range(outputHeight):
            for j in range(outputWidth):
                region = X[:, i*stride:i*stride+filterSize, j*stride:j*stride+filterSize, :]
                output[:, i, j, :] = np.tensordot(region, filters, axes=([1,2,3], [0,1,2])) + bias
        
        return output
    
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
    
    def squeezeExcite(self, X, reduceFilters, reduceBias, expandFilters, expandBias):
        batchSize, height, width, channels = X.shape
        
        se = np.mean(X, axis=(1,2))
        se = np.dot(se, reduceFilters[0,0,:,:]) + reduceBias
        se = self.swish(se)
        se = np.dot(se, expandFilters[0,0,:,:]) + expandBias
        se = 1 / (1 + np.exp(-se))
        
        se = se.reshape(batchSize, 1, 1, channels)
        return X * se
    
    def mbconvBlock(self, X, blockParams):
        inChannels = X.shape[3]
        
        if blockParams['expandRatio'] != 1:
            expand = self.conv2d(X, blockParams['expandFilters'], blockParams['expandBias'])
            expand = self.swish(expand)
        else:
            expand = X
        
        depthwise = self.depthwiseConv2d(expand, blockParams['depthwiseFilters'], 
                                       blockParams['depthwiseBias'], blockParams['stride'])
        depthwise = self.swish(depthwise)
        
        se = self.squeezeExcite(depthwise, blockParams['seReduceFilters'], 
                              blockParams['seReduceBias'], blockParams['seExpandFilters'], 
                              blockParams['seExpandBias'])
        
        project = self.conv2d(se, blockParams['projectFilters'], blockParams['projectBias'])
        
        if inChannels == project.shape[3] and blockParams['stride'] == 1:
            project = project + X
        
        return project
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        current = X
        
        for mbconvBlock in self.mbconvLayers:
            current = self.mbconvBlock(current, mbconvBlock)
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
            