import numpy as np

class Autoencoder:
    def __init__(self, encodingDim=32, hiddenLayers=[64, 32], activation='relu',
                 learningRate=0.001, maxEpochs=100, batchSize=32, randomState=None):
        self.encodingDim = encodingDim
        self.hiddenLayers = hiddenLayers
        self.activation = activation
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.encoderWeights = []
        self.encoderBiases = []
        self.decoderWeights = []
        self.decoderBiases = []
        self.lossHistory = []
        
    def initializeParameters(self, inputDim):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.encoderWeights = []
        self.encoderBiases = []
        self.decoderWeights = []
        self.decoderBiases = []
        
        encoderDims = [inputDim] + self.hiddenLayers + [self.encodingDim]
        decoderDims = [self.encodingDim] + self.hiddenLayers[::-1] + [inputDim]
        
        for i in range(len(encoderDims) - 1):
            scale = np.sqrt(2.0 / encoderDims[i])
            w = np.random.randn(encoderDims[i], encoderDims[i + 1]) * scale
            b = np.zeros(encoderDims[i + 1])
            self.encoderWeights.append(w)
            self.encoderBiases.append(b)
        
        for i in range(len(decoderDims) - 1):
            scale = np.sqrt(2.0 / decoderDims[i])
            w = np.random.randn(decoderDims[i], decoderDims[i + 1]) * scale
            b = np.zeros(decoderDims[i + 1])
            self.decoderWeights.append(w)
            self.decoderBiases.append(b)
    
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
    
    def encode(self, X):
        current = X
        for i in range(len(self.encoderWeights)):
            current = np.dot(current, self.encoderWeights[i]) + self.encoderBiases[i]
            if i < len(self.encoderWeights) - 1:
                current = self.activationFunction(current)
        return current
    
    def decode(self, encoded):
        current = encoded
        for i in range(len(self.decoderWeights)):
            current = np.dot(current, self.decoderWeights[i]) + self.decoderBiases[i]
            if i < len(self.decoderWeights) - 1:
                current = self.activationFunction(current)
            else:
                current = 1 / (1 + np.exp(-current))
        return current
    
    def forward(self, X):
        self.encoded = self.encode(X)
        self.decoded = self.decode(self.encoded)
        return self.decoded
    
    def computeLoss(self, X, reconstructed):
        return np.mean((X - reconstructed) ** 2)
    
    def backward(self, X, reconstructed):
        m = X.shape[0]
        
        dReconstructed = 2 * (reconstructed - X) / m
        
        decoderGradientsW = [np.zeros_like(w) for w in self.decoderWeights]
        decoderGradientsB = [np.zeros_like(b) for b in self.decoderBiases]
        
        dA = dReconstructed
        for i in reversed(range(len(self.decoderWeights))):
            if i == len(self.decoderWeights) - 1:
                dZ = dA * reconstructed * (1 - reconstructed)
            else:
                dZ = dA * self.activationFunction(self.decoderActivations[i], derivative=True)
            
            if i > 0:
                inputActivation = self.decoderActivations[i - 1]
            else:
                inputActivation = self.encoded
            
            decoderGradientsW[i] = np.dot(inputActivation.T, dZ)
            decoderGradientsB[i] = np.sum(dZ, axis=0)
            
            if i > 0:
                dA = np.dot(dZ, self.decoderWeights[i].T)
        
        encoderGradientsW = [np.zeros_like(w) for w in self.encoderWeights]
        encoderGradientsB = [np.zeros_like(b) for b in self.encoderBiases]
        
        dA = np.dot(dZ, self.decoderWeights[0].T)
        for i in reversed(range(len(self.encoderWeights))):
            dZ = dA * self.activationFunction(self.encoderActivations[i], derivative=True)
            
            if i > 0:
                inputActivation = self.encoderActivations[i - 1]
            else:
                inputActivation = X
            
            encoderGradientsW[i] = np.dot(inputActivation.T, dZ)
            encoderGradientsB[i] = np.sum(dZ, axis=0)
            
            if i > 0:
                dA = np.dot(dZ, self.encoderWeights[i].T)
        
        return encoderGradientsW, encoderGradientsB, decoderGradientsW, decoderGradientsB
    
    def updateParameters(self, encGradW, encGradB, decGradW, decGradB):
        for i in range(len(self.encoderWeights)):
            self.encoderWeights[i] -= self.learningRate * encGradW[i]
            self.encoderBiases[i] -= self.learningRate * encGradB[i]
        
        for i in range(len(self.decoderWeights)):
            self.decoderWeights[i] -= self.learningRate * decGradW[i]
            self.decoderBiases[i] -= self.learningRate * decGradB[i]
    
    def fit(self, X):
        X = np.array(X)
        nSamples, inputDim = X.shape
        
        self.initializeParameters(inputDim)
        
        for epoch in range(self.maxEpochs):
            indices = np.random.permutation(nSamples)
            epochLoss = 0
            
            for i in range(0, nSamples, self.batchSize):
                XBatch = X[indices[i:i + self.batchSize]]
                
                self.encoderActivations = [XBatch]
                current = XBatch
                for j in range(len(self.encoderWeights)):
                    current = np.dot(current, self.encoderWeights[j]) + self.encoderBiases[j]
                    if j < len(self.encoderWeights) - 1:
                        current = self.activationFunction(current)
                    self.encoderActivations.append(current)
                
                self.encoded = current
                
                self.decoderActivations = [self.encoded]
                current = self.encoded
                for j in range(len(self.decoderWeights)):
                    current = np.dot(current, self.decoderWeights[j]) + self.decoderBiases[j]
                    if j < len(self.decoderWeights) - 1:
                        current = self.activationFunction(current)
                    self.decoderActivations.append(current)
                
                reconstructed = current
                loss = self.computeLoss(XBatch, reconstructed)
                epochLoss += loss * len(XBatch)
                
                encGradW, encGradB, decGradW, decGradB = self.backward(XBatch, reconstructed)
                self.updateParameters(encGradW, encGradB, decGradW, decGradB)
            
            avgLoss = epochLoss / nSamples
            self.lossHistory.append(avgLoss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avgLoss:.4f}")
        
        return self
    
    def transform(self, X):
        X = np.array(X)
        return self.encode(X)
    
    def reconstruct(self, X):
        X = np.array(X)
        return self.forward(X)