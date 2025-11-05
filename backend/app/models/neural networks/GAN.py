import numpy as np

class GAN:
    def __init__(self, latentDim=100, generatorLayers=[128, 256, 512], 
                 discriminatorLayers=[512, 256, 128], learningRate=0.0002,
                 maxEpochs=100, batchSize=32, randomState=None):
        self.latentDim = latentDim
        self.generatorLayers = generatorLayers
        self.discriminatorLayers = discriminatorLayers
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.generatorWeights = []
        self.generatorBiases = []
        self.discriminatorWeights = []
        self.discriminatorBiases = []
        self.dLossHistory = []
        self.gLossHistory = []
        
    def initializeGenerator(self, outputDim):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        layerDims = [self.latentDim] + self.generatorLayers + [outputDim]
        self.generatorWeights = []
        self.generatorBiases = []
        
        for i in range(len(layerDims) - 1):
            scale = np.sqrt(2.0 / layerDims[i])
            w = np.random.randn(layerDims[i], layerDims[i + 1]) * scale
            b = np.zeros(layerDims[i + 1])
            self.generatorWeights.append(w)
            self.generatorBiases.append(b)
    
    def initializeDiscriminator(self, inputDim):
        layerDims = [inputDim] + self.discriminatorLayers + [1]
        self.discriminatorWeights = []
        self.discriminatorBiases = []
        
        for i in range(len(layerDims) - 1):
            scale = np.sqrt(2.0 / layerDims[i])
            w = np.random.randn(layerDims[i], layerDims[i + 1]) * scale
            b = np.zeros(layerDims[i + 1])
            self.discriminatorWeights.append(w)
            self.discriminatorBiases.append(b)
    
    def generatorActivation(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    def discriminatorActivation(self, x, derivative=False):
        if derivative:
            return 1
        return np.maximum(0.01 * x, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def generatorForward(self, z):
        a = z
        for i in range(len(self.generatorWeights) - 1):
            zLayer = np.dot(a, self.generatorWeights[i]) + self.generatorBiases[i]
            a = self.generatorActivation(zLayer)
        output = np.dot(a, self.generatorWeights[-1]) + self.generatorBiases[-1]
        return np.tanh(output)
    
    def discriminatorForward(self, x):
        a = x
        for i in range(len(self.discriminatorWeights) - 1):
            zLayer = np.dot(a, self.discriminatorWeights[i]) + self.discriminatorBiases[i]
            a = self.discriminatorActivation(zLayer)
        output = np.dot(a, self.discriminatorWeights[-1]) + self.discriminatorBiases[-1]
        return self.sigmoid(output)
    
    def generatorBackward(self, z, discriminatorGrad):
        gradientsW = [np.zeros_like(w) for w in self.generatorWeights]
        gradientsB = [np.zeros_like(b) for b in self.generatorBiases]
        
        a = z
        activations = [a]
        zs = []
        
        for i in range(len(self.generatorWeights) - 1):
            zLayer = np.dot(a, self.generatorWeights[i]) + self.generatorBiases[i]
            a = self.generatorActivation(zLayer)
            zs.append(zLayer)
            activations.append(a)
        
        output = np.dot(a, self.generatorWeights[-1]) + self.generatorBiases[-1]
        generatedData = np.tanh(output)
        
        dOutput = discriminatorGrad * (1 - generatedData**2)
        
        gradientsW[-1] = np.dot(activations[-1].T, dOutput)
        gradientsB[-1] = np.sum(dOutput, axis=0)
        
        dA = np.dot(dOutput, self.generatorWeights[-1].T)
        
        for l in range(len(self.generatorWeights) - 2, -1, -1):
            dZ = dA * self.generatorActivation(zs[l], derivative=True)
            gradientsW[l] = np.dot(activations[l].T, dZ)
            gradientsB[l] = np.sum(dZ, axis=0)
            dA = np.dot(dZ, self.generatorWeights[l].T)
        
        return gradientsW, gradientsB
    
    def discriminatorBackward(self, x, real=True):
        gradientsW = [np.zeros_like(w) for w in self.discriminatorWeights]
        gradientsB = [np.zeros_like(b) for b in self.discriminatorBiases]
        
        a = x
        activations = [a]
        zs = []
        
        for i in range(len(self.discriminatorWeights) - 1):
            zLayer = np.dot(a, self.discriminatorWeights[i]) + self.discriminatorBiases[i]
            a = self.discriminatorActivation(zLayer)
            zs.append(zLayer)
            activations.append(a)
        
        output = np.dot(a, self.discriminatorWeights[-1]) + self.discriminatorBiases[-1]
        prediction = self.sigmoid(output)
        
        if real:
            dOutput = -1 / (prediction + 1e-8)
        else:
            dOutput = 1 / (1 - prediction + 1e-8)
        
        dOutput = dOutput * prediction * (1 - prediction)
        
        gradientsW[-1] = np.dot(activations[-1].T, dOutput)
        gradientsB[-1] = np.sum(dOutput, axis=0)
        
        dA = np.dot(dOutput, self.discriminatorWeights[-1].T)
        
        for l in range(len(self.discriminatorWeights) - 2, -1, -1):
            dZ = dA * (zs[l] > 0)
            dZ[zs[l] <= 0] *= 0.01
            gradientsW[l] = np.dot(activations[l].T, dZ)
            gradientsB[l] = np.sum(dZ, axis=0)
            dA = np.dot(dZ, self.discriminatorWeights[l].T)
        
        return gradientsW, gradientsB
    
    def updateParameters(self, gradientsW, gradientsB, generator=True):
        if generator:
            for i in range(len(self.generatorWeights)):
                self.generatorWeights[i] -= self.learningRate * gradientsW[i]
                self.generatorBiases[i] -= self.learningRate * gradientsB[i]
        else:
            for i in range(len(self.discriminatorWeights)):
                self.discriminatorWeights[i] -= self.learningRate * gradientsW[i]
                self.discriminatorBiases[i] -= self.learningRate * gradientsB[i]
    
    def fit(self, X):
        X = np.array(X)
        nSamples, inputDim = X.shape
        
        self.initializeGenerator(inputDim)
        self.initializeDiscriminator(inputDim)
        
        for epoch in range(self.maxEpochs):
            indices = np.random.permutation(nSamples)
            dLossEpoch = 0
            gLossEpoch = 0
            
            for i in range(0, nSamples, self.batchSize):
                realBatch = X[indices[i:i + self.batchSize]]
                
                z = np.random.normal(0, 1, (len(realBatch), self.latentDim))
                generatedBatch = self.generatorForward(z)
                
                realPredictions = self.discriminatorForward(realBatch)
                fakePredictions = self.discriminatorForward(generatedBatch)
                
                dLossReal = -np.mean(np.log(realPredictions + 1e-8))
                dLossFake = -np.mean(np.log(1 - fakePredictions + 1e-8))
                dLoss = dLossReal + dLossFake
                
                dGradientsW, dGradientsB = self.discriminatorBackward(realBatch, real=True)
                self.updateParameters(dGradientsW, dGradientsB, generator=False)
                
                dGradientsW, dGradientsB = self.discriminatorBackward(generatedBatch, real=False)
                self.updateParameters(dGradientsW, dGradientsB, generator=False)
                
                z = np.random.normal(0, 1, (len(realBatch), self.latentDim))
                generatedBatch = self.generatorForward(z)
                fakePredictions = self.discriminatorForward(generatedBatch)
                
                gLoss = -np.mean(np.log(fakePredictions + 1e-8))
                
                gGradientsW, gGradientsB = self.generatorBackward(z, -1/(fakePredictions + 1e-8))
                self.updateParameters(gGradientsW, gGradientsB, generator=True)
                
                dLossEpoch += dLoss
                gLossEpoch += gLoss
            
            self.dLossHistory.append(dLossEpoch / (nSamples // self.batchSize))
            self.gLossHistory.append(gLossEpoch / (nSamples // self.batchSize))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, D Loss: {self.dLossHistory[-1]:.4f}, G Loss: {self.gLossHistory[-1]:.4f}")
        
        return self
    
    def generate(self, nSamples):
        z = np.random.normal(0, 1, (nSamples, self.latentDim))
        return self.generatorForward(z)
    
    def discriminate(self, X):
        X = np.array(X)
        return self.discriminatorForward(X)