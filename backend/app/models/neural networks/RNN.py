import numpy as np

class RNN:
    def __init__(self, hiddenDim=64, learningRate=0.01, maxEpochs=100, 
                 batchSize=32, randomState=None):
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.Wxh = None
        self.Whh = None
        self.Why = None
        self.bh = None
        self.by = None
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputDim, outputDim):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.Wxh = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Whh = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Why = np.random.randn(self.hiddenDim, outputDim) * 0.01
        self.bh = np.zeros((1, self.hiddenDim))
        self.by = np.zeros((1, outputDim))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        batchSize, seqLen, inputDim = X.shape
        hiddenStates = np.zeros((batchSize, seqLen + 1, self.hiddenDim))
        outputs = np.zeros((batchSize, seqLen, self.Why.shape[1]))
        
        for t in range(seqLen):
            xt = X[:, t, :]
            hiddenStates[:, t + 1, :] = self.tanh(
                np.dot(xt, self.Wxh) + 
                np.dot(hiddenStates[:, t, :], self.Whh) + 
                self.bh
            )
            outputs[:, t, :] = np.dot(hiddenStates[:, t + 1, :], self.Why) + self.by
        
        return hiddenStates, outputs
    
    def computeLoss(self, outputs, y):
        batchSize, seqLen, outputDim = outputs.shape
        probs = self.softmax(outputs.reshape(-1, outputDim))
        targets = y.reshape(-1)
        logLikelihood = -np.log(probs[range(len(targets)), targets])
        return np.sum(logLikelihood) / (batchSize * seqLen)
    
    def backward(self, X, y, hiddenStates, outputs):
        batchSize, seqLen, inputDim = X.shape
        outputDim = self.Why.shape[1]
        
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhNext = np.zeros((batchSize, self.hiddenDim))
        
        probs = self.softmax(outputs.reshape(-1, outputDim))
        targets = y.reshape(-1)
        dy = probs
        dy[range(len(targets)), targets] -= 1
        dy = dy.reshape(batchSize, seqLen, outputDim)
        
        for t in reversed(range(seqLen)):
            dWhy += np.dot(hiddenStates[:, t + 1, :].T, dy[:, t, :])
            dby += np.sum(dy[:, t, :], axis=0, keepdims=True)
            
            dh = np.dot(dy[:, t, :], self.Why.T) + dhNext
            dhRaw = (1 - hiddenStates[:, t + 1, :] ** 2) * dh
            
            dbh += np.sum(dhRaw, axis=0, keepdims=True)
            dWxh += np.dot(X[:, t, :].T, dhRaw)
            dWhh += np.dot(hiddenStates[:, t, :].T, dhRaw)
            
            dhNext = np.dot(dhRaw, self.Whh.T)
        
        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -5, 5, out=d)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def updateParameters(self, dWxh, dWhh, dWhy, dbh, dby):
        self.Wxh -= self.learningRate * dWxh
        self.Whh -= self.learningRate * dWhh
        self.Why -= self.learningRate * dWhy
        self.bh -= self.learningRate * dbh
        self.by -= self.learningRate * dby
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples, seqLen, inputDim = X.shape
        outputDim = len(np.unique(y))
        
        self.initializeParameters(inputDim, outputDim)
        
        for epoch in range(self.maxEpochs):
            indices = np.random.permutation(nSamples)
            epochLoss = 0
            correctPredictions = 0
            
            for i in range(0, nSamples, self.batchSize):
                XBatch = X[indices[i:i + self.batchSize]]
                yBatch = y[indices[i:i + self.batchSize]]
                
                hiddenStates, outputs = self.forward(XBatch)
                loss = self.computeLoss(outputs, yBatch)
                epochLoss += loss * len(XBatch)
                
                preds = np.argmax(outputs, axis=2)
                correctPredictions += np.sum(preds == yBatch)
                
                grads = self.backward(XBatch, yBatch, hiddenStates, outputs)
                self.updateParameters(*grads)
            
            avgLoss = epochLoss / nSamples
            accuracy = correctPredictions / (nSamples * seqLen)
            
            self.lossHistory.append(avgLoss)
            self.accuracyHistory.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avgLoss:.4f}, Accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        _, outputs = self.forward(X)
        return np.argmax(outputs, axis=2)
    
    def predictProba(self, X):
        X = np.array(X)
        _, outputs = self.forward(X)
        return self.softmax(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)