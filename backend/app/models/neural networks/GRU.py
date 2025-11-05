import numpy as np

class GRU:
    def __init__(self, hiddenDim=64, learningRate=0.01, maxEpochs=100,
                 batchSize=32, randomState=None):
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.Wz = None
        self.Wr = None
        self.Wh = None
        self.Uz = None
        self.Ur = None
        self.Uh = None
        self.Wy = None
        self.bz = None
        self.br = None
        self.bh = None
        self.by = None
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputDim, outputDim):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.Wz = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Wr = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Wh = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Uz = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Ur = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Uh = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Wy = np.random.randn(self.hiddenDim, outputDim) * 0.01
        self.bz = np.zeros((1, self.hiddenDim))
        self.br = np.zeros((1, self.hiddenDim))
        self.bh = np.zeros((1, self.hiddenDim))
        self.by = np.zeros((1, outputDim))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        batchSize, seqLen, inputDim = X.shape
        hiddenStates = np.zeros((batchSize, seqLen + 1, self.hiddenDim))
        zGates = np.zeros((batchSize, seqLen, self.hiddenDim))
        rGates = np.zeros((batchSize, seqLen, self.hiddenDim))
        hCandidates = np.zeros((batchSize, seqLen, self.hiddenDim))
        outputs = np.zeros((batchSize, seqLen, self.Wy.shape[1]))
        
        for t in range(seqLen):
            xt = X[:, t, :]
            zt = self.sigmoid(np.dot(xt, self.Wz) + np.dot(hiddenStates[:, t, :], self.Uz) + self.bz)
            rt = self.sigmoid(np.dot(xt, self.Wr) + np.dot(hiddenStates[:, t, :], self.Ur) + self.br)
            htCandidate = self.tanh(np.dot(xt, self.Wh) + np.dot(rt * hiddenStates[:, t, :], self.Uh) + self.bh)
            hiddenStates[:, t + 1, :] = (1 - zt) * hiddenStates[:, t, :] + zt * htCandidate
            
            zGates[:, t, :] = zt
            rGates[:, t, :] = rt
            hCandidates[:, t, :] = htCandidate
            
            outputs[:, t, :] = np.dot(hiddenStates[:, t + 1, :], self.Wy) + self.by
        
        return hiddenStates, zGates, rGates, hCandidates, outputs
    
    def computeLoss(self, outputs, y):
        batchSize, seqLen, outputDim = outputs.shape
        probs = self.softmax(outputs.reshape(-1, outputDim))
        targets = y.reshape(-1)
        logLikelihood = -np.log(probs[range(len(targets)), targets])
        return np.sum(logLikelihood) / (batchSize * seqLen)
    
    def backward(self, X, y, hiddenStates, zGates, rGates, hCandidates, outputs):
        batchSize, seqLen, inputDim = X.shape
        outputDim = self.Wy.shape[1]
        
        dWz, dWr, dWh = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh)
        dUz, dUr, dUh = np.zeros_like(self.Uz), np.zeros_like(self.Ur), np.zeros_like(self.Uh)
        dWy = np.zeros_like(self.Wy)
        dbz, dbr, dbh, dby = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh), np.zeros_like(self.by)
        
        dhNext = np.zeros((batchSize, self.hiddenDim))
        
        probs = self.softmax(outputs.reshape(-1, outputDim))
        targets = y.reshape(-1)
        dy = probs
        dy[range(len(targets)), targets] -= 1
        dy = dy.reshape(batchSize, seqLen, outputDim)
        
        for t in reversed(range(seqLen)):
            dWy += np.dot(hiddenStates[:, t + 1, :].T, dy[:, t, :])
            dby += np.sum(dy[:, t, :], axis=0, keepdims=True)
            
            dh = np.dot(dy[:, t, :], self.Wy.T) + dhNext
            
            dhtCandidate = dh * zGates[:, t, :]
            dzt = dh * (hCandidates[:, t, :] - hiddenStates[:, t, :])
            dhPrev1 = dh * (1 - zGates[:, t, :])
            dhPrev2 = dzt * (1 - zGates[:, t, :]) * zGates[:, t, :]
            
            dhtCandidateRaw = dhtCandidate * (1 - hCandidates[:, t, :] ** 2)
            dztRaw = dzt * zGates[:, t, :] * (1 - zGates[:, t, :])
            drtRaw = np.dot(dhtCandidateRaw, self.Uh.T) * hiddenStates[:, t, :] * rGates[:, t, :] * (1 - rGates[:, t, :])
            
            dWh += np.dot(X[:, t, :].T, dhtCandidateRaw)
            dUh += np.dot((rGates[:, t, :] * hiddenStates[:, t, :]).T, dhtCandidateRaw)
            dbh += np.sum(dhtCandidateRaw, axis=0, keepdims=True)
            
            dWz += np.dot(X[:, t, :].T, dztRaw)
            dUz += np.dot(hiddenStates[:, t, :].T, dztRaw)
            dbz += np.sum(dztRaw, axis=0, keepdims=True)
            
            dWr += np.dot(X[:, t, :].T, drtRaw)
            dUr += np.dot(hiddenStates[:, t, :].T, drtRaw)
            dbr += np.sum(drtRaw, axis=0, keepdims=True)
            
            dhNext = dhPrev1 + np.dot(dztRaw, self.Uz.T) + np.dot(drtRaw, self.Ur.T) + np.dot(dhtCandidateRaw, self.Uh.T) * rGates[:, t, :]
        
        for d in [dWz, dWr, dWh, dUz, dUr, dUh, dWy, dbz, dbr, dbh, dby]:
            np.clip(d, -5, 5, out=d)
        
        return dWz, dWr, dWh, dUz, dUr, dUh, dWy, dbz, dbr, dbh, dby
    
    def updateParameters(self, *grads):
        params = [self.Wz, self.Wr, self.Wh, self.Uz, self.Ur, self.Uh, self.Wy, 
                 self.bz, self.br, self.bh, self.by]
        for i in range(len(params)):
            params[i] -= self.learningRate * grads[i]
    
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
                
                hiddenStates, zGates, rGates, hCandidates, outputs = self.forward(XBatch)
                loss = self.computeLoss(outputs, yBatch)
                epochLoss += loss * len(XBatch)
                
                preds = np.argmax(outputs, axis=2)
                correctPredictions += np.sum(preds == yBatch)
                
                grads = self.backward(XBatch, yBatch, hiddenStates, zGates, rGates, hCandidates, outputs)
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
        _, _, _, _, outputs = self.forward(X)
        return np.argmax(outputs, axis=2)
    
    def predictProba(self, X):
        X = np.array(X)
        _, _, _, _, outputs = self.forward(X)
        return self.softmax(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)