import numpy as np

class LSTM:
    def __init__(self, hiddenDim=64, learningRate=0.01, maxEpochs=100,
                 batchSize=32, randomState=None):
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.Wf = None
        self.Wi = None
        self.Wc = None
        self.Wo = None
        self.Uf = None
        self.Ui = None
        self.Uc = None
        self.Uo = None
        self.Wy = None
        self.bf = None
        self.bi = None
        self.bc = None
        self.bo = None
        self.by = None
        self.lossHistory = []
        self.accuracyHistory = []
        
    def initializeParameters(self, inputDim, outputDim):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.Wf = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Wi = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Wc = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Wo = np.random.randn(inputDim, self.hiddenDim) * 0.01
        self.Uf = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Ui = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Uc = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Uo = np.random.randn(self.hiddenDim, self.hiddenDim) * 0.01
        self.Wy = np.random.randn(self.hiddenDim, outputDim) * 0.01
        self.bf = np.zeros((1, self.hiddenDim))
        self.bi = np.zeros((1, self.hiddenDim))
        self.bc = np.zeros((1, self.hiddenDim))
        self.bo = np.zeros((1, self.hiddenDim))
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
        cellStates = np.zeros((batchSize, seqLen + 1, self.hiddenDim))
        fGates = np.zeros((batchSize, seqLen, self.hiddenDim))
        iGates = np.zeros((batchSize, seqLen, self.hiddenDim))
        oGates = np.zeros((batchSize, seqLen, self.hiddenDim))
        cCandidates = np.zeros((batchSize, seqLen, self.hiddenDim))
        outputs = np.zeros((batchSize, seqLen, self.Wy.shape[1]))
        
        for t in range(seqLen):
            xt = X[:, t, :]
            ft = self.sigmoid(np.dot(xt, self.Wf) + np.dot(hiddenStates[:, t, :], self.Uf) + self.bf)
            it = self.sigmoid(np.dot(xt, self.Wi) + np.dot(hiddenStates[:, t, :], self.Ui) + self.bi)
            ot = self.sigmoid(np.dot(xt, self.Wo) + np.dot(hiddenStates[:, t, :], self.Uo) + self.bo)
            ctCandidate = self.tanh(np.dot(xt, self.Wc) + np.dot(hiddenStates[:, t, :], self.Uc) + self.bc)
            
            cellStates[:, t + 1, :] = ft * cellStates[:, t, :] + it * ctCandidate
            hiddenStates[:, t + 1, :] = ot * self.tanh(cellStates[:, t + 1, :])
            
            fGates[:, t, :] = ft
            iGates[:, t, :] = it
            oGates[:, t, :] = ot
            cCandidates[:, t, :] = ctCandidate
            
            outputs[:, t, :] = np.dot(hiddenStates[:, t + 1, :], self.Wy) + self.by
        
        return hiddenStates, cellStates, fGates, iGates, oGates, cCandidates, outputs
    
    def computeLoss(self, outputs, y):
        batchSize, seqLen, outputDim = outputs.shape
        probs = self.softmax(outputs.reshape(-1, outputDim))
        targets = y.reshape(-1)
        logLikelihood = -np.log(probs[range(len(targets)), targets])
        return np.sum(logLikelihood) / (batchSize * seqLen)
    
    def backward(self, X, y, hiddenStates, cellStates, fGates, iGates, oGates, cCandidates, outputs):
        batchSize, seqLen, inputDim = X.shape
        outputDim = self.Wy.shape[1]
        
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dUf = np.zeros_like(self.Uf)
        dUi = np.zeros_like(self.Ui)
        dUc = np.zeros_like(self.Uc)
        dUo = np.zeros_like(self.Uo)
        dWy = np.zeros_like(self.Wy)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        dby = np.zeros_like(self.by)
        
        dhNext = np.zeros((batchSize, self.hiddenDim))
        dcNext = np.zeros((batchSize, self.hiddenDim))
        
        probs = self.softmax(outputs.reshape(-1, outputDim))
        targets = y.reshape(-1)
        dy = probs
        dy[range(len(targets)), targets] -= 1
        dy = dy.reshape(batchSize, seqLen, outputDim)
        
        for t in reversed(range(seqLen)):
            dWy += np.dot(hiddenStates[:, t + 1, :].T, dy[:, t, :])
            dby += np.sum(dy[:, t, :], axis=0, keepdims=True)
            
            dh = np.dot(dy[:, t, :], self.Wy.T) + dhNext
            do = dh * self.tanh(cellStates[:, t + 1, :])
            doRaw = do * oGates[:, t, :] * (1 - oGates[:, t, :])
            
            dc = dh * oGates[:, t, :] * (1 - self.tanh(cellStates[:, t + 1, :]) ** 2) + dcNext
            df = dc * cellStates[:, t, :]
            dfRaw = df * fGates[:, t, :] * (1 - fGates[:, t, :])
            
            di = dc * cCandidates[:, t, :]
            diRaw = di * iGates[:, t, :] * (1 - iGates[:, t, :])
            
            dcCandidate = dc * iGates[:, t, :]
            dcCandidateRaw = dcCandidate * (1 - cCandidates[:, t, :] ** 2)
            
            dWf += np.dot(X[:, t, :].T, dfRaw)
            dWi += np.dot(X[:, t, :].T, diRaw)
            dWc += np.dot(X[:, t, :].T, dcCandidateRaw)
            dWo += np.dot(X[:, t, :].T, doRaw)
            
            dUf += np.dot(hiddenStates[:, t, :].T, dfRaw)
            dUi += np.dot(hiddenStates[:, t, :].T, diRaw)
            dUc += np.dot(hiddenStates[:, t, :].T, dcCandidateRaw)
            dUo += np.dot(hiddenStates[:, t, :].T, doRaw)
            
            dbf += np.sum(dfRaw, axis=0, keepdims=True)
            dbi += np.sum(diRaw, axis=0, keepdims=True)
            dbc += np.sum(dcCandidateRaw, axis=0, keepdims=True)
            dbo += np.sum(doRaw, axis=0, keepdims=True)
            
            dhPrev = (np.dot(dfRaw, self.Uf.T) + np.dot(diRaw, self.Ui.T) + 
                     np.dot(dcCandidateRaw, self.Uc.T) + np.dot(doRaw, self.Uo.T))
            dcPrev = dc * fGates[:, t, :]
            
            dhNext = dhPrev
            dcNext = dcPrev
        
        for d in [dWf, dWi, dWc, dWo, dUf, dUi, dUc, dUo, dWy, dbf, dbi, dbc, dbo, dby]:
            np.clip(d, -5, 5, out=d)
        
        return dWf, dWi, dWc, dWo, dUf, dUi, dUc, dUo, dWy, dbf, dbi, dbc, dbo, dby
    
    def updateParameters(self, *grads):
        params = [self.Wf, self.Wi, self.Wc, self.Wo, self.Uf, self.Ui, self.Uc, self.Uo, 
                 self.Wy, self.bf, self.bi, self.bc, self.bo, self.by]
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
                
                hiddenStates, cellStates, fGates, iGates, oGates, cCandidates, outputs = self.forward(XBatch)
                loss = self.computeLoss(outputs, yBatch)
                epochLoss += loss * len(XBatch)
                
                preds = np.argmax(outputs, axis=2)
                correctPredictions += np.sum(preds == yBatch)
                
                grads = self.backward(XBatch, yBatch, hiddenStates, cellStates, fGates, iGates, oGates, cCandidates, outputs)
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
        _, _, _, _, _, _, outputs = self.forward(X)
        return np.argmax(outputs, axis=2)
    
    def predictProba(self, X):
        X = np.array(X)
        _, _, _, _, _, _, outputs = self.forward(X)
        return self.softmax(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)