import numpy as np

class Transformer:
    def __init__(self, vocabSize=1000, dModel=512, nHeads=8, dFf=2048, nLayers=6,
                 maxSeqLength=100, learningRate=0.001, maxEpochs=100, 
                 batchSize=32, randomState=None):
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.nHeads = nHeads
        self.dFf = dFf
        self.nLayers = nLayers
        self.maxSeqLength = maxSeqLength
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize
        self.randomState = randomState
        
        self.embedding = None
        self.encoderLayers = []
        self.decoderLayers = []
        self.lossHistory = []
        
    def initializeParameters(self):
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        self.embedding = np.random.randn(self.vocabSize, self.dModel) * 0.01
        
        self.encoderLayers = []
        self.decoderLayers = []
        
        for _ in range(self.nLayers):
            encoderLayer = {
                'multiheadQ': np.random.randn(self.dModel, self.dModel) * 0.01,
                'multiheadK': np.random.randn(self.dModel, self.dModel) * 0.01,
                'multiheadV': np.random.randn(self.dModel, self.dModel) * 0.01,
                'multiheadOut': np.random.randn(self.dModel, self.dModel) * 0.01,
                'ffn1': np.random.randn(self.dModel, self.dFf) * 0.01,
                'ffn2': np.random.randn(self.dFf, self.dModel) * 0.01
            }
            self.encoderLayers.append(encoderLayer)
            
            decoderLayer = {
                'maskedMultiheadQ': np.random.randn(self.dModel, self.dModel) * 0.01,
                'maskedMultiheadK': np.random.randn(self.dModel, self.dModel) * 0.01,
                'maskedMultiheadV': np.random.randn(self.dModel, self.dModel) * 0.01,
                'maskedMultiheadOut': np.random.randn(self.dModel, self.dModel) * 0.01,
                'encoderDecoderQ': np.random.randn(self.dModel, self.dModel) * 0.01,
                'encoderDecoderK': np.random.randn(self.dModel, self.dModel) * 0.01,
                'encoderDecoderV': np.random.randn(self.dModel, self.dModel) * 0.01,
                'encoderDecoderOut': np.random.randn(self.dModel, self.dModel) * 0.01,
                'ffn1': np.random.randn(self.dModel, self.dFf) * 0.01,
                'ffn2': np.random.randn(self.dFf, self.dModel) * 0.01
            }
            self.decoderLayers.append(decoderLayer)
        
        self.outputWeights = np.random.randn(self.dModel, self.vocabSize) * 0.01
        self.outputBias = np.zeros(self.vocabSize)
    
    def positionalEncoding(self, seqLength):
        positions = np.arange(seqLength)[:, np.newaxis]
        dimensions = np.arange(self.dModel)[np.newaxis, :]
        
        angleRates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(self.dModel))
        angleRads = positions * angleRates
        
        angleRads[:, 0::2] = np.sin(angleRads[:, 0::2])
        angleRads[:, 1::2] = np.cos(angleRads[:, 1::2])
        
        return angleRads
    
    def scaledDotProductAttention(self, Q, K, V, mask=None):
        dk = K.shape[-1]
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(dk)
        
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        attention = self.softmax(scores, axis=-1)
        output = np.matmul(attention, V)
        return output, attention
    
    def multiHeadAttention(self, Q, K, V, weights, mask=None):
        batchSize, seqLength, dModel = Q.shape
        
        Q = np.matmul(Q, weights['Q'])
        K = np.matmul(K, weights['K'])
        V = np.matmul(V, weights['V'])
        
        Q = Q.reshape(batchSize, seqLength, self.nHeads, dModel // self.nHeads).transpose(0,2,1,3)
        K = K.reshape(batchSize, seqLength, self.nHeads, dModel // self.nHeads).transpose(0,2,1,3)
        V = V.reshape(batchSize, seqLength, self.nHeads, dModel // self.nHeads).transpose(0,2,1,3)
        
        attentionOutput, attentionWeights = self.scaledDotProductAttention(Q, K, V, mask)
        
        attentionOutput = attentionOutput.transpose(0,2,1,3).reshape(batchSize, seqLength, dModel)
        output = np.matmul(attentionOutput, weights['out'])
        
        return output, attentionWeights
    
    def feedForward(self, x, weights):
        ffn1 = np.maximum(0, np.matmul(x, weights['ffn1']))
        ffn2 = np.matmul(ffn1, weights['ffn2'])
        return ffn2
    
    def layerNorm(self, x, epsilon=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon)
    
    def softmax(self, x, axis=-1):
        expX = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return expX / np.sum(expX, axis=axis, keepdims=True)
    
    def encoderLayer(self, x, layerWeights):
        attnWeights = {
            'Q': layerWeights['multiheadQ'],
            'K': layerWeights['multiheadK'],
            'V': layerWeights['multiheadV'],
            'out': layerWeights['multiheadOut']
        }
        
        attentionOutput, _ = self.multiHeadAttention(x, x, x, attnWeights)
        x = self.layerNorm(x + attentionOutput)
        
        ffnOutput = self.feedForward(x, layerWeights)
        x = self.layerNorm(x + ffnOutput)
        
        return x
    
    def decoderLayer(self, x, encoderOutput, layerWeights, lookAheadMask):
        attnWeights1 = {
            'Q': layerWeights['maskedMultiheadQ'],
            'K': layerWeights['maskedMultiheadK'],
            'V': layerWeights['maskedMultiheadV'],
            'out': layerWeights['maskedMultiheadOut']
        }
        
        maskedAttentionOutput, _ = self.multiHeadAttention(x, x, x, attnWeights1, lookAheadMask)
        x = self.layerNorm(x + maskedAttentionOutput)
        
        attnWeights2 = {
            'Q': layerWeights['encoderDecoderQ'],
            'K': layerWeights['encoderDecoderK'],
            'V': layerWeights['encoderDecoderV'],
            'out': layerWeights['encoderDecoderOut']
        }
        
        encoderDecoderOutput, _ = self.multiHeadAttention(x, encoderOutput, encoderOutput, attnWeights2)
        x = self.layerNorm(x + encoderDecoderOutput)
        
        ffnOutput = self.feedForward(x, layerWeights)
        x = self.layerNorm(x + ffnOutput)
        
        return x
    
    def createPaddingMask(self, seq):
        return (seq == 0)[:, np.newaxis, np.newaxis, :]
    
    def createLookAheadMask(self, size):
        mask = np.triu(np.ones((size, size)), k=1)
        return mask
    
    def forward(self, encoderInput, decoderInput, training=True):
        batchSize, encoderSeqLength = encoderInput.shape
        _, decoderSeqLength = decoderInput.shape
        
        encoderEmbeddings = self.embedding[encoderInput] * np.sqrt(self.dModel)
        encoderEmbeddings += self.positionalEncoding(encoderSeqLength)
        
        decoderEmbeddings = self.embedding[decoderInput] * np.sqrt(self.dModel)
        decoderEmbeddings += self.positionalEncoding(decoderSeqLength)
        
        encoderPaddingMask = self.createPaddingMask(encoderInput)
        
        encoderOutput = encoderEmbeddings
        for layer in self.encoderLayers:
            encoderOutput = self.encoderLayer(encoderOutput, layer)
        
        lookAheadMask = self.createLookAheadMask(decoderSeqLength)
        decoderPaddingMask = self.createPaddingMask(encoderInput)
        combinedMask = np.maximum(lookAheadMask, decoderPaddingMask) if decoderPaddingMask is not None else lookAheadMask
        
        decoderOutput = decoderEmbeddings
        for layer in self.decoderLayers:
            decoderOutput = self.decoderLayer(decoderOutput, encoderOutput, layer, combinedMask)
        
        logits = np.matmul(decoderOutput, self.outputWeights) + self.outputBias
        return logits
    
    def computeLoss(self, logits, targets):
        logProbs = -np.log(self.softmax(logits.reshape(-1, self.vocabSize))[range(len(targets.flatten())), targets.flatten()])
        return np.mean(logProbs)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        nSamples = X.shape[0]
        
        self.initializeParameters()
        
        for epoch in range(self.maxEpochs):
            indices = np.random.permutation(nSamples)
            epochLoss = 0
            
            for i in range(0, nSamples, self.batchSize):
                XBatch = X[indices[i:i + self.batchSize]]
                yBatch = y[indices[i:i + self.batchSize]]
                
                logits = self.forward(XBatch, yBatch[:, :-1])
                loss = self.computeLoss(logits, yBatch[:, 1:])
                epochLoss += loss * len(XBatch)
            
            avgLoss = epochLoss / nSamples
            self.lossHistory.append(avgLoss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avgLoss:.4f}")
        
        return self
    
    def predict(self, X, maxLength=50):
        X = np.array(X)
        batchSize = X.shape[0]
        
        encoderOutput = self.embedding[X] * np.sqrt(self.dModel)
        encoderOutput += self.positionalEncoding(X.shape[1])
        
        for layer in self.encoderLayers:
            encoderOutput = self.encoderLayer(encoderOutput, layer)
        
        decoderInput = np.ones((batchSize, 1), dtype=int)
        outputSequence = []
        
        for i in range(maxLength):
            decoderEmbeddings = self.embedding[decoderInput] * np.sqrt(self.dModel)
            decoderEmbeddings += self.positionalEncoding(decoderInput.shape[1])
            
            lookAheadMask = self.createLookAheadMask(decoderInput.shape[1])
            
            decoderOutput = decoderEmbeddings
            for layer in self.decoderLayers:
                decoderOutput = self.decoderLayer(decoderOutput, encoderOutput, layer, lookAheadMask)
            
            logits = np.matmul(decoderOutput[:, -1, :], self.outputWeights) + self.outputBias
            nextToken = np.argmax(logits, axis=-1)
            
            outputSequence.append(nextToken)
            decoderInput = np.concatenate([decoderInput, nextToken.reshape(-1, 1)], axis=1)
        
        return np.array(outputSequence).T