import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class KNNClassifier:
    def __init__(self, nNeighbors=5, weights='uniform', metric='minkowski', p=2, algorithm='auto'):
        self.nNeighbors = nNeighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.XTrain = None
        self.yTrain = None
        self.classes = None
        self.isFitted = False
    
    def computeDistance(self, X1, X2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((X1 - X2) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X1 - X2), axis=1)
        elif self.metric == 'minkowski':
            return np.sum(np.abs(X1 - X2) ** self.p, axis=1) ** (1 / self.p)
        elif self.metric == 'cosine':
            dotProduct = np.sum(X1 * X2, axis=1)
            norm1 = np.sqrt(np.sum(X1 ** 2, axis=1))
            norm2 = np.sqrt(np.sum(X2 ** 2, axis=1))
            return 1 - dotProduct / (norm1 * norm2 + 1e-8)
        else:
            return cdist(X1, X2, metric=self.metric).diagonal()
    
    def findNeighbors(self, x):
        distances = self.computeDistance(self.XTrain, x.reshape(1, -1))
        
        if self.nNeighbors >= len(distances):
            neighborIndices = np.argsort(distances)
        else:
            neighborIndices = np.argpartition(distances, self.nNeighbors)[:self.nNeighbors]
            neighborIndices = neighborIndices[np.argsort(distances[neighborIndices])]
        
        return neighborIndices, distances[neighborIndices]
    
    def computeWeights(self, distances):
        if self.weights == 'uniform':
            return np.ones(len(distances))
        elif self.weights == 'distance':
            weights = 1 / (distances + 1e-8)
            return weights / np.sum(weights)
        elif self.weights == 'inverse':
            weights = 1 / (1 + distances)
            return weights / np.sum(weights)
        else:
            return np.ones(len(distances))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        self.XTrain = X.copy()
        self.yTrain = y.copy()
        self.classes = np.unique(y)
        self.isFitted = True
        
        return self
    
    def predictSingle(self, x):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        neighborIndices, distances = self.findNeighbors(x)
        neighborLabels = self.yTrain[neighborIndices]
        
        if self.weights == 'uniform':
            mostCommon = Counter(neighborLabels).most_common(1)
            return mostCommon[0][0]
        else:
            weights = self.computeWeights(distances)
            weightedVotes = {}
            
            for label, weight in zip(neighborLabels, weights):
                if label in weightedVotes:
                    weightedVotes[label] += weight
                else:
                    weightedVotes[label] = weight
            
            return max(weightedVotes.items(), key=lambda x: x[1])[0]
    
    def predict(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return np.array([self.predictSingle(x) for x in X])
    
    def predictProbaSingle(self, x):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        neighborIndices, distances = self.findNeighbors(x)
        neighborLabels = self.yTrain[neighborIndices]
        
        if self.weights == 'uniform':
            labelCounts = Counter(neighborLabels)
            total = len(neighborLabels)
            probabilities = {label: count / total for label, count in labelCounts.items()}
        else:
            weights = self.computeWeights(distances)
            weightedCounts = {}
            
            for label, weight in zip(neighborLabels, weights):
                if label in weightedCounts:
                    weightedCounts[label] += weight
                else:
                    weightedCounts[label] = weight
            
            totalWeight = sum(weightedCounts.values())
            probabilities = {label: weight / totalWeight for label, weight in weightedCounts.items()}
        
        probaArray = np.zeros(len(self.classes))
        for i, cls in enumerate(self.classes):
            probaArray[i] = probabilities.get(cls, 0.0)
        
        return probaArray
    
    def predictProba(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        probas = np.array([self.predictProbaSingle(x) for x in X])
        return probas
    
    def predictLogProba(self, X):
        probabilities = self.predictProba(X)
        return np.log(probabilities + 1e-8)
    
    def kneighbors(self, X, nNeighbors=None):
        if not self.isFitted:
            raise ValueError("Model must be fitted before finding neighbors")
        
        if nNeighbors is None:
            nNeighbors = self.nNeighbors
        
        X = np.array(X)
        distances = []
        indices = []
        
        for x in X:
            dist = self.computeDistance(self.XTrain, x.reshape(1, -1))
            if nNeighbors >= len(dist):
                neighborIndices = np.argsort(dist)
                neighborDistances = dist[neighborIndices]
            else:
                neighborIndices = np.argpartition(dist, nNeighbors)[:nNeighbors]
                sortedIndices = np.argsort(dist[neighborIndices])
                neighborIndices = neighborIndices[sortedIndices]
                neighborDistances = dist[neighborIndices]
            
            indices.append(neighborIndices)
            distances.append(neighborDistances)
        
        return np.array(distances), np.array(indices)
    
    def kneighborsGraph(self, X, mode='connectivity', nNeighbors=None):
        if not self.isFitted:
            raise ValueError("Model must be fitted before computing graph")
        
        if nNeighbors is None:
            nNeighbors = self.nNeighbors
        
        X = np.array(X)
        nSamples = len(X)
        graph = np.zeros((nSamples, len(self.XTrain)))
        
        distances, indices = self.kneighbors(X, nNeighbors)
        
        for i in range(nSamples):
            if mode == 'connectivity':
                graph[i, indices[i]] = 1
            elif mode == 'distance':
                graph[i, indices[i]] = distances[i]
        
        return graph
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def getParams(self):
        return {
            'nNeighbors': self.nNeighbors,
            'weights': self.weights,
            'metric': self.metric,
            'p': self.p,
            'algorithm': self.algorithm,
            'nClasses': len(self.classes) if self.classes is not None else 0,
            'nSamples': len(self.XTrain) if self.XTrain is not None else 0,
            'nFeatures': self.XTrain.shape[1] if self.XTrain is not None else 0
        }
    
    def getNeighborInfo(self, X):
        if not self.isFitted:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        neighborInfo = []
        
        for x in X:
            neighborIndices, distances = self.findNeighbors(x)
            neighborLabels = self.yTrain[neighborIndices]
            
            info = {
                'neighborIndices': neighborIndices,
                'distances': distances,
                'neighborLabels': neighborLabels,
                'weights': self.computeWeights(distances) if self.weights != 'uniform' else np.ones(len(distances))
            }
            neighborInfo.append(info)
        
        return neighborInfo