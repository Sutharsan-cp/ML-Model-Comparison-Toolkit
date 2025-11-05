import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import numba

class UMAP:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, 
                 metric='euclidean', random_state=None, learning_rate=1.0,
                 n_epochs=500, spread=1.0, low_memory=False):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.spread = spread
        self.low_memory = low_memory
        
        self.embedding_ = None
        self.graph_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_fuzzy_simplicial_set(self, X):
        """Compute the fuzzy simplicial set (graph) from data"""
        n_samples = X.shape[0]
        
        # Find nearest neighbors
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        
        # Compute fuzzy simplicial set (graph)
        graph = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # Compute local connectivity (rho)
            rho = distances[i, 1]  # Distance to first neighbor (excluding self)
            
            # Compute probabilities
            for j in range(1, self.n_neighbors):  # Skip self
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                
                # Compute probability based on local connectivity
                if distance <= rho:
                    probability = 1.0
                else:
                    probability = np.exp(-(distance - rho) / rho)
                
                graph[i, neighbor_idx] = probability
                graph[neighbor_idx, i] = probability
        
        # Symmetrize and normalize
        graph = np.maximum(graph, graph.T)
        
        return graph
    
    def _compute_low_dimensional_probabilities(self, Y):
        """Compute probabilities in low-dimensional space"""
        n_samples = Y.shape[0]
        
        # Compute pairwise distances
        distances = squareform(pdist(Y, metric='euclidean'))
        
        # Apply transformation
        a = 1.0 / self.spread
        b = 1.0
        probabilities = 1.0 / (1.0 + a * distances ** (2 * b))
        np.fill_diagonal(probabilities, 0)
        
        return probabilities
    
    def _optimize_embedding(self, graph, Y):
        """Optimize the low-dimensional embedding"""
        n_samples = Y.shape[0]
        
        # Precompute some constants
        a = 1.0 / self.spread
        b = 1.0
        min_dist_sq = self.min_dist ** 2
        
        for epoch in range(self.n_epochs):
            # Compute low-dimensional probabilities
            Q = self._compute_low_dimensional_probabilities(Y)
            
            # Compute gradient
            gradient = np.zeros_like(Y)
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    # High-dimensional probability
                    p_ij = graph[i, j]
                    q_ij = Q[i, j]
                    
                    # Distance in low dimension
                    dist_ij = np.linalg.norm(Y[i] - Y[j])
                    dist_sq = dist_ij ** 2
                    
                    # Compute gradient contribution
                    if p_ij > 0:
                        # Attractive force
                        attractive_strength = p_ij * (1.0 / (1.0 + dist_sq))
                    else:
                        attractive_strength = 0.0
                    
                    # Repulsive force (approximated)
                    repulsive_strength = q_ij ** 2 / (1.0 + dist_sq)
                    
                    # Combined force
                    force = attractive_strength - repulsive_strength
                    
                    # Update gradient
                    if dist_ij > 0:
                        direction = (Y[i] - Y[j]) / dist_ij
                        gradient[i] += force * direction
                        gradient[j] -= force * direction
            
            # Update embedding
            Y -= self.learning_rate * gradient
            
            # Reduce learning rate
            self.learning_rate *= 0.99
        
        return Y
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Compute fuzzy simplicial set
        self.graph_ = self._compute_fuzzy_simplicial_set(X)
        
        # Initialize embedding
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.embedding_ = np.random.normal(0, 1, (n_samples, self.n_components)) * 0.0001
        
        # Optimize embedding
        self.embedding_ = self._optimize_embedding(self.graph_, self.embedding_)
        
        return self
    
    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
    
    def transform(self, X):
        # Simplified transform - in practice, this would require more sophisticated methods
        if self.embedding_ is None:
            raise ValueError("Model must be fitted before transform")
        
        # For simplicity, return random embedding
        # A proper implementation would use the trained model to embed new points
        n_samples = X.shape[0]
        return np.random.normal(0, 1, (n_samples, self.n_components))
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'spread': self.spread,
            'embedding_shape': self.embedding_.shape if self.embedding_ is not None else None
        }