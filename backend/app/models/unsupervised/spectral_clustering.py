import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform

class SpectralClustering:
    def __init__(self, n_clusters=8, affinity='rbf', gamma=1.0, 
                 random_state=None, n_neighbors=10):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        
        self.labels_ = None
        self.affinity_matrix_ = None
        self.embedding_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_affinity_matrix(self, X):
        n_samples = X.shape[0]
        
        if self.affinity == 'rbf':
            distances = squareform(pdist(X, metric='euclidean'))
            self.affinity_matrix_ = np.exp(-self.gamma * distances ** 2)
            np.fill_diagonal(self.affinity_matrix_, 0)
        
        elif self.affinity == 'nearest_neighbors':
            from sklearn.neighbors import kneighbors_graph
            self.affinity_matrix_ = kneighbors_graph(
                X, self.n_neighbors, mode='connectivity', include_self=True
            ).toarray()
        
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        
        else:
            raise ValueError(f"Unsupported affinity: {self.affinity}")
        
        return self.affinity_matrix_
    
    def _compute_laplacian(self, affinity):
        # Compute degree matrix
        degree = np.sum(affinity, axis=1)
        D = np.diag(degree)
        
        # Compute normalized Laplacian
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degree))
        L = np.eye(affinity.shape[0]) - D_sqrt_inv @ affinity @ D_sqrt_inv
        
        return L
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Compute affinity matrix
        affinity = self._compute_affinity_matrix(X)
        
        # Compute Laplacian
        laplacian = self._compute_laplacian(affinity)
        
        # Compute eigenvectors
        if n_samples < 1000:
            eigenvalues, eigenvectors = eigh(laplacian)
        else:
            eigenvalues, eigenvectors = eigsh(laplacian, k=self.n_clusters, which='SM')
        
        # Get the first n_clusters eigenvectors
        indices = np.argsort(eigenvalues)[:self.n_clusters]
        self.embedding_ = eigenvectors[:, indices]
        
        # Normalize rows
        row_sums = np.linalg.norm(self.embedding_, axis=1)
        self.embedding_ = self.embedding_ / row_sums[:, np.newaxis]
        
        # Cluster using K-means
        from .kmeans import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(self.embedding_)
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'affinity': self.affinity,
            'gamma': self.gamma,
            'n_neighbors': self.n_neighbors,
            'embedding_shape': self.embedding_.shape if self.embedding_ is not None else None
        }