import numpy as np
from sklearn.neighbors import NearestNeighbors

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        
    def _find_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1 for unvisited/noise
        cluster_id = 0
        self.core_sample_indices_ = []
        
        # Precompute neighbors for efficiency
        neighbors = []
        for i in range(n_samples):
            neighbors.append(self._find_neighbors(X, i))
        
        for i in range(n_samples):
            if self.labels_[i] != -1:  # Already visited
                continue
                
            # Find neighbors
            point_neighbors = neighbors[i]
            
            if len(point_neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
                continue
            
            # Start new cluster
            self.labels_[i] = cluster_id
            self.core_sample_indices_.append(i)
            
            # Expand cluster
            seed_set = set(point_neighbors) - {i}
            while seed_set:
                j = seed_set.pop()
                
                if self.labels_[j] == -1:  # Noise point
                    self.labels_[j] = cluster_id
                
                if self.labels_[j] != -1:  # Already assigned
                    continue
                
                self.labels_[j] = cluster_id
                j_neighbors = neighbors[j]
                
                if len(j_neighbors) >= self.min_samples:
                    self.core_sample_indices_.append(j)
                    seed_set.update(j_neighbors)
            
            cluster_id += 1
        
        self.core_sample_indices_ = np.array(self.core_sample_indices_)
        if len(self.core_sample_indices_) > 0:
            self.components_ = X[self.core_sample_indices_]
        else:
            self.components_ = np.array([])
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def get_params(self):
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'n_clusters': len(np.unique(self.labels_[self.labels_ != -1])) if self.labels_ is not None else 0,
            'n_noise': np.sum(self.labels_ == -1) if self.labels_ is not None else 0,
            'n_core_samples': len(self.core_sample_indices_) if self.core_sample_indices_ is not None else 0
        }