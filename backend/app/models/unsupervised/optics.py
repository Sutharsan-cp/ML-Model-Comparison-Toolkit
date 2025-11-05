import numpy as np
from sklearn.neighbors import NearestNeighbors

class OPTICS:
    def __init__(self, min_samples=5, max_eps=np.inf, metric='euclidean', 
                 cluster_method='xi', xi=0.05, min_cluster_size=None):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.cluster_method = cluster_method
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        
        self.labels_ = None
        self.reachability_ = None
        self.core_distances_ = None
        self.predecessor_ = None
        self.ordering_ = None
        
    def _core_distance(self, distances, min_samples):
        if len(distances) < min_samples:
            return np.inf
        return distances[min_samples - 1]
    
    def _compute_neighbors(self, X):
        nbrs = NearestNeighbors(n_neighbors=self.min_samples, metric=self.metric)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        return distances, indices
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Compute neighbors and core distances
        distances, indices = self._compute_neighbors(X)
        self.core_distances_ = np.array([
            self._core_distance(distances[i], self.min_samples) 
            for i in range(n_samples)
        ])
        
        # Initialize arrays
        self.reachability_ = np.full(n_samples, np.inf)
        self.predecessor_ = np.full(n_samples, -1)
        processed = np.full(n_samples, False)
        self.ordering_ = []
        
        # OPTICS algorithm
        for point_id in range(n_samples):
            if processed[point_id]:
                continue
            
            # Start new cluster
            cluster_seeds = [point_id]
            processed[point_id] = True
            self.ordering_.append(point_id)
            
            while cluster_seeds:
                current_point = cluster_seeds.pop(0)
                
                # Get neighbors within max_eps
                current_distances = distances[current_point]
                current_indices = indices[current_point]
                
                # Filter by max_eps
                mask = current_distances <= self.max_eps
                neighbors = current_indices[mask]
                neighbor_distances = current_distances[mask]
                
                # Update reachability distances
                for neighbor, distance in zip(neighbors, neighbor_distances):
                    if not processed[neighbor]:
                        reachability_dist = max(self.core_distances_[current_point], distance)
                        
                        if reachability_dist < self.reachability_[neighbor]:
                            self.reachability_[neighbor] = reachability_dist
                            self.predecessor_[neighbor] = current_point
                
                # Process unprocessed points
                unprocessed_neighbors = [n for n in neighbors if not processed[n]]
                unprocessed_neighbors.sort(key=lambda x: self.reachability_[x])
                
                cluster_seeds.extend(unprocessed_neighbors)
                for neighbor in unprocessed_neighbors:
                    processed[neighbor] = True
                    self.ordering_.append(neighbor)
        
        # Extract clusters
        self._extract_clusters()
        
        return self
    
    def _extract_clusters(self):
        n_samples = len(self.ordering_)
        
        if self.cluster_method == 'dbscan':
            # Use DBSCAN-like extraction
            self.labels_ = np.full(n_samples, -1)
            cluster_id = 0
            
            for i in self.ordering_:
                if self.reachability_[i] <= self.max_eps:
                    if self.predecessor_[i] == -1 or self.labels_[self.predecessor_[i]] == -1:
                        self.labels_[i] = cluster_id
                        cluster_id += 1
                    else:
                        self.labels_[i] = self.labels_[self.predecessor_[i]]
        
        elif self.cluster_method == 'xi':
            # Xi method for cluster extraction
            self.labels_ = self._extract_clusters_xi()
        
        else:
            raise ValueError(f"Unsupported cluster method: {self.cluster_method}")
    
    def _extract_clusters_xi(self):
        # Simplified Xi method implementation
        n_samples = len(self.ordering_)
        labels = np.full(n_samples, -1)
        
        # This is a simplified version - full Xi method is more complex
        cluster_id = 0
        in_cluster = False
        
        for i in range(n_samples):
            point_id = self.ordering_[i]
            
            if self.reachability_[point_id] <= self.max_eps:
                if not in_cluster:
                    in_cluster = True
                    cluster_id += 1
                labels[point_id] = cluster_id
            else:
                in_cluster = False
                labels[point_id] = -1
        
        return labels
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def get_params(self):
        return {
            'min_samples': self.min_samples,
            'max_eps': self.max_eps,
            'metric': self.metric,
            'cluster_method': self.cluster_method,
            'xi': self.xi,
            'n_clusters': len(np.unique(self.labels_[self.labels_ != -1])) if self.labels_ is not None else 0
        }