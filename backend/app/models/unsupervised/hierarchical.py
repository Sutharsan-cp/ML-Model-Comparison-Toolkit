import numpy as np
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        
        self.labels_ = None
        self.distances_ = None
        self.children_ = None
        self.n_leaves_ = None
        self.n_components_ = None
        
    def _compute_linkage(self, distances, cluster_sizes, i, j):
        if self.linkage == 'single':
            return np.min(distances[np.ix_(i, j)])
        elif self.linkage == 'complete':
            return np.max(distances[np.ix_(i, j)])
        elif self.linkage == 'average':
            return np.mean(distances[np.ix_(i, j)])
        elif self.linkage == 'ward':
            n_i = cluster_sizes[i].sum() if isinstance(i, np.ndarray) else cluster_sizes[i]
            n_j = cluster_sizes[j].sum() if isinstance(j, np.ndarray) else cluster_sizes[j]
            n_k = n_i + n_j
            return np.sqrt((n_i * n_j) / n_k) * distances[np.ix_(i, j)].mean()
        else:
            raise ValueError(f"Unsupported linkage: {self.linkage}")
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        self.distances_ = squareform(pdist(X, metric=self.metric))
        
        # Initialize clusters
        clusters = [np.array([i]) for i in range(n_samples)]
        cluster_sizes = np.ones(n_samples)
        
        self.children_ = []
        current_clusters = clusters.copy()
        
        while len(current_clusters) > self.n_clusters:
            min_distance = np.inf
            merge_i, merge_j = -1, -1
            
            # Find closest clusters
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    distance = self._compute_linkage(
                        self.distances_, cluster_sizes, 
                        current_clusters[i], current_clusters[j]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Merge clusters
            new_cluster = np.concatenate([current_clusters[merge_i], current_clusters[merge_j]])
            self.children_.append([merge_i, merge_j, min_distance, len(new_cluster)])
            
            # Update clusters
            new_clusters = [current_clusters[k] for k in range(len(current_clusters)) 
                          if k != merge_i and k != merge_j]
            new_clusters.append(new_cluster)
            current_clusters = new_clusters
            
            # Update cluster sizes
            cluster_sizes = np.array([len(cluster) for cluster in current_clusters])
        
        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(current_clusters):
            self.labels_[cluster] = cluster_id
        
        self.n_leaves_ = n_samples
        self.n_components_ = len(current_clusters)
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'metric': self.metric,
            'n_leaves': self.n_leaves_,
            'n_components': self.n_components_,
            'children': self.children_
        }