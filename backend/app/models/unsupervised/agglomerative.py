import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean', 
                 compute_full_tree='auto', distance_threshold=None):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold
        
        self.labels_ = None
        self.n_leaves_ = None
        self.n_components_ = None
        self.children_ = None
        
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Compute linkage matrix
        if n_samples > 1:
            distances = pdist(X, metric=self.metric)
            self.linkage_matrix_ = linkage(distances, method=self.linkage)
            self.children_ = self.linkage_matrix_[:, :2].astype(int)
        else:
            self.linkage_matrix_ = np.array([])
            self.children_ = np.array([])
        
        # Get cluster labels
        if self.distance_threshold is not None:
            self.labels_ = fcluster(self.linkage_matrix_, self.distance_threshold, 
                                  criterion='distance')
            self.n_clusters = len(np.unique(self.labels_))
        else:
            self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters, 
                                  criterion='maxclust')
        
        self.n_leaves_ = n_samples
        self.n_components_ = self.n_clusters
        
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
            'distance_threshold': self.distance_threshold
        }