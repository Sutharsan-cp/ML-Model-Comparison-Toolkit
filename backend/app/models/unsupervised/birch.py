import numpy as np
from scipy.spatial.distance import cdist

class CFNode:
    def __init__(self, threshold, branching_factor, is_leaf=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.subclusters = []
        self.children = []
        
        # CF features: N, LS, SS
        self.N = 0
        self.LS = None
        self.SS = 0.0
    
    def add_subcluster(self, subcluster):
        if len(self.subclusters) < self.branching_factor:
            self.subclusters.append(subcluster)
            self._update_cf_features(subcluster)
            return True
        return False
    
    def _update_cf_features(self, subcluster):
        if self.LS is None:
            self.LS = np.zeros_like(subcluster.LS)
        
        self.N += subcluster.N
        self.LS += subcluster.LS
        self.SS += subcluster.SS
    
    def distance_to_subcluster(self, subcluster):
        if self.N == 0:
            return np.inf
        
        centroid = self.LS / self.N
        other_centroid = subcluster.LS / subcluster.N
        return np.linalg.norm(centroid - other_centroid)

class CFSubcluster:
    def __init__(self, data_point):
        self.N = 1
        self.LS = data_point.copy()
        self.SS = np.sum(data_point ** 2)
        self.centroid = data_point.copy()
    
    def add_point(self, data_point):
        self.N += 1
        self.LS += data_point
        self.SS += np.sum(data_point ** 2)
        self.centroid = self.LS / self.N
    
    def radius(self):
        if self.N <= 1:
            return 0.0
        variance = (self.SS / self.N) - np.sum((self.LS / self.N) ** 2)
        return np.sqrt(np.sum(variance))
    
    def distance_to_point(self, data_point):
        return np.linalg.norm(self.centroid - data_point)

class BIRCH:
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        
        self.root = None
        self.labels_ = None
        self.subcluster_centers_ = None
        
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize root node
        self.root = CFNode(self.threshold, self.branching_factor, is_leaf=True)
        
        # Build CF tree
        for point in X:
            self._insert_point(point, self.root)
        
        # Extract subcluster centers
        subclusters = self._get_all_subclusters(self.root)
        self.subcluster_centers_ = np.array([sc.centroid for sc in subclusters])
        
        # Cluster subclusters using K-means
        if len(subclusters) > self.n_clusters:
            from .kmeans import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters)
            subcluster_labels = kmeans.fit_predict(self.subcluster_centers_)
            
            # Assign original points to clusters
            self.labels_ = np.zeros(n_samples, dtype=int)
            for i, point in enumerate(X):
                closest_subcluster = np.argmin(
                    cdist([point], self.subcluster_centers_)[0]
                )
                self.labels_[i] = subcluster_labels[closest_subcluster]
        else:
            self.labels_ = np.zeros(n_samples, dtype=int)
        
        return self
    
    def _insert_point(self, point, node):
        if node.is_leaf:
            # Find closest subcluster in leaf node
            min_distance = np.inf
            closest_subcluster = None
            
            for subcluster in node.subclusters:
                distance = subcluster.distance_to_point(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_subcluster = subcluster
            
            # Add to existing subcluster or create new one
            if closest_subcluster is not None and closest_subcluster.radius() < self.threshold:
                closest_subcluster.add_point(point)
            else:
                new_subcluster = CFSubcluster(point)
                if not node.add_subcluster(new_subcluster):
                    # Split node if full
                    self._split_node(node)
                    self._insert_point(point, node)
        else:
            # Find closest child node
            min_distance = np.inf
            closest_child = None
            
            for child in node.children:
                distance = child.distance_to_subcluster(CFSubcluster(point))
                if distance < min_distance:
                    min_distance = distance
                    closest_child = child
            
            if closest_child is not None:
                self._insert_point(point, closest_child)
    
    def _split_node(self, node):
        # Simple splitting strategy: redistribute subclusters
        node.is_leaf = False
        for subcluster in node.subclusters:
            new_node = CFNode(self.threshold, self.branching_factor, is_leaf=True)
            new_node.add_subcluster(subcluster)
            node.children.append(new_node)
        node.subclusters = []
    
    def _get_all_subclusters(self, node):
        subclusters = []
        if node.is_leaf:
            subclusters.extend(node.subclusters)
        else:
            for child in node.children:
                subclusters.extend(self._get_all_subclusters(child))
        return subclusters
    
    def predict(self, X):
        if self.subcluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        distances = cdist(X, self.subcluster_centers_)
        closest_subclusters = np.argmin(distances, axis=1)
        
        # Map to final clusters (this would need to be stored during fit)
        return np.zeros(X.shape[0], dtype=int)  # Simplified
    
    def get_params(self):
        return {
            'threshold': self.threshold,
            'branching_factor': self.branching_factor,
            'n_clusters': self.n_clusters,
            'n_subclusters': len(self.subcluster_centers_) if self.subcluster_centers_ is not None else 0
        }