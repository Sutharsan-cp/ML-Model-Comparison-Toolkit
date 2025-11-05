import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, n_clusters=8, max_iters=300, tol=1e-4, random_state=None, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.init = init
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape
        
        if self.init == 'k-means++':
            centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(1, self.n_clusters):
                distances = np.min(cdist(X, np.array(centroids)), axis=1)
                probabilities = distances / np.sum(distances)
                cumulative_probs = np.cumsum(probabilities)
                r = np.random.rand()
                
                for i, prob in enumerate(cumulative_probs):
                    if r < prob:
                        centroids.append(X[i])
                        break
            return np.array(centroids)
        
        elif self.init == 'random':
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[indices]
        
        else:
            raise ValueError("Initialization method must be 'k-means++' or 'random'")
    
    def _assign_clusters(self, X, centroids):
        distances = cdist(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids
    
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}")
        
        self.cluster_centers_ = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            self.n_iter_ = iteration + 1
            
            # Assign clusters
            self.labels_ = self._assign_clusters(X, self.cluster_centers_)
            
            # Update centroids
            new_centroids = self._update_centroids(X, self.labels_)
            
            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - self.cluster_centers_)
            self.cluster_centers_ = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        # Compute inertia (within-cluster sum of squares)
        self.inertia_ = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum((cluster_points - self.cluster_centers_[i])**2)
        
        return self
    
    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def transform(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before transform")
        X = np.array(X)
        return cdist(X, self.cluster_centers_)
    
    def score(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before scoring")
        X = np.array(X)
        distances = np.min(cdist(X, self.cluster_centers_), axis=1)
        return -np.sum(distances**2)
    
    def get_params(self):
        return {
            'n_clusters': self.n_clusters,
            'max_iters': self.max_iters,
            'tol': self.tol,
            'init': self.init,
            'n_iter': self.n_iter_,
            'inertia': self.inertia_,
            'cluster_centers': self.cluster_centers_.tolist() if self.cluster_centers_ is not None else None
        }