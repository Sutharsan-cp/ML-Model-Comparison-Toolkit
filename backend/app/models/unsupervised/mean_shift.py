import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

class MeanShift:
    def __init__(self, bandwidth=None, seeds=None, bin_seeding=False, 
                 min_bin_freq=1, cluster_all=True, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.max_iter = max_iter
        self.tol = tol
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = None
    
    def _estimate_bandwidth(self, X, quantile=0.3):
        """Estimate bandwidth using quantile of pairwise distances"""
        n_samples = X.shape[0]
        
        if n_samples > 1000:
            # Use subset for large datasets
            indices = np.random.choice(n_samples, 1000, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
        
        # Compute pairwise distances
        distances = cdist(X_subset, X_subset)
        upper_triangular = distances[np.triu_indices_from(distances, k=1)]
        
        return np.quantile(upper_triangular, quantile)
    
    def _get_seeds(self, X, bandwidth):
        """Get initial seeds for mean shift"""
        if self.seeds is not None:
            return self.seeds
        
        if self.bin_seeding:
            return self._get_bin_seeds(X, bandwidth)
        else:
            return X.copy()
    
    def _get_bin_seeds(self, X, bandwidth):
        """Get seeds using bin seeding"""
        n_samples, n_features = X.shape
        
        # Discretize data into bins
        bin_sizes = bandwidth * np.ones(n_features)
        bin_indices = np.floor(X / bin_sizes)
        
        # Find unique bins
        unique_bins, bin_counts = np.unique(bin_indices, axis=0, return_counts=True)
        
        # Filter bins by frequency
        frequent_bins = unique_bins[bin_counts >= self.min_bin_freq]
        
        # Convert bin indices back to data coordinates
        seeds = frequent_bins * bin_sizes + bin_sizes / 2
        
        return seeds
    
    def _mean_shift_single_seed(self, seed, X, bandwidth):
        """Perform mean shift for a single seed"""
        previous_seed = seed.copy()
        
        for iteration in range(self.max_iter):
            # Find points within bandwidth
            distances = cdist([previous_seed], X)[0]
            within_bandwidth = distances <= bandwidth
            
            if not np.any(within_bandwidth):
                break
            
            # Compute mean of points within bandwidth
            points_within = X[within_bandwidth]
            new_seed = np.mean(points_within, axis=0)
            
            # Check convergence
            shift = np.linalg.norm(new_seed - previous_seed)
            if shift < self.tol:
                break
            
            previous_seed = new_seed
        
        return new_seed, iteration + 1
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Estimate bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X)
        
        # Get initial seeds
        seeds = self._get_seeds(X, self.bandwidth)
        
        # Perform mean shift for each seed
        all_centers = []
        n_iter_total = 0
        
        for seed in seeds:
            center, n_iter = self._mean_shift_single_seed(seed, X, self.bandwidth)
            all_centers.append(center)
            n_iter_total = max(n_iter_total, n_iter)
        
        self.n_iter_ = n_iter_total
        
        # Cluster centers
        all_centers = np.array(all_centers)
        
        # Merge nearby centers
        self.cluster_centers_ = self._merge_centers(all_centers, self.bandwidth)
        
        # Assign labels to original data
        if self.cluster_all:
            distances = cdist(X, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)
        else:
            self.labels_ = -np.ones(n_samples, dtype=int)
            for i, point in enumerate(X):
                distances = cdist([point], self.cluster_centers_)[0]
                if np.min(distances) <= self.bandwidth:
                    self.labels_[i] = np.argmin(distances)
        
        return self
    
    def _merge_centers(self, centers, bandwidth):
        """Merge centers that are within bandwidth of each other"""
        if len(centers) == 0:
            return centers
        
        # Sort centers by density (approximated by number of nearby centers)
        distances = cdist(centers, centers)
        densities = np.sum(distances <= bandwidth, axis=1)
        
        sorted_indices = np.argsort(densities)[::-1]
        sorted_centers = centers[sorted_indices]
        
        merged_centers = []
        used = np.zeros(len(sorted_centers), dtype=bool)
        
        for i, center in enumerate(sorted_centers):
            if used[i]:
                continue
            
            # Find all centers within bandwidth
            distances_to_center = cdist([center], sorted_centers)[0]
            nearby = distances_to_center <= bandwidth
            
            # Merge nearby centers
            merged_center = np.mean(sorted_centers[nearby], axis=0)
            merged_centers.append(merged_center)
            
            used[nearby] = True
        
        return np.array(merged_centers)
    
    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.array(X)
        distances = cdist(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def get_params(self):
        return {
            'bandwidth': self.bandwidth,
            'n_clusters': len(self.cluster_centers_) if self.cluster_centers_ is not None else 0,
            'n_iter': self.n_iter_,
            'cluster_centers': self.cluster_centers_.tolist() if self.cluster_centers_ is not None else None
        }