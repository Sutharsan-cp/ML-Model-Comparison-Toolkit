import numpy as np

class PCA:
    def __init__(self, n_components=None, svd_solver='auto', random_state=None):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.random_state = random_state
        
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X):
        X = np.array(X)
        self.n_samples_, self.n_features_ = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Determine number of components
        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_)
        else:
            self.n_components_ = min(self.n_components, self.n_samples_, self.n_features_)
        
        # Compute SVD
        if self.svd_solver in ['auto', 'full']:
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        else:
            raise ValueError(f"Unsupported SVD solver: {self.svd_solver}")
        
        # Select components
        self.components_ = Vt[:self.n_components_]
        self.singular_values_ = s[:self.n_components_]
        
        # Compute explained variance
        self.explained_variance_ = (self.singular_values_ ** 2) / (self.n_samples_ - 1)
        total_variance = np.var(X_centered, axis=0).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        if self.components_ is None:
            raise ValueError("Model must be fitted before transform")
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        if self.components_ is None:
            raise ValueError("Model must be fitted before inverse_transform")
        X_transformed = np.array(X_transformed)
        return X_transformed @ self.components_ + self.mean_
    
    def get_covariance(self):
        if self.components_ is None:
            raise ValueError("Model must be fitted before get_covariance")
        return (self.components_.T * self.explained_variance_) @ self.components_
    
    def get_precision(self):
        covariance = self.get_covariance()
        return np.linalg.inv(covariance)
    
    def get_params(self):
        return {
            'n_components': self.n_components_,
            'n_features': self.n_features_,
            'n_samples': self.n_samples_,
            'explained_variance_ratio': self.explained_variance_ratio_.tolist() if self.explained_variance_ratio_ is not None else None,
            'cumulative_variance_ratio': np.cumsum(self.explained_variance_ratio_).tolist() if self.explained_variance_ratio_ is not None else None,
            'singular_values': self.singular_values_.tolist() if self.singular_values_ is not None else None
        }