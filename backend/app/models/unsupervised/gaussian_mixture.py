import numpy as np
from scipy.special import logsumexp

class GaussianMixture:
    def __init__(self, n_components=3, max_iters=100, tol=1e-3, 
                 init_params='kmeans', random_state=None, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.init_params = init_params
        self.random_state = random_state
        self.reg_covar = reg_covar
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.responsibilities_ = None
        self.lower_bound_ = -np.inf
        self.n_iter_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        if self.init_params == 'random':
            # Initialize weights uniformly
            self.weights_ = np.ones(self.n_components) / self.n_components
            
            # Initialize means with random samples
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[indices]
            
            # Initialize covariances as identity matrices
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        elif self.init_params == 'kmeans':
            # Use K-means for initialization
            from .kmeans import KMeans
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            
            self.weights_ = np.zeros(self.n_components)
            self.means_ = np.zeros((self.n_components, n_features))
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            
            for i in range(self.n_components):
                cluster_points = X[labels == i]
                self.weights_[i] = len(cluster_points) / n_samples
                self.means_[i] = cluster_points.mean(axis=0)
                if len(cluster_points) > 1:
                    self.covariances_[i] = np.cov(cluster_points.T)
                else:
                    self.covariances_[i] = np.eye(n_features)
    
    def _multivariate_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        diff = X - mean
        try:
            precision = np.linalg.inv(cov)
            determinant = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            # Add regularization for singular matrices
            cov = cov + self.reg_covar * np.eye(n_features)
            precision = np.linalg.inv(cov)
            determinant = np.linalg.det(cov)
        
        exponent = -0.5 * np.sum(diff @ precision * diff, axis=1)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * determinant)
        return norm_const * np.exp(exponent)
    
    def _e_step(self, X):
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            log_resp[:, k] = np.log(self.weights_[k]) + np.log(
                self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
            )
        
        # Normalize using logsumexp for numerical stability
        log_sum = logsumexp(log_resp, axis=1)
        log_resp -= log_sum[:, np.newaxis]
        self.responsibilities_ = np.exp(log_resp)
        
        return np.mean(log_sum)  # Average log likelihood
    
    def _m_step(self, X):
        n_samples = X.shape[0]
        
        # Effective number of points assigned to each component
        Nk = np.sum(self.responsibilities_, axis=0)
        
        # Update weights
        self.weights_ = Nk / n_samples
        
        # Update means
        self.means_ = (self.responsibilities_.T @ X) / Nk[:, np.newaxis]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (self.responsibilities_[:, k, np.newaxis] * diff).T @ diff / Nk[k]
            # Add regularization for numerical stability
            self.covariances_[k] += self.reg_covar * np.eye(X.shape[1])
    
    def fit(self, X):
        X = np.array(X)
        self._initialize_parameters(X)
        
        self.lower_bound_ = -np.inf
        
        for iteration in range(self.max_iters):
            self.n_iter_ = iteration + 1
            
            # E-step
            current_lower_bound = self._e_step(X)
            
            # Check convergence
            if np.abs(current_lower_bound - self.lower_bound_) < self.tol:
                break
                
            self.lower_bound_ = current_lower_bound
            
            # M-step
            self._m_step(X)
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        self._e_step(X)
        return np.argmax(self.responsibilities_, axis=1)
    
    def predict_proba(self, X):
        X = np.array(X)
        self._e_step(X)
        return self.responsibilities_
    
    def score_samples(self, X):
        X = np.array(X)
        log_prob = np.zeros(X.shape[0])
        for k in range(self.n_components):
            log_prob += self.weights_[k] * self._multivariate_gaussian(
                X, self.means_[k], self.covariances_[k]
            )
        return np.log(log_prob + 1e-8)
    
    def bic(self, X):
        n_samples = X.shape[0]
        n_params = (self.n_components - 1) + \
                  self.n_components * X.shape[1] + \
                  self.n_components * X.shape[1] * (X.shape[1] + 1) // 2
        
        return -2 * np.sum(self.score_samples(X)) + n_params * np.log(n_samples)
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'n_iter': self.n_iter_,
            'lower_bound': self.lower_bound_,
            'weights': self.weights_.tolist() if self.weights_ is not None else None,
            'means': self.means_.tolist() if self.means_ is not None else None
        }