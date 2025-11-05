import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, 
                 learning_rate=200.0, n_iter=1000, random_state=None, 
                 metric='euclidean', init='random', verbose=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.metric = metric
        self.init = init
        self.verbose = verbose
        
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_pairwise_affinities(self, X, perplexity):
        """Compute pairwise affinities using Gaussian kernel"""
        n_samples = X.shape[0]
        
        # Compute squared Euclidean distances
        distances = squareform(pdist(X, metric=self.metric))
        distances_squared = distances ** 2
        
        # Binary search for sigma that gives desired perplexity
        affinities = np.zeros((n_samples, n_samples))
        log_perplexity = np.log(perplexity)
        
        for i in range(n_samples):
            # Binary search for sigma
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0
            
            # Compute current affinities
            distances_i = distances_squared[i, np.concatenate([np.arange(0, i), np.arange(i+1, n_samples)])]
            
            for _ in range(50):  # Max 50 iterations
                # Compute affinities with current beta
                affinities_i = np.exp(-distances_i * beta)
                sum_affinities = np.sum(affinities_i)
                
                if sum_affinities == 0:
                    affinities_i = np.ones_like(affinities_i) / len(affinities_i)
                    sum_affinities = 1.0
                else:
                    affinities_i /= sum_affinities
                
                # Compute Shannon entropy
                entropy = -np.sum(affinities_i * np.log(affinities_i + 1e-8))
                
                # Check if we found the right perplexity
                entropy_diff = entropy - log_perplexity
                if np.abs(entropy_diff) < 1e-5:
                    break
                
                # Update beta
                if entropy_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2
            
            # Fill in the affinities matrix
            indices = np.concatenate([np.arange(0, i), np.arange(i+1, n_samples)])
            affinities[i, indices] = affinities_i
        
        # Symmetrize and normalize
        affinities = (affinities + affinities.T) / (2 * n_samples)
        np.fill_diagonal(affinities, 0)
        
        return affinities
    
    def _compute_low_dimensional_affinities(self, Y):
        """Compute low-dimensional affinities using Student's t-distribution"""
        n_samples = Y.shape[0]
        
        # Compute squared Euclidean distances in low dimension
        distances = squareform(pdist(Y, metric='euclidean'))
        distances_squared = distances ** 2
        
        # Compute Student's t-distribution
        numerator = 1.0 / (1.0 + distances_squared)
        np.fill_diagonal(numerator, 0)
        
        # Normalize
        denominator = np.sum(numerator)
        if denominator == 0:
            denominator = 1.0
        
        affinities = numerator / denominator
        
        return affinities
    
    def _compute_gradient(self, P, Q, Y):
        """Compute gradient of KL divergence"""
        n_samples = Y.shape[0]
        
        # Compute pairwise differences
        Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        
        # Compute gradient
        pq_diff = P - Q
        attraction = pq_diff[:, :, np.newaxis] * Y_diff
        
        # Compute repulsion using the t-distribution formulation
        distances_squared = np.sum(Y_diff ** 2, axis=2)
        repulsion_factor = 1.0 / (1.0 + distances_squared)
        repulsion = repulsion_factor[:, :, np.newaxis] * Y_diff
        
        gradient = 4.0 * (np.sum(attraction, axis=1) - np.sum(repulsion * Q[:, :, np.newaxis], axis=1))
        
        return gradient
    
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Compute high-dimensional affinities
        if self.verbose:
            print("Computing high-dimensional affinities...")
        P = self._compute_pairwise_affinities(X, self.perplexity)
        
        # Early exaggeration
        P *= self.early_exaggeration
        
        # Initialize low-dimensional representation
        if self.init == 'random':
            self.embedding_ = np.random.randn(n_samples, self.n_components) * 1e-4
        elif self.init == 'pca':
            from .pca import PCA
            pca = PCA(n_components=self.n_components)
            self.embedding_ = pca.fit_transform(X)
        else:
            raise ValueError(f"Unsupported initialization: {self.init}")
        
        # Optimization
        if self.verbose:
            print("Optimizing embedding...")
        
        momentum = 0.8
        gains = np.ones_like(self.embedding_)
        
        for iteration in range(self.n_iter):
            # Compute low-dimensional affinities
            Q = self._compute_low_dimensional_affinities(self.embedding_)
            
            # Compute gradient
            gradient = self._compute_gradient(P, Q, self.embedding_)
            
            # Update gains
            gains = (gains + 0.2) * ((gradient > 0) != (gains > 0)) + \
                    (gains * 0.8) * ((gradient > 0) == (gains > 0))
            gains = np.clip(gains, 0.01, np.inf)
            
            # Update embedding with momentum
            update = momentum * gradient - self.learning_rate * gains * gradient
            self.embedding_ += update
            
            # Center the embedding
            self.embedding_ -= np.mean(self.embedding_, axis=0)
            
            # Compute KL divergence for monitoring
            if iteration == 100:
                # Stop early exaggeration
                P /= self.early_exaggeration
            
            if self.verbose and iteration % 100 == 0:
                kl_div = np.sum(P * np.log((P + 1e-8) / (Q + 1e-8)))
                print(f"Iteration {iteration}, KL divergence: {kl_div:.4f}")
        
        self.n_iter_ = iteration + 1
        return self
    
    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'n_iter': self.n_iter_,
            'embedding_shape': self.embedding_.shape if self.embedding_ is not None else None
        }