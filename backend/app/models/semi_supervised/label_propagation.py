import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

class LabelPropagation:
    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7, 
                 alpha=0.2, max_iter=1000, tol=1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
        self.label_distributions_ = None
        self.transduction_ = None
        self.n_iter_ = None
        
    def _build_graph(self, X):
        n_samples = X.shape[0]
        
        if self.kernel == 'knn':
            # KNN graph
            knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            knn.fit(X)
            W = knn.kneighbors_graph(X, mode='connectivity').toarray()
            
        elif self.kernel == 'rbf':
            # RBF kernel
            W = rbf_kernel(X, X, gamma=self.gamma)
            # Keep only n_neighbors connections
            if self.n_neighbors is not None:
                knn = NearestNeighbors(n_neighbors=self.n_neighbors)
                knn.fit(X)
                knn_mask = knn.kneighbors_graph(X).toarray().astype(bool)
                W = W * knn_mask
        
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        
        return W
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        labeled_indices = np.where(y != -1)[0]
        unlabeled_indices = np.where(y == -1)[0]
        
        if len(labeled_indices) == 0:
            raise ValueError("No labeled samples provided")
        
        # Get unique classes
        self.classes_ = np.unique(y[labeled_indices])
        n_classes = len(self.classes_)
        
        # Initialize label distributions
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        
        # Set initial labels for labeled points
        for i in labeled_indices:
            class_idx = np.where(self.classes_ == y[i])[0][0]
            self.label_distributions_[i, class_idx] = 1.0
        
        # Build graph
        W = self._build_graph(X)
        
        # Normalize graph
        D = np.sum(W, axis=1)
        D_sqrt = np.sqrt(D)
        W_normalized = W / (D_sqrt[:, np.newaxis] * D_sqrt)
        
        # Create mask for labeled points
        labeled_mask = np.zeros(n_samples, dtype=bool)
        labeled_mask[labeled_indices] = True
        
        # Label propagation
        previous_distributions = self.label_distributions_.copy()
        
        for iteration in range(self.max_iter):
            # Propagate labels
            new_distributions = self.alpha * W_normalized @ self.label_distributions_
            
            # Keep original labels for labeled points
            new_distributions[labeled_mask] = previous_distributions[labeled_mask]
            
            # Normalize
            row_sums = np.sum(new_distributions, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            new_distributions /= row_sums
            
            # Check convergence
            change = np.linalg.norm(new_distributions - self.label_distributions_)
            self.label_distributions_ = new_distributions
            
            if change < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        
        # Create transduction
        self.transduction_ = self.classes_[np.argmax(self.label_distributions_, axis=1)]
        
        return self
    
    def predict(self, X):
        if self.label_distributions_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # For simplicity, return transduction for training points
        # In practice, you'd need to propagate to new points
        return self.transduction_
    
    def predict_proba(self, X):
        if self.label_distributions_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.label_distributions_
    
    def get_params(self):
        return {
            'kernel': self.kernel,
            'gamma': self.gamma,
            'n_neighbors': self.n_neighbors,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'n_iter': self.n_iter_,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else 0
        }