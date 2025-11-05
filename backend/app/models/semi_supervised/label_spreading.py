import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import rbf_kernel

class LabelSpreading:
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
        
        if self.kernel == 'rbf':
            W = rbf_kernel(X, X, gamma=self.gamma)
        elif self.kernel == 'knn':
            from sklearn.neighbors import kneighbors_graph
            W = kneighbors_graph(X, self.n_neighbors, mode='connectivity', include_self=True)
            W = 0.5 * (W + W.T)  # Symmetrize
            W = W.toarray()
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
        
        # Normalized graph Laplacian
        D = np.diag(np.sum(W, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
        S = D_inv_sqrt @ W @ D_inv_sqrt
        
        # Create clamping factor matrix
        clamp_matrix = np.zeros((n_samples, n_classes))
        clamp_matrix[labeled_indices] = self.label_distributions_[labeled_indices]
        
        # Label spreading iterations
        F = self.label_distributions_.copy()
        
        for iteration in range(self.max_iter):
            # Spread labels
            F_new = self.alpha * (S @ F) + (1 - self.alpha) * clamp_matrix
            
            # Check convergence
            change = np.linalg.norm(F_new - F)
            F = F_new
            
            if change < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        self.label_distributions_ = F
        
        # Create transduction
        self.transduction_ = self.classes_[np.argmax(self.label_distributions_, axis=1)]
        
        return self
    
    def predict(self, X):
        if self.label_distributions_ is None:
            raise ValueError("Model must be fitted before prediction")
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