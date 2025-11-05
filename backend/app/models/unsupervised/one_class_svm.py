import numpy as np
from scipy.optimize import minimize

class OneClassSVM:
    def __init__(self, kernel='rbf', nu=0.5, gamma='scale', degree=3, coef0=0.0,
                 tol=1e-3, max_iter=1000, random_state=None):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = 0.0
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (X1.shape[1] * np.var(X1)) if len(X1) > 0 else 1.0
            elif self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1] if X1.shape[1] > 0 else 1.0
            else:
                gamma = self.gamma
            
            norm_squared = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-gamma * norm_squared)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_objective(self, alpha, K):
        return 0.5 * alpha.T @ K @ alpha
    
    def _compute_gradient(self, alpha, K):
        return K @ alpha
    
    def _compute_constraints(self, alpha):
        return np.array([np.sum(alpha) - 1])
    
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        if self.gamma == 'scale':
            self.gamma = 1.0 / (n_features * np.var(X)) if n_features > 0 else 1.0
        elif self.gamma == 'auto':
            self.gamma = 1.0 / n_features if n_features > 0 else 1.0
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Set up optimization problem
        bounds = [(0, 1.0 / (self.nu * n_samples)) for _ in range(n_samples)]
        constraints = [{'type': 'eq', 'fun': lambda alpha: self._compute_constraints(alpha)}]
        
        alpha_initial = np.ones(n_samples) / n_samples
        
        result = minimize(
            fun=self._compute_objective,
            x0=alpha_initial,
            args=(K,),
            method='SLSQP',
            jac=self._compute_gradient,
            constraints=constraints,
            bounds=bounds,
            options={'ftol': self.tol, 'maxiter': self.max_iter, 'disp': False}
        )
        
        alpha = result.x
        support_vector_mask = alpha > 1e-5
        
        self.support_vectors_ = X[support_vector_mask]
        self.dual_coef_ = alpha[support_vector_mask]
        
        # Compute intercept
        if len(self.support_vectors_) > 0:
            K_sv = self._compute_kernel(self.support_vectors_, self.support_vectors_)
            self.intercept_ = np.median(
                np.sum(self.dual_coef_.reshape(-1, 1) * K_sv, axis=0)
            )
        
        self.is_fitted = True
        return self
    
    def decision_function(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        K = self._compute_kernel(X, self.support_vectors_)
        return np.sum(self.dual_coef_ * K, axis=1) - self.intercept_
    
    def predict(self, X):
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, 1, -1)
    
    def score_samples(self, X):
        return self.decision_function(X)
    
    def get_params(self):
        return {
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'n_support_vectors': len(self.support_vectors_) if self.support_vectors_ is not None else 0,
            'intercept': self.intercept_
        }