import numpy as np
from scipy.optimize import minimize

class SemiSupervisedSVM:
    def __init__(self, C=1.0, C_star=0.1, kernel='linear', gamma='scale', 
                 degree=3, coef0=0.0, max_iter=1000, tol=1e-3, random_state=None):
        self.C = C
        self.C_star = C_star
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = 0.0
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (X1.shape[1] * np.var(X1)) if len(X1) > 0 else 1.0
            elif self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1] if X1.shape[1] > 0 else 1.0
            else:
                gamma = self.gamma
            
            norm_squared = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-gamma * norm_squared)
        elif self.kernel == 'poly':
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_objective(self, alpha, K, y_labeled, n_labeled, n_unlabeled):
        alpha_labeled = alpha[:n_labeled]
        alpha_unlabeled = alpha[n_labeled:]
        
        # Standard SVM objective for labeled data
        labeled_obj = 0.5 * alpha_labeled.T @ (K[:n_labeled, :n_labeled] * np.outer(y_labeled, y_labeled)) @ alpha_labeled - np.sum(alpha_labeled)
        
        # Additional objective for unlabeled data
        unlabeled_obj = 0.5 * self.C_star * alpha_unlabeled.T @ K[n_labeled:, n_labeled:] @ alpha_unlabeled
        
        return labeled_obj + unlabeled_obj
    
    def _compute_gradient(self, alpha, K, y_labeled, n_labeled, n_unlabeled):
        alpha_labeled = alpha[:n_labeled]
        alpha_unlabeled = alpha[n_labeled:]
        
        grad_labeled = (K[:n_labeled, :n_labeled] * np.outer(y_labeled, y_labeled)) @ alpha_labeled - np.ones(n_labeled)
        grad_unlabeled = self.C_star * K[n_labeled:, n_labeled:] @ alpha_unlabeled
        
        return np.concatenate([grad_labeled, grad_unlabeled])
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Separate labeled and unlabeled data
        labeled_mask = y != -1
        unlabeled_mask = ~labeled_mask
        
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        X_unlabeled = X[unlabeled_mask]
        
        n_labeled = len(X_labeled)
        n_unlabeled = len(X_unlabeled)
        n_total = n_labeled + n_unlabeled
        
        if n_labeled == 0:
            raise ValueError("No labeled samples provided")
        
        self.classes_ = np.unique(y_labeled)
        if len(self.classes_) != 2:
            raise ValueError("SemiSupervisedSVM supports only binary classification")
        
        # Convert labels to -1, 1
        y_binary = np.where(y_labeled == self.classes_[0], -1, 1)
        
        # Combine labeled and unlabeled data
        X_combined = np.vstack([X_labeled, X_unlabeled])
        
        # Compute kernel matrix
        K = self._compute_kernel(X_combined, X_combined)
        
        # Set up optimization
        bounds = [(0, self.C) for _ in range(n_labeled)] + [(0, self.C_star) for _ in range(n_unlabeled)]
        
        # Linear constraint: sum of alpha * y = 0 for labeled data
        constraints = [{'type': 'eq', 'fun': lambda alpha: np.sum(alpha[:n_labeled] * y_binary)}]
        
        # Initial guess
        alpha_initial = np.zeros(n_total)
        
        # Solve optimization problem
        result = minimize(
            fun=self._compute_objective,
            x0=alpha_initial,
            args=(K, y_binary, n_labeled, n_unlabeled),
            method='SLSQP',
            jac=self._compute_gradient,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': self.tol, 'disp': False}
        )
        
        alpha = result.x
        
        # Extract support vectors
        support_mask = alpha > 1e-5
        self.support_vectors_ = X_combined[support_mask]
        
        # Extract dual coefficients
        alpha_support = alpha[support_mask]
        y_support = np.concatenate([y_binary, np.ones(n_unlabeled)])[support_mask]  # Approximate for unlabeled
        self.dual_coef_ = alpha_support * y_support
        
        # Compute intercept
        if len(self.support_vectors_) > 0:
            margin_points = (alpha[:n_labeled] > 1e-5) & (alpha[:n_labeled] < self.C - 1e-5)
            if np.any(margin_points):
                K_margin = self._compute_kernel(X_labeled[margin_points], self.support_vectors_)
                self.intercept_ = np.mean(
                    y_binary[margin_points] - np.sum(self.dual_coef_ * K_margin, axis=1)
                )
        
        return self
    
    def decision_function(self, X):
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        K = self._compute_kernel(X, self.support_vectors_)
        return K @ self.dual_coef_ + self.intercept_
    
    def predict(self, X):
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, self.classes_[1], self.classes_[0])
    
    def get_params(self):
        return {
            'C': self.C,
            'C_star': self.C_star,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'n_support_vectors': len(self.support_vectors_) if self.support_vectors_ is not None else 0
        }