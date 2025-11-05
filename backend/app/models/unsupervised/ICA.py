import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

class FastICA:
    def __init__(self, n_components=None, algorithm='parallel', fun='logcosh',
                 fun_args=None, max_iter=200, tol=1e-4, w_init=None, random_state=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state
        
        self.components_ = None
        self.mixing_ = None
        self.mean_ = None
        self.n_iter_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _center(self, X):
        self.mean_ = np.mean(X, axis=0)
        return X - self.mean_
    
    def _whiten(self, X):
        # PCA whitening
        cov = np.cov(X, rowvar=False)
        d, V = linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(d)[::-1]
        d = d[idx]
        V = V[:, idx]
        
        if self.n_components is not None:
            V = V[:, :self.n_components]
            d = d[:self.n_components]
        
        # Whitening matrix
        D = np.diag(1.0 / np.sqrt(d))
        self.whitening_ = np.dot(V, D)
        
        return np.dot(X, self.whitening_)
    
    def _logcosh(self, x, alpha=1.0):
        return np.tanh(alpha * x)
    
    def _exp(self, x):
        return x * np.exp(-x**2 / 2)
    
    def _cube(self, x):
        return x**3
    
    def _g(self, x, fun='logcosh'):
        if fun == 'logcosh':
            return self._logcosh(x)
        elif fun == 'exp':
            return self._exp(x)
        elif fun == 'cube':
            return self._cube(x)
        else:
            raise ValueError(f"Unsupported function: {fun}")
    
    def _g_prime(self, x, fun='logcosh'):
        if fun == 'logcosh':
            return 1.0 - np.tanh(x)**2
        elif fun == 'exp':
            return (1 - x**2) * np.exp(-x**2 / 2)
        elif fun == 'cube':
            return 3 * x**2
        else:
            raise ValueError(f"Unsupported function: {fun}")
    
    def _sym_decorrelation(self, W):
        # Symmetric decorrelation
        s, U = linalg.eigh(np.dot(W, W.T))
        return np.dot(np.dot(U * (1.0 / np.sqrt(s)), U.T), W)
    
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Center and whiten the data
        X_centered = self._center(X)
        X_white = self._whiten(X_centered)
        
        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        
        # Initialize weights
        if self.w_init is None:
            w_init = np.random.normal(0, 1, (n_components, n_components))
        else:
            w_init = self.w_init
        
        W = self._sym_decorrelation(w_init)
        
        # ICA algorithm
        for n_iter in range(self.max_iter):
            W_old = W.copy()
            
            if self.algorithm == 'parallel':
                # Parallel update
                gwtx = self._g(np.dot(W, X_white.T), self.fun)
                g_wtx = self._g_prime(np.dot(W, X_white.T), self.fun)
                W1 = np.dot(gwtx, X_white) / n_samples - np.dot(np.diag(g_wtx.mean(axis=1)), W)
                
            elif self.algorithm == 'deflation':
                # Deflation (one unit at a time)
                W1 = np.zeros_like(W)
                for j in range(n_components):
                    w = W[j, :]
                    
                    for _ in range(self.max_iter):
                        w_old = w.copy()
                        
                        gwtx = self._g(np.dot(w, X_white.T), self.fun)
                        g_wtx = self._g_prime(np.dot(w, X_white.T), self.fun)
                        
                        w = (np.dot(gwtx, X_white) / n_samples - g_wtx.mean() * w)
                        
                        # Decorrelate
                        w -= np.dot(np.dot(w, W1[:j].T), W1[:j])
                        w /= np.linalg.norm(w)
                        
                        if np.abs(np.abs(np.dot(w, w_old)) - 1) < self.tol:
                            break
                    
                    W1[j, :] = w
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            W = self._sym_decorrelation(W1)
            
            # Check convergence
            lim = max(abs(abs(np.diag(np.dot(W, W_old.T))) - 1))
            if lim < self.tol:
                break
        
        self.n_iter_ = n_iter + 1
        self.components_ = np.dot(W, self.whitening_)
        self.mixing_ = linalg.pinv(self.components_)
        
        return self
    
    def transform(self, X):
        if self.components_ is None:
            raise ValueError("Model must be fitted before transform")
        X = np.array(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        if self.components_ is None:
            raise ValueError("Model must be fitted before inverse_transform")
        X_transformed = np.array(X_transformed)
        return np.dot(X_transformed, self.components_) + self.mean_
    
    def get_params(self):
        return {
            'n_components': self.n_components,
            'algorithm': self.algorithm,
            'fun': self.fun,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'n_iter': self.n_iter_,
            'components_shape': self.components_.shape if self.components_ is not None else None
        }