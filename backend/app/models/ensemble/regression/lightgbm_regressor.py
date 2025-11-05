import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class LightGBMRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1,
                 num_leaves=31, min_data_in_leaf=20, feature_fraction=1.0,
                 bagging_fraction=1.0, bagging_freq=0, lambda_l1=0.0, 
                 lambda_l2=0.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.random_state = random_state
        
        self.estimators_ = []
        self.initial_prediction_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_gradient_hessian(self, y, pred):
        """Compute gradient and hessian for regression (MSE loss)"""
        gradient = 2 * (pred - y)
        hessian = 2 * np.ones_like(y)
        return gradient, hessian
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize with mean
        self.initial_prediction_ = np.mean(y)
        current_predictions = np.full(n_samples, self.initial_prediction_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Compute gradients and hessians
            gradients, hessians = self._compute_gradient_hessian(y, current_predictions)
            
            # Feature fraction
            if self.feature_fraction < 1.0:
                feature_size = int(self.feature_fraction * n_features)
                feature_indices = np.random.choice(n_features, feature_size, replace=False)
            else:
                feature_indices = np.arange(n_features)
            
            # Bagging
            if self.bagging_fraction < 1.0 and (i % max(1, self.bagging_freq) == 0):
                sample_size = int(self.bagging_fraction * n_samples)
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            else:
                sample_indices = np.arange(n_samples)
            
            X_subsample = X[sample_indices][:, feature_indices]
            gradients_subsample = gradients[sample_indices]
            hessians_subsample = hessians[sample_indices]
            
            # Fit tree with LightGBM-like parameters
            max_depth = self.max_depth if self.max_depth > 0 else None
            
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                max_leaf_nodes=self.num_leaves,
                min_samples_leaf=max(1, self.min_data_in_leaf),
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Use hessians as sample weights
            tree.fit(X_subsample, -gradients_subsample, sample_weight=hessians_subsample)
            
            # Update predictions
            leaf_predictions = tree.predict(X[:, feature_indices])
            current_predictions += self.learning_rate * leaf_predictions
            
            self.estimators_.append((tree, feature_indices))
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        predictions = np.full(n_samples, self.initial_prediction_)
        
        # Add contributions from all trees
        for tree, feature_indices in self.estimators_:
            X_subset = X[:, feature_indices]
            predictions += self.learning_rate * tree.predict(X_subset)
        
        return predictions
    
    def score(self, X, y):
        predictions = self.predict(X)
        return -mean_squared_error(y, predictions)
    
    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_data_in_leaf': self.min_data_in_leaf,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'random_state': self.random_state
        }