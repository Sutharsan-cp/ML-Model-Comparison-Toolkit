import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_child_weight=1, gamma=0, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        
        self.estimators_ = []
        self.initial_prediction_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_gradient_hessian(self, y, pred):
        """Compute gradient and hessian for regression (MSE loss)"""
        gradient = 2 * (pred - y)  # Gradient of MSE
        hessian = 2 * np.ones_like(y)  # Hessian of MSE is constant
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
            
            # Subsample data and features
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            else:
                sample_indices = np.arange(n_samples)
            
            if self.colsample_bytree < 1.0:
                feature_size = int(self.colsample_bytree * n_features)
                feature_indices = np.random.choice(n_features, feature_size, replace=False)
            else:
                feature_indices = np.arange(n_features)
            
            X_subsample = X[sample_indices][:, feature_indices]
            gradients_subsample = gradients[sample_indices]
            hessians_subsample = hessians[sample_indices]
            
            # Fit tree using custom criterion (simplified)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=max(2, int(self.min_child_weight)),
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
        return -mean_squared_error(y, predictions)  # Negative MSE for consistency
    
    def staged_predict(self, X):
        """Predict at each stage of boosting"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        predictions = np.full(n_samples, self.initial_prediction_)
        
        for tree, feature_indices in self.estimators_:
            X_subset = X[:, feature_indices]
            predictions += self.learning_rate * tree.predict(X_subset)
            yield predictions.copy()
    
    def get_feature_importance(self):
        """Compute feature importance"""
        if not self.estimators_:
            return None
        
        n_features = self.estimators_[0][1].shape[0]
        importances = np.zeros(n_features)
        
        for tree, feature_indices in self.estimators_:
            tree_importance = tree.feature_importances_
            for idx, feature_idx in enumerate(feature_indices):
                if feature_idx < n_features:
                    importances[feature_idx] += tree_importance[idx]
        
        # Normalize
        if importances.sum() > 0:
            importances /= importances.sum()
        
        return importances
    
    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_lambda': self.reg_lambda,
            'reg_alpha': self.reg_alpha,
            'random_state': self.random_state
        }