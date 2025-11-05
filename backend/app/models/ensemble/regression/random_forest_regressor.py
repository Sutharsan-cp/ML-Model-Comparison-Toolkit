import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.estimators_ = []
        self.estimator_features_ = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        
        self.estimators_ = []
        self.estimator_features_ = []
        
        for i in range(self.n_estimators):
            # Create decision tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                sample_indices = np.arange(n_samples)
            
            # Feature sampling
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            
            X_bootstrap = X[sample_indices][:, feature_indices]
            y_bootstrap = y[sample_indices]
            
            # Train tree
            tree.fit(X_bootstrap, y_bootstrap)
            
            self.estimators_.append(tree)
            self.estimator_features_.append(feature_indices)
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        
        for i, (tree, feature_indices) in enumerate(zip(self.estimators_, self.estimator_features_)):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        
        # Average predictions
        return np.mean(predictions, axis=1)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return -mean_squared_error(y, predictions)  # Negative MSE for consistency
    
    def get_feature_importances(self):
        if not self.estimators_:
            return None
        
        n_features = len(self.estimator_features_[0])
        importances = np.zeros(n_features)
        
        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):
            tree_importances = tree.feature_importances_
            for idx, feature_idx in enumerate(feature_indices):
                if feature_idx < n_features:
                    importances[feature_idx] += tree_importances[idx]
        
        return importances / len(self.estimators_)
    
    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state
        }