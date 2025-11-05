import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        
        self.estimators_ = []
        self.initial_prediction_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize with mean
        self.initial_prediction_ = np.mean(y)
        current_predictions = np.full(n_samples, self.initial_prediction_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - current_predictions
            
            # Subsample data
            if self.subsample < 1.0:
                subsample_size = int(self.subsample * n_samples)
                sample_indices = np.random.choice(n_samples, subsample_size, replace=False)
                X_subsample = X[sample_indices]
                residuals_subsample = residuals[sample_indices]
            else:
                X_subsample = X
                residuals_subsample = residuals
            
            # Fit regression tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i if self.random_state else None
            )
            
            tree.fit(X_subsample, residuals_subsample)
            
            # Update predictions
            tree_predictions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predictions
            
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        predictions = np.full(n_samples, self.initial_prediction_)
        
        # Add contributions from all trees
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def score(self, X, y):
        predictions = self.predict(X)
        return -mean_squared_error(y, predictions)  # Negative MSE for consistency
    
    def staged_predict(self, X):
        """Predict at each stage of boosting"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        predictions = np.full(n_samples, self.initial_prediction_)
        
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
            yield predictions.copy()
    
    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
            'random_state': self.random_state
        }