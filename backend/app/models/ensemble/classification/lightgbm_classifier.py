import numpy as np
from sklearn.tree import DecisionTreeRegressor

class LightGBMClassifier:
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
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _log_odds(self, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))
    
    def _compute_gradient_hessian(self, y, pred):
        """Compute gradient and hessian for binary classification"""
        prob = self._sigmoid(pred)
        gradient = prob - y
        hessian = prob * (1 - prob)
        return gradient, hessian
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("LightGBMClassifier supports only binary classification")
        
        # Convert to binary labels (0, 1)
        y_binary = np.where(y == self.classes_[0], 0, 1)
        
        # Initialize with log-odds
        pos_prob = np.mean(y_binary)
        self.initial_prediction_ = self._log_odds(pos_prob)
        current_predictions = np.full(n_samples, self.initial_prediction_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Compute gradients and hessians
            gradients, hessians = self._compute_gradient_hessian(y_binary, current_predictions)
            
            # Feature fraction (similar to colsample_bytree)
            if self.feature_fraction < 1.0:
                feature_size = int(self.feature_fraction * n_features)
                feature_indices = np.random.choice(n_features, feature_size, replace=False)
            else:
                feature_indices = np.arange(n_features)
            
            # Bagging (subsampling)
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
    
    def predict_proba(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        predictions = np.full(n_samples, self.initial_prediction_)
        
        # Add contributions from all trees
        for tree, feature_indices in self.estimators_:
            X_subset = X[:, feature_indices]
            predictions += self.learning_rate * tree.predict(X_subset)
        
        # Convert to probabilities
        probas_positive = self._sigmoid(predictions)
        probas_negative = 1 - probas_positive
        
        return np.column_stack([probas_negative, probas_positive])
    
    def predict(self, X):
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return np.array([self.classes_[idx] for idx in class_indices])
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
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