import numpy as np
from sklearn.tree import DecisionTreeRegressor

class XGBoostClassifier:
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
    
    def _xgboost_split_criterion(self, gradient_left, hessian_left, gradient_right, hessian_right, reg_lambda):
        """Compute XGBoost split criterion (similar to information gain)"""
        def gain(g, h, reg_lambda):
            return g**2 / (h + reg_lambda)
        
        total_gain = (gain(gradient_left.sum(), hessian_left.sum(), reg_lambda) +
                     gain(gradient_right.sum(), hessian_right.sum(), reg_lambda) -
                     gain(gradient_left.sum() + gradient_right.sum(), 
                         hessian_left.sum() + hessian_right.sum(), reg_lambda))
        
        return total_gain
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("XGBoostClassifier supports only binary classification")
        
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
            
            # Use hessians as sample weights (approximation of XGBoost behavior)
            tree.fit(X_subsample, -gradients_subsample, sample_weight=hessians_subsample)
            
            # For leaf values, compute optimal weight (simplified)
            # In real XGBoost, this would use the Newton-Raphson update
            leaf_predictions = tree.predict(X[:, feature_indices])
            
            # Update predictions with learning rate
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
    
    def get_feature_importance(self, importance_type='weight'):
        """Compute feature importance (simplified version)"""
        if not self.estimators_:
            return None
        
        n_features = self.estimators_[0][1].shape[0]  # Number of features in first tree
        importances = np.zeros(n_features)
        
        for tree, feature_indices in self.estimators_:
            # Use tree's feature importance
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