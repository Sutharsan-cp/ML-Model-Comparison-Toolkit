import numpy as np
from sklearn.tree import DecisionTreeRegressor

class CatBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.03, depth=6,
                 l2_leaf_reg=3, random_strength=1, bagging_temperature=1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
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
    
    def _compute_ordered_boost_gradient(self, y, pred, sample_indices):
        """Compute ordered boosting gradients (simplified version)"""
        # This is a simplified version - real CatBoost uses permutation-based ordered boosting
        prob = self._sigmoid(pred)
        gradient = prob - y
        return gradient
    
    def _get_bayesian_bootstrap_weights(self, n_samples, temperature=1.0):
        """Generate Bayesian bootstrap weights"""
        if temperature == 0:
            return np.ones(n_samples)
        
        # Generate Dirichlet-distributed weights
        weights = np.random.exponential(scale=temperature, size=n_samples)
        weights /= weights.sum()
        return weights
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("CatBoostClassifier supports only binary classification")
        
        # Convert to binary labels (0, 1)
        y_binary = np.where(y == self.classes_[0], 0, 1)
        
        # Initialize with log-odds
        pos_prob = np.mean(y_binary)
        self.initial_prediction_ = self._log_odds(pos_prob)
        current_predictions = np.full(n_samples, self.initial_prediction_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Generate Bayesian bootstrap weights
            sample_weights = self._get_bayesian_bootstrap_weights(n_samples, self.bagging_temperature)
            
            # Compute gradients (simplified ordered boosting)
            gradients = self._compute_ordered_boost_gradient(y_binary, current_predictions, np.arange(n_samples))
            
            # Fit tree with CatBoost-like parameters
            tree = DecisionTreeRegressor(
                max_depth=self.depth,
                min_samples_leaf=max(1, int(self.l2_leaf_reg)),
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Fit with sample weights
            tree.fit(X, -gradients, sample_weight=sample_weights)
            
            # Add random noise to leaf values (simulating random strength)
            leaf_predictions = tree.predict(X)
            
            if self.random_strength > 0:
                noise = np.random.normal(0, self.random_strength * 0.1, size=leaf_predictions.shape)
                leaf_predictions += noise
            
            # Update predictions
            current_predictions += self.learning_rate * leaf_predictions
            
            self.estimators_.append(tree)
        
        return self
    
    def predict_proba(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        predictions = np.full(n_samples, self.initial_prediction_)
        
        # Add contributions from all trees
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        
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
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_strength': self.random_strength,
            'bagging_temperature': self.bagging_temperature,
            'random_state': self.random_state
        }