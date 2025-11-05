import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

class GradientBoostingClassifier:
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
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _log_odds(self, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier supports only binary classification")
        
        # Convert to binary labels (0, 1)
        y_binary = np.where(y == self.classes_[0], 0, 1)
        
        # Initialize with log-odds
        pos_prob = np.mean(y_binary)
        self.initial_prediction_ = self._log_odds(pos_prob)
        current_predictions = np.full(n_samples, self.initial_prediction_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Convert to probabilities
            current_probs = self._sigmoid(current_predictions)
            
            # Compute pseudo-residuals (negative gradient)
            residuals = y_binary - current_probs
            
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
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def staged_predict_proba(self, X):
        """Predict probabilities at each stage of boosting"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        predictions = np.full(n_samples, self.initial_prediction_)
        
        for i, tree in enumerate(self.estimators_):
            predictions += self.learning_rate * tree.predict(X)
            probas_positive = self._sigmoid(predictions)
            probas_negative = 1 - probas_positive
            yield np.column_stack([probas_negative, probas_positive])
    
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