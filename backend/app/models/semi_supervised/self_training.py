import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class SelfTrainingClassifier:
    def __init__(self, base_estimator, threshold=0.9, max_iter=50, 
                 criterion='threshold', random_state=None):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.max_iter = max_iter
        self.criterion = criterion
        self.random_state = random_state
        
        self.iteration_ = 0
        self.labeled_iter_ = None
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Initialize
        n_samples = X.shape[0]
        self.labeled_iter_ = np.full(n_samples, -1)
        
        labeled_mask = y != -1
        unlabeled_mask = ~labeled_mask
        
        self.classes_ = np.unique(y[labeled_mask])
        
        # Initial training on labeled data
        X_labeled = X[labeled_mask]
        y_labeled = y[labeled_mask]
        
        self.base_estimator.fit(X_labeled, y_labeled)
        self.labeled_iter_[labeled_mask] = 0
        
        self.iteration_ = 1
        
        while self.iteration_ <= self.max_iter and np.any(unlabeled_mask):
            # Predict probabilities for unlabeled data
            X_unlabeled = X[unlabeled_mask]
            
            if hasattr(self.base_estimator, 'predict_proba'):
                probas = self.base_estimator.predict_proba(X_unlabeled)
                max_probas = np.max(probas, axis=1)
                predictions = self.base_estimator.predict(X_unlabeled)
            else:
                # For estimators without predict_proba, use decision function
                predictions = self.base_estimator.predict(X_unlabeled)
                max_probas = np.ones(len(predictions))  # Assume high confidence
                
                if hasattr(self.base_estimator, 'decision_function'):
                    decision_scores = self.base_estimator.decision_function(X_unlabeled)
                    if decision_scores.ndim == 1:
                        max_probas = np.abs(decision_scores)
                    else:
                        max_probas = np.max(decision_scores, axis=1)
            
            # Select samples to label based on criterion
            if self.criterion == 'threshold':
                high_confidence = max_probas >= self.threshold
            elif self.criterion == 'k_best':
                k = min(len(max_probas), int(self.threshold * len(max_probas)))
                if k > 0:
                    high_confidence = np.zeros_like(max_probas, dtype=bool)
                    top_k_indices = np.argpartition(max_probas, -k)[-k:]
                    high_confidence[top_k_indices] = True
                else:
                    high_confidence = np.zeros_like(max_probas, dtype=bool)
            else:
                raise ValueError(f"Unsupported criterion: {self.criterion}")
            
            if not np.any(high_confidence):
                break
            
            # Get indices of newly labeled samples
            unlabeled_indices = np.where(unlabeled_mask)[0]
            new_labeled_indices = unlabeled_indices[high_confidence]
            new_labels = predictions[high_confidence]
            
            # Update labels
            y[new_labeled_indices] = new_labels
            self.labeled_iter_[new_labeled_indices] = self.iteration_
            
            # Update masks
            labeled_mask = y != -1
            unlabeled_mask = ~labeled_mask
            
            # Retrain on expanded labeled set
            X_labeled = X[labeled_mask]
            y_labeled = y[labeled_mask]
            
            self.base_estimator.fit(X_labeled, y_labeled)
            
            self.iteration_ += 1
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        return self.base_estimator.predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self)
        if hasattr(self.base_estimator, 'predict_proba'):
            return self.base_estimator.predict_proba(X)
        else:
            raise NotImplementedError("Base estimator does not support predict_proba")
    
    def get_params(self):
        return {
            'threshold': self.threshold,
            'max_iter': self.max_iter,
            'criterion': self.criterion,
            'iteration': self.iteration_,
            'n_labeled_samples': np.sum(self.labeled_iter_ != -1) if hasattr(self, 'labeled_iter_') else 0
        }