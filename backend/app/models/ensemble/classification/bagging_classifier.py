import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

class BaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, 
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        
        self.estimators_ = []
        self.estimator_features_ = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        n_samples_bootstrap = int(self.max_samples * n_samples)
        n_features_bootstrap = int(self.max_features * n_features)
        
        self.estimators_ = []
        self.estimator_features_ = []
        self.classes_ = np.unique(y)
        
        for i in range(self.n_estimators):
            # Clone base estimator
            estimator = clone(self.base_estimator)
            
            # Sample data
            if self.bootstrap:
                sample_indices = resample(range(n_samples), 
                                        n_samples=n_samples_bootstrap,
                                        random_state=self.random_state + i if self.random_state else None)
            else:
                sample_indices = np.random.choice(n_samples, n_samples_bootstrap, replace=False)
            
            # Sample features
            if self.bootstrap_features:
                feature_indices = resample(range(n_features),
                                         n_samples=n_features_bootstrap,
                                         random_state=self.random_state + i if self.random_state else None)
            else:
                feature_indices = np.random.choice(n_features, n_features_bootstrap, replace=False)
            
            X_bootstrap = X[sample_indices][:, feature_indices]
            y_bootstrap = y[sample_indices]
            
            # Train estimator
            estimator.fit(X_bootstrap, y_bootstrap)
            
            self.estimators_.append(estimator)
            self.estimator_features_.append(feature_indices)
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        
        for i, (estimator, feature_indices) in enumerate(zip(self.estimators_, self.estimator_features_)):
            X_subset = X[:, feature_indices]
            predictions[:, i] = estimator.predict(X_subset)
        
        # Majority voting
        final_predictions = []
        for sample_predictions in predictions:
            unique, counts = np.unique(sample_predictions, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        X = np.array(X)
        probas = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, (estimator, feature_indices) in enumerate(zip(self.estimators_, self.estimator_features_)):
            X_subset = X[:, feature_indices]
            if hasattr(estimator, 'predict_proba'):
                probas += estimator.predict_proba(X_subset)
            else:
                # If no predict_proba, use one-hot encoding
                predictions = estimator.predict(X_subset)
                for j, pred in enumerate(predictions):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    probas[j, class_idx] += 1
        
        return probas / len(self.estimators_)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        return {
            'base_estimator': self.base_estimator,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'random_state': self.random_state
        }