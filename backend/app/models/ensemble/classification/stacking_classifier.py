import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, final_estimator=None, cv=5, 
                 use_probas=False, passthrough=False, random_state=None):
        self.estimators = estimators  # List of (name, estimator) tuples
        self.final_estimator = final_estimator if final_estimator is not None else LogisticRegression()
        self.cv = cv
        self.use_probas = use_probas
        self.passthrough = passthrough
        self.random_state = random_state
        
        self.estimators_ = []
        self.final_estimator_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Train base estimators
        self.estimators_ = []
        for name, estimator in self.estimators:
            fitted_estimator = clone(estimator).fit(X, y)
            self.estimators_.append((name, fitted_estimator))
        
        # Generate meta-features using cross-validation
        meta_features = []
        
        for name, estimator in self.estimators:
            if self.use_probas:
                # Use predicted probabilities
                cv_probas = cross_val_predict(estimator, X, y, cv=self.cv, 
                                            method='predict_proba', n_jobs=-1)
                meta_features.append(cv_probas)
            else:
                # Use predicted classes
                cv_predictions = cross_val_predict(estimator, X, y, cv=self.cv, n_jobs=-1)
                # Convert to one-hot encoding
                one_hot = np.zeros((n_samples, n_classes))
                for i, pred in enumerate(cv_predictions):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    one_hot[i, class_idx] = 1
                meta_features.append(one_hot)
        
        # Stack meta-features
        X_meta = np.hstack(meta_features)
        
        # Add original features if passthrough is True
        if self.passthrough:
            X_meta = np.hstack([X_meta, X])
        
        # Train final estimator on meta-features
        self.final_estimator_ = clone(self.final_estimator).fit(X_meta, y)
        
        # Retrain base estimators on full data
        self.estimators_ = []
        for name, estimator in self.estimators:
            fitted_estimator = clone(estimator).fit(X, y)
            self.estimators_.append((name, fitted_estimator))
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        meta_features = []
        
        for name, estimator in self.estimators_:
            if self.use_probas:
                probas = estimator.predict_proba(X)
                meta_features.append(probas)
            else:
                predictions = estimator.predict(X)
                one_hot = np.zeros((X.shape[0], len(self.classes_)))
                for i, pred in enumerate(predictions):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    one_hot[i, class_idx] = 1
                meta_features.append(one_hot)
        
        X_meta = np.hstack(meta_features)
        
        if self.passthrough:
            X_meta = np.hstack([X_meta, X])
        
        return self.final_estimator_.predict(X_meta)
    
    def predict_proba(self, X):
        X = np.array(X)
        meta_features = []
        
        for name, estimator in self.estimators_:
            if self.use_probas:
                probas = estimator.predict_proba(X)
                meta_features.append(probas)
            else:
                predictions = estimator.predict(X)
                one_hot = np.zeros((X.shape[0], len(self.classes_)))
                for i, pred in enumerate(predictions):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    one_hot[i, class_idx] = 1
                meta_features.append(one_hot)
        
        X_meta = np.hstack(meta_features)
        
        if self.passthrough:
            X_meta = np.hstack([X_meta, X])
        
        if hasattr(self.final_estimator_, 'predict_proba'):
            return self.final_estimator_.predict_proba(X_meta)
        else:
            # If no predict_proba, return one-hot encoding based on predictions
            predictions = self.final_estimator_.predict(X_meta)
            probas = np.zeros((X.shape[0], len(self.classes_)))
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                probas[i, class_idx] = 1
            return probas
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        return {
            'estimators': self.estimators,
            'final_estimator': self.final_estimator,
            'cv': self.cv,
            'use_probas': self.use_probas,
            'passthrough': self.passthrough,
            'random_state': self.random_state
        }