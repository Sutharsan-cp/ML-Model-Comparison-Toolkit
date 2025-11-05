import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, final_estimator=None, cv=5, 
                 passthrough=False, random_state=None):
        self.estimators = estimators  # List of (name, estimator) tuples
        self.final_estimator = final_estimator if final_estimator is not None else LinearRegression()
        self.cv = cv
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
        
        # Train base estimators
        self.estimators_ = []
        for name, estimator in self.estimators:
            fitted_estimator = clone(estimator).fit(X, y)
            self.estimators_.append((name, fitted_estimator))
        
        # Generate meta-features using cross-validation
        meta_features = []
        
        for name, estimator in self.estimators:
            cv_predictions = cross_val_predict(estimator, X, y, cv=self.cv, n_jobs=-1)
            meta_features.append(cv_predictions.reshape(-1, 1))
        
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
            predictions = estimator.predict(X)
            meta_features.append(predictions.reshape(-1, 1))
        
        X_meta = np.hstack(meta_features)
        
        if self.passthrough:
            X_meta = np.hstack([X_meta, X])
        
        return self.final_estimator_.predict(X_meta)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return -mean_squared_error(y, predictions)  # Negative MSE for consistency
    
    def get_params(self, deep=True):
        return {
            'estimators': self.estimators,
            'final_estimator': self.final_estimator,
            'cv': self.cv,
            'passthrough': self.passthrough,
            'random_state': self.random_state
        }