import numpy as np

class IsolationTree:
    def __init__(self, max_height, random_state=None):
        self.max_height = max_height
        self.random_state = random_state
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.height = 0
        self.is_leaf = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, current_height=0):
        n_samples, n_features = X.shape
        
        if current_height >= self.max_height or n_samples <= 1:
            self.is_leaf = True
            self.height = current_height
            self.n_samples = n_samples
            return self
        
        self.split_feature = np.random.randint(n_features)
        min_val = X[:, self.split_feature].min()
        max_val = X[:, self.split_feature].max()
        
        if min_val == max_val:
            self.is_leaf = True
            self.height = current_height
            self.n_samples = n_samples
            return self
        
        self.split_value = np.random.uniform(min_val, max_val)
        
        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask
        
        self.left = IsolationTree(self.max_height, self.random_state)
        self.right = IsolationTree(self.max_height, self.random_state)
        
        self.left.fit(X[left_mask], current_height + 1)
        self.right.fit(X[right_mask], current_height + 1)
        
        return self
    
    def path_length(self, x):
        if self.is_leaf:
            if self.n_samples <= 1:
                return self.height
            else:
                return self.height + 2 * (np.log(self.n_samples - 1) + 0.577215) - 2 * (self.n_samples - 1) / self.n_samples
        
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', 
                 random_state=None, max_features=1.0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.max_features = max_features
        
        self.estimators_ = []
        self.max_height_ = None
        self.n_samples_ = None
        self.offset_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_max_height(self, n_samples):
        return np.ceil(np.log2(max(1, n_samples)))
    
    def _average_path_length(self, n):
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.577215) - 2 * (n - 1) / n
    
    def fit(self, X):
        X = np.array(X)
        self.n_samples_, n_features = X.shape
        
        # Determine max_samples
        if self.max_samples == 'auto':
            max_samples = min(256, self.n_samples_)
        else:
            max_samples = self.max_samples
        
        # Determine max_features
        if isinstance(self.max_features, float):
            max_features = max(1, int(self.max_features * n_features))
        else:
            max_features = self.max_features
        
        self.max_height_ = self._calculate_max_height(max_samples)
        
        # Build trees
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Sample data
            sample_indices = np.random.choice(self.n_samples_, max_samples, replace=False)
            X_sample = X[sample_indices]
            
            # Sample features if needed
            if max_features < n_features:
                feature_indices = np.random.choice(n_features, max_features, replace=False)
                X_sample = X_sample[:, feature_indices]
            
            tree = IsolationTree(self.max_height_, self.random_state)
            tree.fit(X_sample)
            self.estimators_.append((tree, feature_indices if max_features < n_features else None))
        
        # Calculate offset for anomaly score
        self.offset_ = -self._average_path_length(max_samples)
        
        return self
    
    def decision_function(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        path_lengths = np.zeros(n_samples)
        
        for tree, feature_indices in self.estimators_:
            for i in range(n_samples):
                x = X[i]
                if feature_indices is not None:
                    x = x[feature_indices]
                path_lengths[i] += tree.path_length(x)
        
        average_path_lengths = path_lengths / self.n_estimators
        scores = 2 ** (-average_path_lengths / self._average_path_length(self.max_samples))
        
        return scores + self.offset_
    
    def predict(self, X):
        scores = self.decision_function(X)
        
        if self.contamination == 'auto':
            threshold = 0.5
        else:
            threshold = np.percentile(scores, 100 * self.contamination)
        
        return np.where(scores >= threshold, 1, -1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
    
    def score_samples(self, X):
        return -self.decision_function(X)
    
    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'n_samples': self.n_samples_,
            'max_height': self.max_height_,
            'offset': self.offset_
        }