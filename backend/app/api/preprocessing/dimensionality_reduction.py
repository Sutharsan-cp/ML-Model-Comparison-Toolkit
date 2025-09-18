import pandas as pd
import numpy as np

class DimensionalityReducer:
    """
    Flexible dimensionality reduction with multiple techniques.

    Methods:
      - pca
      - svd
      - lda (requires labels)
      - random
      - variance_threshold
      - correlation_filter
      - auto (default: picks based on dataset)

    Usage:
      reducer = DimensionalityReducer(method="pca")
      X_new, meta = reducer.fit_transform(X, y, num_cols)
    """

    def __init__(self, method="auto", explained_variance_threshold=0.95,
                 n_components=None, variance_threshold=0.0,
                 correlation_threshold=0.95, random_dim=10):
        self.method = method
        self.thresh = explained_variance_threshold
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_dim = random_dim

    # -------- PCA --------
    def _pca(self, X):
        Xc = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        explained = (S**2) / (len(X) - 1)
        explained_ratio = explained / explained.sum()
        cum = explained_ratio.cumsum()
        n_comp = self.n_components or (cum < self.thresh).sum() + 1
        Z = np.dot(Xc, Vt[:n_comp].T)
        return Z, {"method": "pca", "n_components": n_comp, "explained_ratio": explained_ratio[:n_comp].tolist()}

    # -------- SVD --------
    def _svd(self, X):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        n_comp = self.n_components or min(X.shape)
        Z = np.dot(X, Vt[:n_comp].T)
        return Z, {"method": "svd", "n_components": n_comp}

    # -------- LDA --------
    def _lda(self, X, y):
        classes = np.unique(y)
        mean_overall = np.mean(X, axis=0)

        Sw = np.zeros((X.shape[1], X.shape[1]))
        Sb = np.zeros((X.shape[1], X.shape[1]))

        for c in classes:
            Xc = X[y == c]
            mean_c = np.mean(Xc, axis=0)
            Sw += np.dot((Xc - mean_c).T, (Xc - mean_c))
            n_c = Xc.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(-1,1)
            Sb += n_c * (mean_diff @ mean_diff.T)

        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
        idx = np.argsort(-eigvals.real)
        eigvecs = eigvecs[:, idx]
        n_comp = self.n_components or min(len(classes)-1, X.shape[1])
        W = eigvecs[:, :n_comp].real
        Z = X @ W
        return Z, {"method": "lda", "n_components": n_comp}

    # -------- Random Projection --------
    def _random(self, X):
        n_comp = self.n_components or min(self.random_dim, X.shape[1])
        R = np.random.randn(X.shape[1], n_comp)
        Z = X @ R
        return Z, {"method": "random", "n_components": n_comp}

    # -------- Variance Threshold --------
    def _variance_threshold(self, X, cols):
        variances = X.var(axis=0)
        keep_idx = np.where(variances > self.variance_threshold)[0]
        Z = X[:, keep_idx]
        kept_cols = [cols[i] for i in keep_idx]
        return Z, {"method": "variance_threshold", "kept_columns": kept_cols}

    # -------- Correlation Filter --------
    def _correlation_filter(self, df, cols):
        corr = df[cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        kept_cols = [c for c in cols if c not in to_drop]
        return df[kept_cols].values, {"method": "correlation_filter", "dropped": to_drop, "kept": kept_cols}

    # -------- Main Entry --------
    def fit_transform(self, df: pd.DataFrame, num_cols: list, y=None):
        if not num_cols:
            return df, {}

        X = df[num_cols].fillna(0).values

        # auto mode selection
        method = self.method
        if method == "auto":
            if y is not None and len(np.unique(y)) < len(y) / 2:
                method = "lda"
            elif X.shape[1] > X.shape[0]:
                method = "svd"
            else:
                method = "pca"

        if method == "pca":
            Z, meta = self._pca(X)
            pc_names = [f"PC{i+1}" for i in range(Z.shape[1])]
            df_new = pd.concat([df.drop(columns=num_cols), pd.DataFrame(Z, columns=pc_names, index=df.index)], axis=1)
            return df_new, meta

        elif method == "svd":
            Z, meta = self._svd(X)
            comp_names = [f"SVD{i+1}" for i in range(Z.shape[1])]
            df_new = pd.concat([df.drop(columns=num_cols), pd.DataFrame(Z, columns=comp_names, index=df.index)], axis=1)
            return df_new, meta

        elif method == "lda" and y is not None:
            Z, meta = self._lda(X, y.values)
            comp_names = [f"LDA{i+1}" for i in range(Z.shape[1])]
            df_new = pd.concat([df.drop(columns=num_cols), pd.DataFrame(Z, columns=comp_names, index=df.index)], axis=1)
            return df_new, meta

        elif method == "random":
            Z, meta = self._random(X)
            comp_names = [f"RP{i+1}" for i in range(Z.shape[1])]
            df_new = pd.concat([df.drop(columns=num_cols), pd.DataFrame(Z, columns=comp_names, index=df.index)], axis=1)
            return df_new, meta

        elif method == "variance_threshold":
            Z, meta = self._variance_threshold(X, num_cols)
            kept_cols = meta['kept_columns']
            df_new = pd.concat([df.drop(columns=num_cols), pd.DataFrame(Z, columns=kept_cols, index=df.index)], axis=1)
            return df_new, meta

        elif method == "correlation_filter":
            Z, meta = self._correlation_filter(df, num_cols)
            kept_cols = meta['kept']
            df_new = pd.concat([df.drop(columns=num_cols), pd.DataFrame(Z, columns=kept_cols, index=df.index)], axis=1)
            return df_new, meta

        else:
            return df, {"method": "none"}
