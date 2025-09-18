import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Flexible feature engineering for numeric features.

    Supports:
      - polynomial: add squared & interaction terms
      - log: log transform (for skewed features)
      - sqrt: square-root transform
      - binarize: binary feature (x > mean/median)
      - discretize: binning continuous features
      - interactions: pairwise ratios & products
      - auto: choose best based on skewness
    """

    def __init__(self, method="auto", bins=5, binning="quantile"):
        """
        method: str, feature engineering method
        bins: int, number of bins if method="discretize"
        binning: "quantile" or "uniform" binning
        """
        self.method = method
        self.bins = bins
        self.binning = binning

    def _polynomial(self, df, num_cols):
        result = df.copy()
        for c in num_cols:
            result[f"{c}__sq"] = result[c] ** 2
        # add interactions (limited to first few to avoid explosion)
        for i in range(min(len(num_cols), 5)):
            for j in range(i+1, min(len(num_cols), 5)):
                a, b = num_cols[i], num_cols[j]
                result[f"{a}__x__{b}"] = result[a] * result[b]
        return result, {"method": "polynomial"}

    def _log(self, df, num_cols):
        result = df.copy()
        for c in num_cols:
            result[f"{c}__log"] = np.log1p(result[c].clip(lower=0))
        return result, {"method": "log"}

    def _sqrt(self, df, num_cols):
        result = df.copy()
        for c in num_cols:
            result[f"{c}__sqrt"] = np.sqrt(result[c].clip(lower=0))
        return result, {"method": "sqrt"}

    def _binarize(self, df, num_cols):
        result = df.copy()
        for c in num_cols:
            thr = result[c].median()
            result[f"{c}__bin"] = (result[c] > thr).astype(int)
        return result, {"method": "binarize"}

    def _discretize(self, df, num_cols):
        result = df.copy()
        meta = {"method": "discretize", "bins": {}}
        for c in num_cols:
            if self.binning == "quantile":
                result[f"{c}__bin"] = pd.qcut(result[c], self.bins, labels=False, duplicates="drop")
            else:  # uniform width
                result[f"{c}__bin"] = pd.cut(result[c], self.bins, labels=False)
            meta["bins"][c] = self.bins
        return result, meta

    def _interactions(self, df, num_cols):
        result = df.copy()
        meta = {"method": "interactions"}
        for i in range(min(len(num_cols), 5)):
            for j in range(i+1, min(len(num_cols), 5)):
                a, b = num_cols[i], num_cols[j]
                result[f"{a}__div__{b}"] = result[a] / (result[b].replace(0, np.nan))
        return result.fillna(0), meta

    def _auto(self, df, num_cols):
        result = df.copy()
        meta = {"method": "auto", "applied": {}}
        for c in num_cols:
            skew = result[c].skew()
            if abs(skew) > 1:  # highly skewed → log
                result[f"{c}__log"] = np.log1p(result[c].clip(lower=0))
                meta["applied"][c] = "log"
            else:  # mildly skewed → polynomial
                result[f"{c}__sq"] = result[c] ** 2
                meta["applied"][c] = "polynomial"
        return result, meta

    def transform(self, df: pd.DataFrame, num_cols: list):
        if not num_cols:
            return df, {"method": "none"}

        if self.method == "polynomial":
            return self._polynomial(df, num_cols)
        elif self.method == "log":
            return self._log(df, num_cols)
        elif self.method == "sqrt":
            return self._sqrt(df, num_cols)
        elif self.method == "binarize":
            return self._binarize(df, num_cols)
        elif self.method == "discretize":
            return self._discretize(df, num_cols)
        elif self.method == "interactions":
            return self._interactions(df, num_cols)
        elif self.method == "auto":
            return self._auto(df, num_cols)
        else:
            return df, {"method": "none"}
