import pandas as pd
import numpy as np

class Scaler:
    """
    Flexible scaling/normalization techniques.

    Supports:
      - standard: Z-score (mean=0, std=1)
      - minmax: [0,1] range
      - maxabs: divide by max absolute value
      - robust: median and IQR scaling
      - l1: normalize rows so sum(abs(x))=1
      - l2: normalize rows so ||x||_2=1
      - auto: automatic selection based on data

    Usage:
      scaler = Scaler(method="standard")
      X_scaled, meta = scaler.fit_transform(X, num_cols)
    """

    def __init__(self, method="standard", feature_range=(0, 1)):
        self.method = method
        self.feature_range = feature_range
        self.params = {}

    # -------- Standard Scaling --------
    def _standard(self, s):
        mean, std = s.mean(), s.std() or 1
        return (s - mean) / std, {"mean": mean, "std": std}

    # -------- MinMax Scaling --------
    def _minmax(self, s):
        mn, mx = s.min(), s.max()
        rng = mx - mn or 1
        low, high = self.feature_range
        scaled = (s - mn) / rng
        return scaled * (high - low) + low, {"min": mn, "max": mx}

    # -------- MaxAbs Scaling --------
    def _maxabs(self, s):
        max_abs = s.abs().max() or 1
        return s / max_abs, {"max_abs": max_abs}

    # -------- Robust Scaling --------
    def _robust(self, s):
        med = s.median()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1 or 1
        return (s - med) / iqr, {"median": med, "iqr": iqr}

    # -------- Row Normalization (L1/L2) --------
    def _normalize_rows(self, df, num_cols, norm="l2"):
        X = df[num_cols].values.astype(float)
        if norm == "l1":
            denom = np.sum(np.abs(X), axis=1, keepdims=True)
        else:  # l2
            denom = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
        denom[denom == 0] = 1
        X_norm = X / denom
        df_new = df.copy()
        df_new[num_cols] = X_norm
        return df_new, {"norm": norm}

    # -------- Auto Selection --------
    def _auto(self, s):
        if s.max() <= 1 and s.min() >= 0:
            return s, {"method": "none"}
        if (s.max() - s.min()) > (10 * s.median() if s.median() != 0 else 1e-6):
            return self._robust(s)
        return self._standard(s)

    # -------- Fit & Transform --------
    def fit_transform(self, df: pd.DataFrame, num_cols: list):
        result = df.copy()
        meta = {}

        if not num_cols:
            return result, meta

        if self.method in ["l1", "l2"]:
            result, meta = self._normalize_rows(result, num_cols, norm=self.method)
            return result, meta

        for c in num_cols:
            if self.method == "standard":
                result[c], info = self._standard(result[c])
            elif self.method == "minmax":
                result[c], info = self._minmax(result[c])
            elif self.method == "maxabs":
                result[c], info = self._maxabs(result[c])
            elif self.method == "robust":
                result[c], info = self._robust(result[c])
            elif self.method == "auto":
                result[c], info = self._auto(result[c])
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")

            self.params[c] = info

        meta["scaled_columns"] = num_cols
        meta["method"] = self.method
        return result, meta
