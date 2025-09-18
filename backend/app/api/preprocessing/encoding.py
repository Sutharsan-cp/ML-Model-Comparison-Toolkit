import pandas as pd
import numpy as np
import hashlib

class CategoricalEncoder:
    """
    Flexible encoder for categorical features.

    Supports:
      - onehot: dummy variables
      - ordinal: map categories to integers
      - frequency: map categories to relative frequency
      - binary: binary digit expansion of category index
      - hashing: hash categories into fixed bins

    Usage:
      encoder = CategoricalEncoder(method="onehot")
      X, meta = encoder.fit_transform(df, cat_cols)

      # OR per-column methods:
      encoder = CategoricalEncoder(methods={"col1": "onehot", "col2": "frequency"})
    """

    def __init__(self, method="auto", methods=None, hashing_bins=8):
        """
        method: str, default="auto" → choose automatically
        methods: dict, column → encoding method
        hashing_bins: int, number of bins for hashing encoder
        """
        self.method = method
        self.methods = methods or {}
        self.hashing_bins = hashing_bins
        self.enc_info = {}

    def _onehot(self, df, col):
        dummies = pd.get_dummies(df[col].fillna("MISSING"), prefix=col)
        return dummies, {"method": "onehot", "columns": list(dummies.columns)}

    def _ordinal(self, df, col):
        cats = {v: i for i, v in enumerate(df[col].dropna().unique(), start=0)}
        mapped = df[col].map(cats).fillna(-1).astype(int)
        return mapped.to_frame(col), {"method": "ordinal", "mapping": cats}

    def _frequency(self, df, col):
        freq = df[col].value_counts(normalize=True)
        mapped = df[col].map(freq).fillna(0)
        return mapped.to_frame(col), {"method": "frequency", "mapping": freq.to_dict()}

    def _binary(self, df, col):
        cats = {v: i for i, v in enumerate(df[col].dropna().unique(), start=0)}
        max_val = max(cats.values()) if cats else 0
        n_bits = max_val.bit_length() or 1
        codes = df[col].map(cats).fillna(-1).astype(int)

        bin_cols = []
        for b in range(n_bits):
            bin_cols.append(codes.apply(lambda x: (x >> b) & 1 if x >= 0 else 0))
        df_bin = pd.concat(bin_cols, axis=1)
        df_bin.columns = [f"{col}_bin{b}" for b in range(n_bits)]
        return df_bin, {"method": "binary", "mapping": cats, "n_bits": n_bits}

    def _hashing(self, df, col):
        def hash_val(x):
            if pd.isna(x):
                return 0
            h = int(hashlib.md5(str(x).encode()).hexdigest(), 16)
            return h % self.hashing_bins

        hashed = df[col].apply(hash_val)
        return hashed.to_frame(col), {"method": "hashing", "bins": self.hashing_bins}

    def fit_transform(self, df: pd.DataFrame, cat_cols: list):
        result = df.copy()
        meta = {}

        for col in cat_cols:
            # decide method
            method = self.methods.get(col, self.method)

            if method == "onehot":
                new_df, info = self._onehot(result, col)
                result = pd.concat([result.drop(columns=[col]), new_df], axis=1)

            elif method == "ordinal":
                new_df, info = self._ordinal(result, col)
                result[col] = new_df[col]

            elif method == "frequency":
                new_df, info = self._frequency(result, col)
                result[col] = new_df[col]

            elif method == "binary":
                new_df, info = self._binary(result, col)
                result = pd.concat([result.drop(columns=[col]), new_df], axis=1)

            elif method == "hashing":
                new_df, info = self._hashing(result, col)
                result[col] = new_df[col]

            elif method == "auto":
                # heuristic: onehot if <=10 unique, else ordinal
                n_unique = result[col].nunique(dropna=False)
                if n_unique <= 10:
                    new_df, info = self._onehot(result, col)
                    result = pd.concat([result.drop(columns=[col]), new_df], axis=1)
                else:
                    new_df, info = self._ordinal(result, col)
                    result[col] = new_df[col]

            else:
                raise ValueError(f"Unknown encoding method: {method}")

            meta[col] = info

        return result, meta
