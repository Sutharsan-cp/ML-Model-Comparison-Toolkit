import pandas as pd
import numpy as np

class DataCleaner:
    """
    Data cleaning + flexible imputation.

    Features:
      - Drop duplicates
      - Drop high-missing columns
      - Drop constant columns
      - Impute missing values with mean/median/mode/constant/ffill/bfill
      - Auto mode chooses best imputation strategy per column
    """

    def __init__(self, missing_threshold: float = 0.95, impute_strategy="auto", constant_value=0):
        """
        missing_threshold: drop column if %missing > threshold
        impute_strategy: "mean", "median", "mode", "constant", "ffill", "bfill", or "auto"
        constant_value: used if strategy="constant"
        """
        self.missing_threshold = missing_threshold
        self.impute_strategy = impute_strategy
        self.constant_value = constant_value
        self.impute_info = {}

    def _impute_numeric(self, s: pd.Series):
        if self.impute_strategy == "mean":
            val = s.mean()
            return s.fillna(val), {"method": "mean", "value": val}
        elif self.impute_strategy == "median":
            val = s.median()
            return s.fillna(val), {"method": "median", "value": val}
        elif self.impute_strategy == "mode":
            val = s.mode().iloc[0] if not s.mode().empty else 0
            return s.fillna(val), {"method": "mode", "value": val}
        elif self.impute_strategy == "constant":
            return s.fillna(self.constant_value), {"method": "constant", "value": self.constant_value}
        elif self.impute_strategy == "ffill":
            return s.fillna(method="ffill"), {"method": "ffill"}
        elif self.impute_strategy == "bfill":
            return s.fillna(method="bfill"), {"method": "bfill"}
        elif self.impute_strategy == "auto":
            skew = s.skew(skipna=True)
            if abs(skew) > 1:  # highly skewed → median
                val = s.median()
                return s.fillna(val), {"method": "median", "value": val}
            else:
                val = s.mean()
                return s.fillna(val), {"method": "mean", "value": val}
        else:
            return s, {"method": "none"}

    def _impute_categorical(self, s: pd.Series):
        if self.impute_strategy == "mode" or self.impute_strategy == "auto":
            val = s.mode().iloc[0] if not s.mode().empty else "MISSING"
            return s.fillna(val), {"method": "mode", "value": val}
        elif self.impute_strategy == "constant":
            return s.fillna(self.constant_value), {"method": "constant", "value": self.constant_value}
        elif self.impute_strategy in ["ffill", "bfill"]:
            return s.fillna(method=self.impute_strategy), {"method": self.impute_strategy}
        else:
            return s.fillna("MISSING"), {"method": "fallback", "value": "MISSING"}

    def clean_data(self, df: pd.DataFrame):
        summary = {}
        df = df.copy()

        # Drop duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates()
        summary["duplicates_removed"] = int(dup_count)

        # Drop high missing columns
        missing_pct = df.isna().mean()
        drop_cols = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
        summary["dropped_high_missing"] = drop_cols

        # Drop constant columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
        summary["constant_columns"] = constant_cols

        # Imputation
        impute_meta = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col], info = self._impute_numeric(df[col])
            else:
                df[col], info = self._impute_categorical(df[col])
            impute_meta[col] = info

        summary["imputation"] = impute_meta
        summary["final_shape"] = df.shape

        return df, summary
