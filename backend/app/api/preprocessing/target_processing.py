import pandas as pd
import numpy as np

class TargetProcessor:
    """
    Detects and processes target variable for regression/classification.
    """

    def __init__(self):
        self.classes_ = None
        self.task_type = None  # "regression" or "classification"

    def detect_target(self, df: pd.DataFrame, target_col: str = None):
        """Pick target column (last col if not specified)."""
        if target_col is None:
            target_col = df.columns[-1]
        return target_col

    def process_target(self, df: pd.DataFrame, target_col: str = None):
        target_col = self.detect_target(df, target_col)
        y = df[target_col]

        # Decide type: regression vs classification
        if pd.api.types.is_numeric_dtype(y):
            self.task_type = "regression"
            y_clean = pd.to_numeric(y, errors="coerce")
            y_clean = y_clean.fillna(y_clean.median())
            return y_clean, {"target": target_col, "task_type": "regression"}

        else:
            self.task_type = "classification"
            y_clean = y.astype(str).fillna("MISSING")
            self.classes_ = sorted(y_clean.unique())
            mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
            y_encoded = y_clean.map(mapping)
            return y_encoded, {
                "target": target_col,
                "task_type": "classification",
                "classes": mapping,
            }
