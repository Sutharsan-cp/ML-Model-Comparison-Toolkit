import pandas as pd
import numpy as np

class TypeConverter:
    """
    Converts columns into appropriate types:
      - numeric
      - categorical
      - datetime
      - text
    """

    def __init__(self, categorical_threshold=0.05):
        """
        categorical_threshold: fraction of unique values below which
                               a column is treated as categorical
        """
        self.categorical_threshold = categorical_threshold
        self.col_types = {}

    def convert_types(self, df: pd.DataFrame):
        df_new = df.copy()
        meta = {}

        for col in df_new.columns:
            s = df_new[col]

            # Try numeric conversion
            s_num = pd.to_numeric(s, errors="coerce")
            num_null_frac = s_num.isna().mean()

            if num_null_frac < 0.3:  # mostly numeric
                df_new[col] = s_num
                self.col_types[col] = "numeric"
                meta[col] = "numeric"
                continue

            # Datetime check
            try:
                s_date = pd.to_datetime(s, errors="coerce")  # removed deprecated argument
                if s_date.notna().sum() > 0.7 * len(s):  # mostly valid dates
                    df_new[col] = s_date
                    self.col_types[col] = "datetime"
                    meta[col] = "datetime"
                    continue
            except Exception:
                pass

            # Categorical check
            nunique = s.nunique(dropna=True)
            if nunique / max(1, len(s)) <= self.categorical_threshold:
                df_new[col] = s.astype("category")
                # Add MISSING category safely
                if "MISSING" not in df_new[col].cat.categories:
                    df_new[col] = df_new[col].cat.add_categories(["MISSING"])
                df_new[col].fillna("MISSING", inplace=True)
                self.col_types[col] = "categorical"
                meta[col] = "categorical"
            else:
                df_new[col] = s.astype(str)
                df_new[col].fillna("MISSING", inplace=True)  # for text columns
                self.col_types[col] = "text"
                meta[col] = "text"

        return df_new, meta
