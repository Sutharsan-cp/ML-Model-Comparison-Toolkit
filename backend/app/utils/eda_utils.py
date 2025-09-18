import numpy as np
import pandas as pd
import scipy.stats as stats
from . import statistics_utils as su  # relative import

def summary(data, feature_names=None):
    """
    Return a dictionary of stats for both numeric and categorical features.
    data: 2D array-like or pandas DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=feature_names)

    summary_dict = {}

    for col in data.columns:
        col_data = data[col].dropna()

        if pd.api.types.is_numeric_dtype(col_data):
            # Numeric feature → delegate to stats_utils
            q1, q3 = np.percentile(col_data, [25, 75])
            summary_dict[col] = {
                "type": "numeric",
                "mean": su.mean(col_data),
                "median": su.median(col_data),
                "mode": col_data.mode().iloc[0] if not col_data.mode().empty else None,
                "variance": su.variance(col_data),
                "std_dev": su.std_dev(col_data),
                "skewness": su.skewness(col_data),
                "kurtosis": su.kurtosis(col_data),
                "min": su.min_value(col_data),
                "max": su.max_value(col_data),
                "range": su.range_value(col_data),
                "q1": q1,
                "q3": q3,
                "missing_count": data[col].isna().sum(),
                "unique_count": col_data.nunique()
            }
        else:
            # Categorical feature
            value_counts = col_data.value_counts()
            summary_dict[col] = {
                "type": "categorical",
                "mode": col_data.mode().iloc[0] if not col_data.mode().empty else None,
                "unique_values": col_data.unique().tolist(),
                "unique_count": col_data.nunique(),
                "top_5_frequent": value_counts.head(5).to_dict(),
                "missing_count": data[col].isna().sum(),
            }

    return summary_dict
