import pandas as pd

from .data_loader import DataLoader
from .data_cleaning import DataCleaner
from .type_conversion import TypeConverter
from .encoding import CategoricalEncoder
from .scaling import Scaler
from .target_processing import TargetProcessor
from .dimensionality_reduction import DimensionalityReducer
from .feature_engineering import FeatureEngineer


class PreprocessingPipeline:
    """
    Full preprocessing pipeline:
      1. Load dataset
      2. Clean data
      3. Type conversion
      4. Encode categorical features
      5. Scale numeric features
      6. Feature engineering
      7. Dimensionality reduction
      8. Target processing
    """

    def __init__(self, base_dir,
                 encoding_method="auto",
                 scaling_method="auto",
                 fe_method="auto",
                 dr_method="auto"):
        self.loader = DataLoader(base_dir)
        self.cleaner = DataCleaner()
        self.converter = TypeConverter()
        self.encoder = CategoricalEncoder(method=encoding_method)
        self.scaler = Scaler(method=scaling_method)
        self.fe = FeatureEngineer(method=fe_method)
        self.reducer = DimensionalityReducer(method=dr_method)
        self.target_processor = TargetProcessor()

    def preprocess_data(self, filepath: str, target_col: str = None):
        # 1. Load
        df, load_meta = self.loader.load(filepath)

        # 2. Clean
        df, clean_meta = self.cleaner.clean_data(df)

        # 3. Type conversion
        df, type_meta = self.converter.convert_types(df)

        # Detect target
        target_col = self.target_processor.detect_target(df, target_col)

        # Separate X, y
        X = df.drop(columns=[target_col])
        y, target_meta = self.target_processor.process_target(df, target_col)

        # Separate numeric/categorical
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == "object"]

        # 4. Encode categoricals
        X, enc_meta = self.encoder.fit_transform(X, cat_cols)

        # 5. Scale numerics
        X, scale_meta = self.scaler.fit_transform(X, num_cols)

        # 6. Feature engineering
        X, fe_meta = self.fe.transform(X, num_cols)

        # 7. Dimensionality reduction
        X, dr_meta = self.reducer.fit_transform(X, num_cols)

        # Combine
        df_processed = pd.concat([X, y], axis=1)

        # Collect metadata
        meta = {
            "load": load_meta,
            "clean": clean_meta,
            "types": type_meta,
            "encoding": enc_meta,
            "scaling": scale_meta,
            "feature_engineering": fe_meta,
            "dimensionality_reduction": dr_meta,
            "target": target_meta,
            "final_shape": df_processed.shape
        }

        return df_processed, meta
