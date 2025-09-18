import os
import pandas as pd

class DataLoader:
    """
    Loads datasets in different formats (CSV, TSV, Excel: xlsx/xls).
    Converts Excel files to CSV before processing.
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load(self, filepath: str):
        """Load dataset given a relative or absolute path."""
        full_path = os.path.join(self.base_dir, filepath) if not os.path.isabs(filepath) else filepath
        ext = os.path.splitext(full_path)[-1].lower()

        try:
            # If Excel, convert to CSV first
            if ext in [".xlsx", ".xls"]:
                try:
                    if ext == ".xlsx":
                        df = pd.read_excel(full_path, engine="openpyxl")
                    else:  # .xls
                        df = pd.read_excel(full_path, engine="xlrd")
                except Exception as e:
                    raise RuntimeError(f"Failed to read Excel file {full_path}: {e}")

                # Save as CSV in the same directory
                csv_path = os.path.splitext(full_path)[0] + "_converted.csv"
                df.to_csv(csv_path, index=False)
                print(f"✅ Converted {full_path} to {csv_path}")
                full_path = csv_path
                ext = ".csv"

            # Now read as CSV/TSV
            if ext == ".csv":
                df = pd.read_csv(full_path)
            elif ext in [".tsv", ".txt"]:
                df = pd.read_csv(full_path, sep="\t")
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Optional: Add basic preprocessing for missing categorical values
            for col in df.columns:
                if pd.api.types.is_categorical_dtype(df[col]):
                    if "MISSING" not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories(["MISSING"])
                    df[col].fillna("MISSING", inplace=True)

            return df, {"path": full_path, "rows": df.shape[0], "cols": df.shape[1]}

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {full_path}: {e}")
