import os
import sys
from pathlib import Path
import pandas as pd

from api.preprocessing.preprocessing_pipeline import PreprocessingPipeline

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "rawDatasets"
PROCESSED_DIR = BASE_DIR / "data" / "processedDatasets"

# Ensure processed directories exist
(PROCESSED_DIR / "regression").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "classification").mkdir(parents=True, exist_ok=True)

def process_dataset(pipeline, filepath: Path, task: str):
    print(f"\nProcessing: {filepath.name}")
    try:
        df_processed, meta = pipeline.preprocess_data(str(filepath))
        print(f"✅ Completed. Final shape: {df_processed.shape}, Task: {meta['target']['task_type']}")

        # Save processed dataset
        out_dir = PROCESSED_DIR / meta["target"]["task_type"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filepath.name.replace(filepath.suffix, "_processed.csv")
        df_processed.to_csv(out_path, index=False)
        print(f"💾 Saved processed dataset to: {out_path}\n")

    except Exception as e:
        print(f"❌ Failed to process {filepath.name}: {e}")


def list_datasets():
    reg_files = list((RAW_DIR / "regression").glob("*.*"))
    cls_files = list((RAW_DIR / "classification").glob("*.*"))
    return reg_files, cls_files


def main():
    pipeline = PreprocessingPipeline(
        base_dir=RAW_DIR,
        encoding_method="auto",
        scaling_method="auto",
        fe_method="auto",
        dr_method="auto"
    )

    while True:
        print("\n" + "="*60)
        print(" DATASET PREPROCESSING SYSTEM")
        print("="*60)
        print("\nMain Menu:")
        print("----------------------------------------")
        print("1. Process Regression Datasets")
        print("2. Process Classification Datasets")
        print("3. Process All Datasets")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        reg_files, cls_files = list_datasets()

        if choice == "1":
            print("\nREGRESSION DATASETS:")
            for i, f in enumerate(reg_files, start=1):
                print(f"{i}. {f.name}")
            print(f"{len(reg_files)+1}. Process ALL regression datasets")
            print(f"{len(reg_files)+2}. Back to Main Menu")

            sub_choice = input("\nSelect regression dataset: ").strip()
            if sub_choice.isdigit():
                sub_choice = int(sub_choice)
                if 1 <= sub_choice <= len(reg_files):
                    process_dataset(pipeline, reg_files[sub_choice-1], "regression")
                elif sub_choice == len(reg_files)+1:
                    for f in reg_files:
                        process_dataset(pipeline, f, "regression")
                else:
                    continue

        elif choice == "2":
            print("\nCLASSIFICATION DATASETS:")
            for i, f in enumerate(cls_files, start=1):
                print(f"{i}. {f.name}")
            print(f"{len(cls_files)+1}. Process ALL classification datasets")
            print(f"{len(cls_files)+2}. Back to Main Menu")

            sub_choice = input("\nSelect classification dataset: ").strip()
            if sub_choice.isdigit():
                sub_choice = int(sub_choice)
                if 1 <= sub_choice <= len(cls_files):
                    process_dataset(pipeline, cls_files[sub_choice-1], "classification")
                elif sub_choice == len(cls_files)+1:
                    for f in cls_files:
                        process_dataset(pipeline, f, "classification")
                else:
                    continue

        elif choice == "3":
            print("\nProcessing ALL datasets...")
            for f in reg_files + cls_files:
                process_dataset(pipeline, f, "auto")
        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()