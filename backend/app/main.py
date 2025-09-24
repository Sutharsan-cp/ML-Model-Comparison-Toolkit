import os
import sys
from pathlib import Path
import pandas as pd

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from api.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from api.train import TrainingInterface  # Import from api folder

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

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
        print(" MACHINE LEARNING SYSTEM")
        print("="*60)
        print("\nMain Menu:")
        print("----------------------------------------")
        print("1. Preprocess Datasets")
        print("2. Train Models")
        print("3. Test Imports")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            # Preprocessing menu
            reg_files, cls_files = list_datasets()
            
            print("\nPreprocessing Menu:")
            print("1. Process Regression Datasets")
            print("2. Process Classification Datasets")
            print("3. Process All Datasets")
            print("4. Back to Main Menu")
            
            sub_choice = input("\nEnter choice (1-4): ").strip()
            
            if sub_choice == "1":
                for f in reg_files:
                    process_dataset(pipeline, f, "regression")
            elif sub_choice == "2":
                for f in cls_files:
                    process_dataset(pipeline, f, "classification")
            elif sub_choice == "3":
                for f in reg_files + cls_files:
                    process_dataset(pipeline, f, "auto")
            elif sub_choice == "4":
                continue
            else:
                print("Invalid choice!")
                
        elif choice == "2":
            # Training menu
            training_interface = TrainingInterface()
            training_interface.run_training_menu()
            
        elif choice == "3":
            # Test imports
            print("\nTesting imports...")
            try:
                from api.test_import import main as test_imports
                test_imports()
            except ImportError as e:
                print(f"Error running import tests: {e}")
            
        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()