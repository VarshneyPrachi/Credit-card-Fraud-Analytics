
from pathlib import Path
import pandas as pd


# Automatically locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def validate_dataset():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - DATA VALIDATION")
    print("=" * 60)

    # 1. Check whether the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print(f"\nDataset: {DATA_PATH}")
    print("Loading dataset...")

    # 2. Load dataset
    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset Overview ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(
        f"Memory usage: "
        f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
    )

    # 3. Display sample records
    print("\n--- First 5 Records ---")
    print(df.head().to_string(index=False))

    # 4. Validate expected columns
    expected_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    missing_columns = [
        col for col in expected_columns
        if col not in df.columns
    ]

    print("\n--- Schema Validation ---")
    print(f"Expected columns: {len(expected_columns)}")
    print(f"Missing expected columns: {missing_columns}")
    print(f"Has correct column count: {len(df.columns) == 31}")

    # 5. Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())

    # 6. Duplicate records
    duplicate_count = int(df.duplicated().sum())

    print("\n--- Duplicate Analysis ---")
    print(f"Duplicate rows: {duplicate_count:,}")

    # 7. Fraud distribution
    print("\n--- Fraud Distribution ---")
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' is missing.")

    print(df["Class"].value_counts(dropna=False).sort_index())

    total = len(df)
    fraud_count = int((df["Class"] == 1).sum())
    legitimate_count = int((df["Class"] == 0).sum())

    print(f"\nTotal transactions: {total:,}")
    print(f"Legitimate transactions: {legitimate_count:,}")
    print(f"Fraud transactions: {fraud_count:,}")

    if total:
        print(f"Fraud rate: {fraud_count / total * 100:.4f}%")

    # 8. Transaction amount analysis
    if "Amount" in df.columns:
        print("\n--- Transaction Amount Statistics ---")
        print(df["Amount"].describe().round(2).to_string())

    # 9. Basic data quality checks
    print("\n--- Data Quality Checks ---")
    print(f"Missing values present: {df.isnull().any().any()}")
    print(f"Duplicate rows present: {duplicate_count > 0}")
    print(
        "Target contains only 0 and 1: "
        f"{df['Class'].isin([0, 1]).all()}"
    )

    print("\nValidation completed successfully.")
    print("Original dataset has not been modified.")


if __name__ == "__main__":
    validate_dataset()
