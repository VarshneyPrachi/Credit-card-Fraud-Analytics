
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def audit_duplicates():
    print("=" * 65)
    print("CREDIT CARD FRAUD DETECTION - DUPLICATE AUDIT")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("The Class column is missing.")

    feature_columns = [
        column for column in df.columns if column != "Class"
    ]

    # Exact duplicate rows, including the target label.
    exact_duplicate_mask = df.duplicated(keep="first")
    exact_duplicate_count = int(exact_duplicate_mask.sum())

    # Extra duplicate occurrences, broken down by target class.
    duplicate_class_counts = (
        df.loc[exact_duplicate_mask, "Class"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    # Find rows whose predictor values match another row,
    # even if their Class labels differ.
    feature_duplicate_mask = df.duplicated(
        subset=feature_columns,
        keep=False
    )

    feature_duplicate_rows = df.loc[feature_duplicate_mask]

    if feature_duplicate_rows.empty:
        conflicting_groups = pd.DataFrame()
    else:
        # A group is conflicting when identical predictors
        # are associated with more than one target label.
        group_sizes = (
            feature_duplicate_rows
            .groupby(feature_columns, dropna=False)["Class"]
            .agg(["size", "nunique"])
            .reset_index()
        )

        conflicting_groups = group_sizes.loc[
            group_sizes["nunique"] > 1
        ].copy()

    print(f"\nTotal records: {len(df):,}")
    print(f"Exact duplicate occurrences: {exact_duplicate_count:,}")

    print("\n--- Exact duplicates by class ---")
    print(f"Class 0: {int(duplicate_class_counts[0]):,}")
    print(f"Class 1: {int(duplicate_class_counts[1]):,}")

    print("\n--- Predictor-level duplicate analysis ---")
    print(
        "Rows with repeated predictor values: "
        f"{len(feature_duplicate_rows):,}"
    )
    print(
        "Groups with identical predictors but conflicting labels: "
        f"{len(conflicting_groups):,}"
    )

    # Save any groups where the same predictors have
    # contradictory labels, for manual investigation.
    conflict_path = OUTPUT_DIR / "duplicate_label_conflicts.csv"

    if not conflicting_groups.empty:
        conflicting_groups.to_csv(conflict_path, index=False)
        print(f"Conflict details saved to: {conflict_path}")
    else:
        # Do not leave a stale report from an earlier run.
        if conflict_path.exists():
            conflict_path.unlink()
        print("No conflicting-label groups found.")

    report = {
        "total_records": int(len(df)),
        "exact_duplicate_occurrences": exact_duplicate_count,
        "exact_duplicate_class_0": int(duplicate_class_counts[0]),
        "exact_duplicate_class_1": int(duplicate_class_counts[1]),
        "rows_with_repeated_predictors": int(
            len(feature_duplicate_rows)
        ),
        "conflicting_predictor_groups": int(
            len(conflicting_groups)
        ),
        "raw_dataset_modified": False
    }

    report_path = OUTPUT_DIR / "duplicate_audit.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nAudit report saved: {report_path}")
    print("\nDuplicate audit completed.")
    print("The original CSV has not been modified.")


if __name__ == "__main__":
    audit_duplicates()
