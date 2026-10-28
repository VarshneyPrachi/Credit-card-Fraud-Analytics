
from pathlib import Path
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PRED_PATH = ROOT / "data" / "processed" / "test_predictions.csv"

load_dotenv(ROOT / ".env")

BATCH_SIZE = 1000
THRESHOLD = 0.50


def main():
    print("=" * 65)
    print("CREDIT CARD FRAUD - LOAD MODEL PREDICTIONS")
    print("=" * 65)

    required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [key for key in required_env if not os.getenv(key)]

    if missing:
        raise ValueError(
            f"Missing database settings in .env: {', '.join(missing)}"
        )

    if os.getenv("DB_NAME") != "fraud_analytics":
        raise RuntimeError(
            "Safety check failed: DB_NAME must be fraud_analytics."
        )

    if not RAW_PATH.exists() or not PRED_PATH.exists():
        raise FileNotFoundError(
            "The raw dataset or test_predictions.csv could not be found."
        )

    # ------------------------------------------------------
    # 1. Load data and validate the deduplicated row mapping
    # ------------------------------------------------------
    print("\nReading dataset and model predictions...")

    raw = pd.read_csv(RAW_PATH)
    dedup = raw.drop_duplicates().reset_index(drop=True)
    pred = pd.read_csv(PRED_PATH)

    required_columns = [
        "row_index",
        "Time",
        "Amount",
        "actual_class",
        "logistic_regression_probability",
        "logistic_regression_prediction",
        "random_forest_probability",
        "random_forest_prediction",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in pred.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    if pred.empty:
        raise ValueError("Prediction file is empty.")

    if pred[required_columns].isnull().any().any():
        raise ValueError("Missing values found in prediction data.")

    row_index = pred["row_index"].to_numpy(dtype=float)

    if (
        not np.isfinite(row_index).all()
        or not np.equal(row_index, np.floor(row_index)).all()
    ):
        raise ValueError("row_index must contain valid integers.")

    pred["row_index"] = row_index.astype(np.int64)

    if pred["row_index"].duplicated().any():
        raise ValueError("Duplicate row_index values found.")

    if (
        (pred["row_index"] < 0)
        | (pred["row_index"] >= len(dedup))
    ).any():
        raise ValueError("Prediction row_index is out of range.")

    mapped = dedup.iloc[pred["row_index"].to_numpy()]

    checks = {
        "Class": np.array_equal(
            pred["actual_class"].to_numpy(dtype=int),
            mapped["Class"].to_numpy(dtype=int),
        ),
        "Time": np.array_equal(
            pred["Time"].to_numpy(dtype=float),
            mapped["Time"].to_numpy(dtype=float),
        ),
        "Amount": np.array_equal(
            pred["Amount"].to_numpy(dtype=float),
            mapped["Amount"].to_numpy(dtype=float),
        ),
    }

    print(f"Original dataset rows: {len(raw):,}")
    print(f"Deduplicated rows: {len(dedup):,}")
    print(f"Prediction rows: {len(pred):,}")

    for column, passed in checks.items():
        print(f"{column} mapping: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError(
            "Prediction mapping failed. Nothing has been inserted."
        )

    # transaction_id is one-based; row_index is zero-based.
    pred["transaction_id"] = pred["row_index"] + 1

    # ------------------------------------------------------
    # 2. Validate probabilities and predicted classes
    # ------------------------------------------------------
    probability_columns = [
        "logistic_regression_probability",
        "random_forest_probability",
    ]

    for column in probability_columns:
        values = pred[column].to_numpy(dtype=float)

        if (
            not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ValueError(
                f"Invalid probability values in {column}."
            )

    model_configs = [
        (
            "Logistic Regression",
            "logistic_regression_probability",
            "logistic_regression_prediction",
        ),
        (
            "Random Forest",
            "random_forest_probability",
            "random_forest_prediction",
        ),
    ]

    for model_name, probability_col, prediction_col in model_configs:
        probabilities = pred[probability_col].to_numpy(dtype=float)
        predictions = pred[prediction_col].to_numpy(dtype=int)

        if not np.isin(predictions, [0, 1]).all():
            raise ValueError(
                f"Invalid predicted classes for {model_name}."
            )

        expected = (probabilities >= THRESHOLD).astype(int)

        if not np.array_equal(predictions, expected):
            raise ValueError(
                f"Predictions for {model_name} do not match "
                f"the {THRESHOLD:.2f} threshold."
            )

    # ------------------------------------------------------
    # 3. Prepare the two-model prediction records
    # ------------------------------------------------------
    logistic = pd.DataFrame({
        "transaction_id": pred["transaction_id"],
        "model_name": "Logistic Regression",
        "fraud_probability": pred[
            "logistic_regression_probability"
        ].astype(float),
        "predicted_class": pred[
            "logistic_regression_prediction"
        ].astype(int),
        "decision_threshold": THRESHOLD,
    })

    forest = pd.DataFrame({
        "transaction_id": pred["transaction_id"],
        "model_name": "Random Forest",
        "fraud_probability": pred[
            "random_forest_probability"
        ].astype(float),
        "predicted_class": pred[
            "random_forest_prediction"
        ].astype(int),
        "decision_threshold": THRESHOLD,
    })

    output = pd.concat(
        [logistic, forest],
        ignore_index=True,
    )

    # ------------------------------------------------------
    # 4. Connect to the intended database
    # ------------------------------------------------------
    url = URL.create(
        "mysql+pymysql",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.environ["DB_NAME"],
    )

    engine = create_engine(url, pool_pre_ping=True)

    try:
        # --------------------------------------------------
        # 5. Verify the transaction mapping against MySQL
        # --------------------------------------------------
        with engine.connect() as conn:
            database = conn.execute(
                text("SELECT DATABASE()")
            ).scalar_one()

            if database != "fraud_analytics":
                raise RuntimeError(
                    f"Connected to unexpected database: {database}"
                )

            transaction_count = conn.execute(
                text("SELECT COUNT(*) FROM transactions")
            ).scalar_one()

            existing_predictions = conn.execute(
                text("SELECT COUNT(*) FROM transaction_predictions")
            ).scalar_one()

        if transaction_count != len(dedup):
            raise RuntimeError(
                f"Expected {len(dedup):,} transaction rows, "
                f"but MySQL contains {transaction_count:,}."
            )

        if existing_predictions != 0:
            raise RuntimeError(
                "transaction_predictions is not empty "
                f"({existing_predictions:,} rows). "
                "Stopping to prevent duplicate inserts."
            )

        print("\nValidating mapping against MySQL...")

        for start in range(0, len(pred), BATCH_SIZE):
            batch = pred.iloc[start:start + BATCH_SIZE]
            ids = batch["transaction_id"].tolist()

            placeholders = ", ".join(
                f":id{i}" for i in range(len(ids))
            )
            params = {
                f"id{i}": value for i, value in enumerate(ids)
            }

            query = text(
                "SELECT transaction_id, elapsed_seconds, amount, fraud_label "
                "FROM transactions "
                f"WHERE transaction_id IN ({placeholders})"
            )

            with engine.connect() as conn:
                db_rows = pd.read_sql_query(
                    query, conn, params=params
                )

            if len(db_rows) != len(batch):
                raise RuntimeError(
                    f"MySQL mapping check failed at batch {start}."
                )

            check = batch.merge(
                db_rows,
                on="transaction_id",
                how="left",
                validate="one_to_one",
            )

            time_ok = np.isclose(
                check["Time"].to_numpy(dtype=float),
                check["elapsed_seconds"].to_numpy(dtype=float),
                rtol=0,
                atol=1e-9,
            )

            amount_ok = np.isclose(
                check["Amount"].to_numpy(dtype=float),
                check["amount"].to_numpy(dtype=float),
                rtol=0,
                atol=0.011,
            )

            class_ok = (
                check["actual_class"].to_numpy(dtype=int)
                == check["fraud_label"].to_numpy(dtype=int)
            )

            if not (time_ok & amount_ok & class_ok).all():
                raise RuntimeError(
                    f"MySQL mapping mismatch in batch starting at {start}. "
                    "Nothing has been inserted."
                )

        print("MySQL mapping validation: PASS")
        print(f"Rows prepared for insertion: {len(output):,}")

        # --------------------------------------------------
        # 6. Insert both models' predictions atomically
        # --------------------------------------------------
        with engine.begin() as conn:
            # Recheck inside the insertion transaction.
            current_count = conn.execute(
                text("SELECT COUNT(*) FROM transaction_predictions")
            ).scalar_one()

            if current_count != 0:
                raise RuntimeError(
                    "Prediction table is no longer empty. Aborting."
                )

            output.to_sql(
                "transaction_predictions",
                con=conn,
                if_exists="append",
                index=False,
                chunksize=2000,
                method="multi",
            )

            final_count = conn.execute(
                text("SELECT COUNT(*) FROM transaction_predictions")
            ).scalar_one()

            if final_count != len(output):
                raise RuntimeError(
                    f"Expected {len(output):,} rows but found "
                    f"{final_count:,}. Rolling back."
                )

        print("\n" + "=" * 65)
        print("PREDICTIONS LOADED SUCCESSFULLY")
        print("=" * 65)
        print(f"Total prediction rows: {len(output):,}")
        print(f"Logistic Regression: {len(logistic):,}")
        print(f"Random Forest: {len(forest):,}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
