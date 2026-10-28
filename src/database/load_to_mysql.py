
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"

load_dotenv(ROOT / ".env")


def main():
    print("=" * 55)
    print("CREDIT CARD FRAUD - MYSQL DATA LOADER")
    print("=" * 55)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise ValueError(
            f"Missing settings in .env: {', '.join(missing)}"
        )

    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Original rows: {original_rows:,}")
    print(f"Rows after exact deduplication: {len(df):,}")

    transactions = pd.DataFrame({
        "transaction_id": df.index + 1,
        "elapsed_seconds": df["Time"],
        "amount": df["Amount"],
        "fraud_label": df["Class"].astype(int),
    })

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
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            existing = connection.execute(
                text("SELECT COUNT(*) FROM transactions")
            ).scalar_one()

        if existing > 0:
            raise RuntimeError(
                f"transactions already contains {existing:,} rows. "
                "Stopping to avoid duplicate imports."
            )

        transactions.to_sql(
            "transactions",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=2000,
            method="multi",
        )

        with engine.connect() as connection:
            total = connection.execute(
                text("SELECT COUNT(*) FROM transactions")
            ).scalar_one()

            fraud = connection.execute(
                text(
                    "SELECT COUNT(*) FROM transactions "
                    "WHERE fraud_label = 1"
                )
            ).scalar_one()

            legitimate = connection.execute(
                text(
                    "SELECT COUNT(*) FROM transactions "
                    "WHERE fraud_label = 0"
                )
            ).scalar_one()

        print("\n--- Import Verification ---")
        print(f"Database rows: {total:,}")
        print(f"Legitimate transactions: {legitimate:,}")
        print(f"Fraud transactions: {fraud:,}")

        if total != len(transactions):
            raise RuntimeError("Imported row count does not match.")

        print("\nImport completed successfully!")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
