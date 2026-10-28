"""FraudGuard API starter. Uses read-only MySQL queries and never fabricates live data."""

import os
from pathlib import Path
from typing import Literal

import joblib
import mysql.connector
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
ROOT_DIR = PROJECT_DIR

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(ROOT_DIR / ".env", override=False)

app = FastAPI(
    title="FraudGuard API",
    description="Credit card fraud analytics and ML inference API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def get_connection():
    """Create a MySQL connection using the existing .env settings."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME", "fraud_analytics"),
        connection_timeout=5,
    )


# ---------------------------------------------------------
# Saved models
# ---------------------------------------------------------

MODEL_DIR = ROOT_DIR / "models"

MODEL_FILES = {
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
}

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

_models = {}
_model_errors = {}


def load_models():
    """Load available saved model files and verify their feature schema."""
    for name, path in MODEL_FILES.items():
        try:
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            model = joblib.load(path)

            feature_count = getattr(model, "n_features_in_", None)
            if feature_count != len(FEATURES):
                raise ValueError(
                    f"Expected {len(FEATURES)} input features, "
                    f"but model expects {feature_count}."
                )

            model_features = getattr(model, "feature_names_in_", None)
            if model_features is not None:
                if list(model_features) != FEATURES:
                    raise ValueError(
                        "Saved model feature names do not match "
                        "the expected Time, V1-V28, Amount schema."
                    )

            _models[name] = model
            _model_errors.pop(name, None)

        except Exception as exc:
            _models.pop(name, None)
            _model_errors[name] = str(exc)


load_models()


# ---------------------------------------------------------
# Request validation
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    model_name: Literal["Random Forest", "Logistic Regression"] = (
        "Random Forest"
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    Time: float = Field(ge=0, le=1e12)
    Amount: float = Field(ge=0, le=1e12)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# ---------------------------------------------------------
# Health and model status
# ---------------------------------------------------------

@app.get("/api/health")
def health():
    database_status = "connected"

    try:
        conn = get_connection()
        try:
            conn.ping(reconnect=False, attempts=1, delay=0)
        finally:
            conn.close()
    except Exception:
        database_status = "unavailable"

    return {
        "status": "ok",
        "service": "FraudGuard API",
        "database": database_status,
        "models": {
            name: name in _models
            for name in MODEL_FILES
        },
    }


@app.get("/api/models/performance")
def model_performance():
    """Return the recorded evaluation metrics for both trained models."""
    return {
        "source": "recorded held-out test metrics",
        "models": [
            {
                "name": "Logistic Regression",
                "accuracy": 0.9736893118,
                "precision": 0.0547422024,
                "recall": 0.9148936170,
                "f1": 0.1033033033,
                "roc_auc": 0.9804113240,
                "average_precision": 0.77959,
            },
            {
                "name": "Random Forest",
                "accuracy": 0.9996299233,
                "precision": 0.9397590361,
                "recall": 0.8297872340,
                "f1": 0.8813559322,
                "roc_auc": 0.9708101714,
                "average_precision": 0.88537,
            },
        ],
        "loaded_models": list(_models.keys()),
        "model_errors": _model_errors,
    }


# ---------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------

@app.get("/api/dashboard/summary")
def dashboard_summary():
    conn = None

    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT
                COUNT(*) AS total_transactions,
                COALESCE(SUM(fraud_label), 0) AS confirmed_fraud
            FROM transactions
        """)

        row = cur.fetchone() or {}
        total = int(row.get("total_transactions") or 0)
        fraud = int(row.get("confirmed_fraud") or 0)

        return {
            "total_transactions": total,
            "confirmed_fraud": fraud,
            "fraud_rate_percent": round(
                fraud * 100 / total, 3
            ) if total else 0,
            "source": "MySQL",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to query dashboard summary: {exc}",
        ) from exc

    finally:
        if conn is not None and conn.is_connected():
            conn.close()


# ---------------------------------------------------------
# Fraud timeline
# ---------------------------------------------------------

@app.get("/api/analytics/timeline")
def fraud_timeline():
    conn = None

    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT
                FLOOR(elapsed_seconds / 3600) AS hour_bucket,
                COUNT(*) AS total_transactions,
                COALESCE(SUM(fraud_label), 0) AS fraud_count,
                SUM(
                    CASE WHEN fraud_label = 0 THEN 1 ELSE 0 END
                ) AS legitimate_count
            FROM transactions
            GROUP BY FLOOR(elapsed_seconds / 3600)
            ORDER BY hour_bucket
        """)

        return {
            "data": cur.fetchall(),
            "source": "MySQL transactions",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to query timeline: {exc}",
        ) from exc

    finally:
        if conn is not None and conn.is_connected():
            conn.close()


# ---------------------------------------------------------
# Transaction amount distribution
# ---------------------------------------------------------

@app.get("/api/analytics/amount-buckets")
def fraud_amount_buckets():
    conn = None

    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT
                CASE
                    WHEN amount < 25 THEN '0-25'
                    WHEN amount < 100 THEN '25-100'
                    WHEN amount < 250 THEN '100-250'
                    ELSE '250+'
                END AS amount_bucket,
                COUNT(*) AS total_transactions,
                COALESCE(SUM(fraud_label), 0) AS fraud_count,
                MIN(amount) AS minimum_amount
            FROM transactions
            GROUP BY amount_bucket
            ORDER BY minimum_amount
        """)

        rows = cur.fetchall()

        for row in rows:
            row.pop("minimum_amount", None)

        return {
            "data": rows,
            "source": "MySQL transactions",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to query amount buckets: {exc}",
        ) from exc

    finally:
        if conn is not None and conn.is_connected():
            conn.close()


# ---------------------------------------------------------
# Recent transactions
# ---------------------------------------------------------

@app.get("/api/transactions/recent")
def recent_transactions():
    conn = None

    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT
                transaction_id,
                elapsed_seconds,
                amount,
                fraud_label
            FROM transactions
            ORDER BY transaction_id DESC
            LIMIT 10
        """)

        return {
            "data": cur.fetchall(),
            "source": "MySQL transactions",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to query recent transactions: {exc}",
        ) from exc

    finally:
        if conn is not None and conn.is_connected():
            conn.close()


# ---------------------------------------------------------
# Genuine ML prediction
# ---------------------------------------------------------

@app.post("/api/predict")
def predict_transaction(request: PredictionRequest):
    model_name = request.model_name
    model = _models.get(model_name)

    if model is None:
        detail = _model_errors.get(
            model_name,
            "The requested model is not loaded.",
        )
        raise HTTPException(status_code=503, detail=detail)

    values = request.model_dump()
    values.pop("model_name")
    threshold = values.pop("threshold")

    # Explicit ordering is essential for correct inference.
    row = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    try:
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                "The loaded model has no positive fraud class (1)."
            )

        fraud_index = classes.index(1)
        probability = float(probabilities[fraud_index])
        predicted_class = int(probability >= threshold)

        return {
            "model_name": model_name,
            "fraud_probability": probability,
            "predicted_class": predicted_class,
            "threshold": threshold,
            "risk_level": (
                "High" if probability >= 0.75
                else "Medium" if probability >= 0.25
                else "Low"
            ),
            "source": "saved ML model",
            "features_used": len(FEATURES),
            "feature_order": FEATURES,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        ) from exc
