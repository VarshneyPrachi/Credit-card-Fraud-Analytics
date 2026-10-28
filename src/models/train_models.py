
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"

for folder in [MODEL_DIR, METRICS_DIR, FIGURES_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def evaluate_model(name, model, X_test, y_test):
    """Calculate evaluation metrics using the default 0.5 threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "threshold": 0.5,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    print(f"\n--- {name} ---")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))

    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1-score:", round(metrics["f1_score"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print(
        "PR-AUC (average precision):",
        round(metrics["pr_auc_average_precision"], 4),
    )

    return metrics, probabilities, predictions


def main():
    print("=" * 65)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty or "Class" not in df.columns:
        raise ValueError("Dataset is empty or Class is missing.")

    if df.isnull().any().any():
        raise ValueError("Missing values found. Review data first.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    # Exact duplicates include the target label.
    exact_duplicates = int(df.duplicated().sum())
    print(f"\nInput rows: {len(df):,}")
    print(f"Exact duplicate occurrences: {exact_duplicates:,}")

    # Remove only exact duplicate records from the modeling copy.
    # The original CSV remains unchanged.
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Rows after exact deduplication: {len(df):,}")

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    # Group rows with identical predictor values so they cannot
    # appear in both train and test sets, even if labels differ.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    if train_groups.intersection(test_groups):
        raise RuntimeError("Feature-group leakage detected.")

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError("Training or test split lacks a target class.")

    print("\n--- Split Summary ---")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Testing fraud cases: {int(y_test.sum()):,}")
    print("Identical predictor groups overlap: No")

    # --------------------------------------------------
    # MODEL 1: LOGISTIC REGRESSION
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    logistic.fit(X_train, y_train)

    # --------------------------------------------------
    # MODEL 2: RANDOM FOREST
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_forest.fit(X_train, y_train)

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    results = []
    prediction_data = pd.DataFrame({
        "row_index": test_idx,
        "Time": X_test["Time"].to_numpy(),
        "Amount": X_test["Amount"].to_numpy(),
        "actual_class": y_test.to_numpy(),
    })

    models = {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
    }

    for name, model in models.items():
        metrics, probabilities, predictions = evaluate_model(
            name, model, X_test, y_test
        )

        results.append(metrics)

        safe_name = name.lower().replace(" ", "_")

        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib",
        )

        prediction_data[f"{safe_name}_probability"] = probabilities
        prediction_data[f"{safe_name}_prediction"] = predictions

        cm = np.array(metrics["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
        ax.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title(f"{name} — Confusion Matrix")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="black",
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"{safe_name}_confusion_matrix.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df.drop(columns=["confusion_matrix"]).to_csv(
        METRICS_DIR / "model_comparison.csv",
        index=False,
    )

    with open(
        METRICS_DIR / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    prediction_data.to_csv(
        PROCESSED_DIR / "test_predictions.csv",
        index=False,
    )

    # Random Forest feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": random_forest.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        METRICS_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    top_features = importance.head(15).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title("Random Forest — Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "random_forest_feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n--- Model Comparison ---")
    print(
        results_df[
            [
                "model",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc_average_precision",
            ]
        ].to_string(index=False)
    )

    print("\nSaved trained models to:", MODEL_DIR)
    print("Saved metrics to:", METRICS_DIR)
    print("Saved test predictions to:", PROCESSED_DIR)
    print("\nModel training completed successfully.")


if __name__ == "__main__":
    main()
