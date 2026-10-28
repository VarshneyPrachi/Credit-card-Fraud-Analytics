
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# Consistent colors for our fraud analytics project
NAVY = "#14213D"
TEAL = "#00A6A6"
RED = "#E63946"
GREY = "#8D99AE"


def save_figure(filename):
    """Save a high-resolution figure and close it."""
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {output_path.relative_to(PROJECT_ROOT)}")


def run_eda():
    print("=" * 65)
    print("CREDIT CARD FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Class"].isna().any() or not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only non-null 0 and 1 labels.")

    print(f"\nDataset shape: {df.shape}")
    print(f"Total transactions: {len(df):,}")

    # --------------------------------------------------
    # 1. TARGET DISTRIBUTION
    # --------------------------------------------------

    counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    fraud_rate = df["Class"].mean() * 100

    print("\n--- Target Distribution ---")
    print(f"Legitimate: {counts[0]:,}")
    print(f"Fraudulent: {counts[1]:,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        ["Legitimate", "Fraudulent"],
        [counts[0], counts[1]],
        color=[NAVY, RED],
        width=0.55
    )

    ax.set_title("Transaction Class Distribution", fontsize=15, weight="bold")
    ax.set_ylabel("Number of transactions")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({count / len(df) * 100:.3f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_ylim(0, counts.max() * 1.15)
    save_figure("01_class_distribution.png")

    # --------------------------------------------------
    # 2. TRANSACTION AMOUNT DISTRIBUTION
    # --------------------------------------------------

    print("\n--- Transaction Amount Analysis ---")
    print(df["Amount"].describe().round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="Amount",
        bins=80,
        color=TEAL,
        ax=ax
    )

    ax.set_title("Transaction Amount Distribution", fontsize=15, weight="bold")
    ax.set_xlabel("Transaction amount")
    ax.set_ylabel("Number of transactions")

    save_figure("02_amount_distribution.png")

    # --------------------------------------------------
    # 3. AMOUNTS BY CLASS
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    amount_data = df.copy()
    amount_data["Log Amount"] = np.log1p(amount_data["Amount"])
    amount_data["Transaction Type"] = amount_data["Class"].map(
        {0: "Legitimate", 1: "Fraudulent"}
    )

    sns.boxplot(
        data=amount_data,
        x="Transaction Type",
        y="Log Amount",
        hue="Transaction Type",
        order=["Legitimate", "Fraudulent"],
        palette={
            "Legitimate": NAVY,
            "Fraudulent": RED
        },
        legend=False,
        showfliers=False,
        ax=ax
    )

    ax.set_title(
        "Transaction Amount by Class (Log Scale)",
        fontsize=15,
        weight="bold"
    )
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + amount)")

    save_figure("03_amount_by_class.png")

    # --------------------------------------------------
    # 4. TRANSACTION ACTIVITY OVER ELAPSED TIME
    # --------------------------------------------------

    # Time is elapsed seconds, not a real calendar timestamp.
    time_data = df.copy()
    time_data["Elapsed Hour"] = (
        time_data["Time"] // 3600
    ).astype(int)

    hourly = (
        time_data.groupby(["Elapsed Hour", "Class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        hourly.index,
        hourly[0],
        color=NAVY,
        label="Legitimate",
        linewidth=2
    )
    ax.plot(
        hourly.index,
        hourly[1],
        color=RED,
        label="Fraudulent",
        linewidth=2,
        marker="o",
        markersize=3
    )

    ax.set_title(
        "Transaction Activity by Elapsed Hour",
        fontsize=15,
        weight="bold"
    )
    ax.set_xlabel("Elapsed hour since dataset start")
    ax.set_ylabel("Transaction count")
    ax.legend()

    save_figure("04_transactions_over_time.png")

    # --------------------------------------------------
    # 5. CORRELATION HEATMAP
    # --------------------------------------------------

    # Class correlations can help exploration, but are not
    # proof of causal relationships or model importance.
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.1,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.75},
        ax=ax
    )

    ax.set_title(
        "Transaction Feature Correlation Heatmap",
        fontsize=17,
        weight="bold",
        pad=15
    )

    save_figure("05_correlation_heatmap.png")

    # --------------------------------------------------
    # 6. DUPLICATE INVESTIGATION
    # --------------------------------------------------

    duplicate_mask = df.duplicated()
    duplicate_rows = int(duplicate_mask.sum())

    # Counts only the additional occurrences beyond the first
    # record in each exact duplicate group.
    duplicate_label_counts = (
        df.loc[duplicate_mask, "Class"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    # All records belonging to duplicate groups, including
    # the first occurrence in each group.
    repeated_group_records = int(
        df.duplicated(keep=False).sum()
    )

    print("\n--- Duplicate Investigation ---")
    print(f"Extra duplicate occurrences: {duplicate_rows:,}")
    print(
        "Extra duplicate occurrences labeled legitimate: "
        f"{duplicate_label_counts[0]:,}"
    )
    print(
        "Extra duplicate occurrences labeled fraud: "
        f"{duplicate_label_counts[1]:,}"
    )
    print(
        "Records belonging to duplicate groups: "
        f"{repeated_group_records:,}"
    )

    # --------------------------------------------------
    # 7. CLASS-WISE STATISTICS
    # --------------------------------------------------

    class_stats = (
        df.groupby("Class")["Amount"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
    )

    class_stats.index = ["Legitimate", "Fraudulent"]
    class_stats.to_csv(METRICS_DIR / "amount_statistics_by_class.csv")

    print("\n--- Amount Statistics by Class ---")
    print(class_stats.to_string())

    # --------------------------------------------------
    # 8. SAVE REPRODUCIBLE SUMMARY
    # --------------------------------------------------

    summary = {
        "total_transactions": int(len(df)),
        "total_columns": int(df.shape[1]),
        "legitimate_transactions": int(counts[0]),
        "fraudulent_transactions": int(counts[1]),
        "fraud_rate_percent": round(float(fraud_rate), 6),
        "missing_values_total": int(df.isna().sum().sum()),
        "extra_duplicate_occurrences": duplicate_rows,
        "records_in_duplicate_groups": repeated_group_records,
        "amount_mean": round(float(df["Amount"].mean()), 4),
        "amount_median": round(float(df["Amount"].median()), 4),
        "amount_max": round(float(df["Amount"].max()), 4),
        "time_interpretation": "Elapsed seconds, not calendar timestamps"
    }

    summary_path = METRICS_DIR / "eda_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"\nSummary saved: {summary_path.relative_to(PROJECT_ROOT)}")
    print("\nEDA completed successfully.")
    print("The original CSV was not modified.")


if __name__ == "__main__":
    run_eda()
