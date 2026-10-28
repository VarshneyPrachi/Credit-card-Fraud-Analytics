# 🛡️ FraudGuard — Credit Card Fraud Analytics

<p align="center">
  <strong>Detect suspicious transactions. Understand fraud patterns. Make data-driven decisions.</strong>
</p>

<p align="center">
  <img src="tableau/screenshots/fraud-overview.png" alt="FraudGuard Analytics Dashboard" width="900">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="MySQL">
  <img src="https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white" alt="Tableau">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
</p>

## 📌 Project Overview

FraudGuard is an end-to-end credit card fraud analytics project combining data analysis, machine learning, SQL, business intelligence, and a web-based dashboard. It analyzes highly imbalanced transaction data, compares fraud detection models, and presents key insights through interactive analytics.

The project is designed to demonstrate how raw transaction data can be transformed into actionable insights for fraud monitoring and investigation.

## 📊 Dataset at a Glance

| Metric | Project result |
|---|---:|
| Original dataset records | 284,807 |
| Deduplicated transactions | 283,726 |
| Confirmed fraud transactions | 473 |
| Legitimate transactions | 283,253 |
| Fraud rate after deduplication | Approximately 0.167% |

**Important:** Fraud detection is a highly imbalanced classification problem. Accuracy alone is not sufficient to judge whether a model is effective.

## 🧠 Machine Learning Results

Recorded held-out test-set metrics from the project:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 97.37% | 99.96% |
| Precision | 5.47% | 93.98% |
| Recall | 91.49% | 82.98% |
| F1-score | 10.33% | 88.14% |
| ROC-AUC | 98.04% | 97.08% |
| Average precision | 77.96% | 88.54% |

**Key finding:** Random Forest achieved substantially higher precision and F1-score in the recorded evaluation. Logistic Regression achieved higher recall, but its low precision indicates many more false-positive alerts. The preferred model depends on the business cost of missed fraud versus unnecessary investigations.

## ✨ Project Features

- **Data preparation:** Audit duplicate records and prepare transaction data for analysis.
- **Exploratory data analysis:** Examine class imbalance, transaction amounts, and fraud patterns.
- **Machine learning:** Train and compare Logistic Regression and Random Forest.
- **Model evaluation:** Assess precision, recall, F1-score, ROC-AUC, and average precision.
- **MySQL analytics:** Store transaction and prediction records for querying.
- **Tableau reporting:** Present fraud distributions, trends, and model performance.
- **FastAPI backend:** Expose dashboard data and prediction endpoints.
- **React dashboard:** Provide a web interface for fraud analytics and risk checking.

## 🏗️ Architecture

```text
Credit Card Transaction Dataset
              |
              v
    Python Data Preparation
              |
              v
 Exploratory Data Analysis
              |
              v
     Machine Learning
   Logistic Regression
      Random Forest
              |
        +-----+------+
        |            |
        v            v
    MySQL DB     Model Outputs
        |            |
        +-----+------+
              |
        +-----+------+
        |            |
        v            v
      Tableau     FastAPI
                   |
                   v
               React UI
```

## 🖼️ Tableau Dashboard

### Fraud Overview
![Fraud Overview](tableau/screenshots/fraud-overview.png)

*Overview of transaction volume, fraud counts, fraud rate, and key performance indicators.*

### Transaction Analysis
![Transaction Analysis](tableau/screenshots/transaction-analysis.png)

*Transaction amount distribution and fraud-related patterns.*

### Model Performance
![Model Performance](tableau/screenshots/model-performance.png)

*Comparison of classification metrics across the trained models.*

> Replace these example image paths with your actual exported Tableau screenshots. Remove any section for a dashboard that has not yet been created.

## 💻 Code Highlights

### 1. Load and inspect transaction data

```python
import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.isnull().sum().sum())
```

### 2. Audit duplicate records

```python
duplicate_count = df.duplicated().sum()

print("Total records:", len(df))
print("Exact duplicate occurrences:", duplicate_count)

clean_df = df.drop_duplicates().copy()

print("Records after deduplication:", len(clean_df))
```

### 3. Train a Random Forest classifier

The following is a representative example. Use the same preprocessing, split strategy, and feature ordering as the model actually saved in this project.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
```

### 4. Query fraud counts with SQL

```sql
SELECT
    fraud_label,
    COUNT(*) AS transaction_count,
    ROUND(
        100.0 * COUNT(*) /
        (SELECT COUNT(*) FROM transactions),
        3
    ) AS percentage
FROM transactions
GROUP BY fraud_label
ORDER BY fraud_label;
```

### 5. Run the FastAPI backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` to explore the available API endpoints.

### 6. Run the React frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite.

> The snippets above illustrate the main workflow. Check them against your actual source files before treating them as exact reproductions of the implementation.

## 🚀 Getting Started

### Prerequisites

- Python and pip
- Node.js and npm
- MySQL Server
- Tableau Desktop or Tableau Public, if you want to explore the dashboards

### Clone the repository

```bash
git clone https://github.com/VarshneyPrachi/Credit-card-Fraud-Analytics.git
cd Credit-card-Fraud-Analytics
```

Follow the instructions in the backend and frontend folders to configure dependencies and environment variables. Configure database credentials locally; never commit passwords or API secrets.

## 📁 Repository Guide

| Folder | Purpose |
|---|---|
| `data/` | Dataset documentation and data files, where permitted |
| `python/` | Data preparation and analysis scripts |
| `ml/` | Model training, evaluation, and outputs |
| `models/` | Saved model artifacts |
| `sql/` | Database schema and analytical queries |
| `tableau/` | Tableau workbooks and exported screenshots |
| `backend/` | FastAPI application |
| `frontend/` | React dashboard |
| `docs/` | Architecture and technical documentation |

## 🔒 Data and Responsible Use

The project uses anonymized transaction features from a public fraud detection dataset. Review the dataset's license and redistribution terms before publishing data or derived files. Do not upload private credentials, local environment files, or sensitive transaction information.

Model outputs are analytical estimates and should not be treated as definitive evidence of fraud without appropriate investigation.

## 🔮 Future Improvements

- Connect every dashboard visualization to live database queries.
- Add real transaction-level model inference using the complete feature set.
- Improve model calibration and threshold selection.
- Add explainability with feature importance or SHAP.
- Add automated tests and reproducible model evaluation.
- Deploy the application with appropriate security controls.

## 👩‍💻 Author

**Prachi Varshney**

[GitHub Repository](https://github.com/VarshneyPrachi/Credit-card-Fraud-Analytics)

---

*Built as a portfolio project exploring fraud detection, analytics engineering, machine learning, and business intelligence.*
