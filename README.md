<p align="center">
  <h1 align="center">💳 Credit Card Fraud Detection</h1>
  <p align="center">
    ML pipeline for detecting fraudulent credit card transactions using supervised learning and anomaly detection on heavily imbalanced data.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-orange?logo=xgboost" alt="XGBoost">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-yellow?logo=scikit-learn" alt="sklearn">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
</p>

---

## Overview

Credit card fraud accounts for just **0.17%** of transactions (578:1 imbalance). A naive "predict all legitimate" model hits 99.83% accuracy while catching **zero** fraud — making accuracy useless. This pipeline uses **PR-AUC** as the primary metric and tunes thresholds based on real business costs.

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud.

---

## Project Structure

```
fraud_detection/
├── src/
│   ├── fraud_detector.py          # Core ML pipeline
│   └── autoencoder.py             # PyTorch autoencoder for anomaly detection
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory data analysis
│   ├── 02_Modeling.ipynb          # Model training and comparison
│   └── 03_Threshold_Analysis.ipynb
├── models/
│   ├── xgboost_fraud.pkl          # Trained model
│   ├── scaler.pkl                 # Fitted RobustScaler
│   ├── save_models.py             # Retrain and export models
│   └── model_card.md              # Model documentation
├── predict.py                     # Score new transactions via CLI
├── data/                          # Place creditcard.csv here
├── reports/                       # Auto-generated metrics and figures
└── requirements.txt
```

---

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Download dataset (requires Kaggle account)
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/

# Run the full pipeline
python src/fraud_detector.py

# Score transactions
python predict.py                          # demo mode
python predict.py --amount 250 --hour 3    # single transaction
python predict.py --csv data/sample.csv    # batch scoring
```

---

## Models & Results

### Supervised

| Model | PR-AUC | Recall | Precision | F1 |
|-------|--------|--------|-----------|-----|
| **XGBoost** | **0.850** | 0.867 | 0.697 | 0.773 |
| Random Forest | 0.787 | 0.857 | 0.583 | 0.694 |
| Logistic Regression | 0.722 | 0.878 | 0.541 | 0.669 |

### Unsupervised (Anomaly Detection)

| Model | PR-AUC | Recall | Notes |
|-------|--------|--------|-------|
| Isolation Forest | 0.143 | 0.755 | No labels needed — O(n log n) |
| One-Class SVM | 0.131 | 0.816 | O(n²), trained on 5K sample |

A **PyTorch Autoencoder** is also included as a standalone module ([autoencoder.py](src/autoencoder.py)) — it trains on legitimate transactions only and flags high reconstruction error as fraud.

> Anomaly detectors are trained on **legitimate transactions only** — high reconstruction error or short isolation path = fraud signal.

---

## Key Techniques

- **Resampling:** SMOTE / ADASYN on training data only (never before train/test split)
- **Threshold tuning:** Minimizes `FN × $500 + FP × $15` — optimal threshold ~0.30 instead of default 0.5
- **Feature engineering:** Log-transformed amounts, hour-of-day, night flag
- **Top feature:** V14 (30.5% importance) — PCA component correlated with fraud patterns
- **Scaling:** RobustScaler (handles outliers better than StandardScaler)

---

## Tech Stack

scikit-learn · XGBoost · imbalanced-learn (SMOTE) · PyTorch · pandas · NumPy · matplotlib · seaborn · Plotly · Jupyter

---

<p align="center">
  <sub>Dataset: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Kaggle Credit Card Fraud Detection</a> — ULB Machine Learning Group</sub>
</p>
