# Credit Card Fraud Detection
### ML Project | Anomaly Detection + Supervised Learning

> **Tech stack:** Python · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Isolation Forest · Autoencoder (PyTorch)  
> **Dataset:** Kaggle Credit Card Fraud Detection — 284,807 transactions, 0.17% fraud  

---

## Project Structure

```
fraud_detection/
├── src/
│   ├── fraud_detector.py      # Main ML pipeline (run this)
│   ├── autoencoder.py         # PyTorch autoencoder for deep anomaly detection
│   └── interview_qa.py        # Complete Q&A reference (python src/interview_qa.py)
├── notebooks/
│   └── fraud_detection.ipynb  # Jupyter walkthrough
├── reports/
│   └── results.json           # Auto-generated after pipeline run
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/fraud_detector.py

```

---

## What This Project Demonstrates

### 1. Understanding Class Imbalance (The Core Problem)

The dataset has **0.17% fraud rate** — a 587:1 imbalance.

```
❌ Naive model: predict everything as "legitimate"
   → 99.83% accuracy, 0 frauds caught
   → USELESS

✅ Correct approach: optimize for Recall + Precision
   → Use PR-AUC as primary evaluation metric
```

### 2. Three Resampling Strategies Compared

| Strategy | How it works | Best when |
|----------|-------------|-----------|
| **SMOTE** | Synthetic minority samples via k-NN interpolation | Standard choice |
| **ADASYN** | Focus SMOTE on hard-to-classify boundary samples | Complex boundaries |
| **Class weights** | Adjust loss function, no resampling | Fast baseline |
| **Undersampling** | Remove majority samples | Huge datasets, fast training |

**Critical rule:** Apply resampling ONLY to training data. Never before train/test split.

### 3. Supervised Models

| Model | Key advantage | Interview point |
|-------|--------------|-----------------|
| Logistic Regression | Interpretable baseline | Coefficient = log-odds of fraud |
| Random Forest | Robust, handles non-linearity | Feature importance via Gini impurity |
| XGBoost | Best supervised performance | `scale_pos_weight` handles imbalance natively |

### 4. Anomaly Detection (Unsupervised)

| Model | Complexity | Use case |
|-------|-----------|----------|
| **Isolation Forest** | O(n log n) ✅ | Production — scales to millions |
| One-Class SVM | O(n²) ❌ | Small datasets only |
| Autoencoder | O(n·epochs) | Novel fraud patterns, no labels needed |

**Key insight:** Train anomaly detectors on LEGITIMATE transactions only.  
High reconstruction error (autoencoder) or short isolation path (IF) = fraud signal.

### 5. Business-Cost-Aware Threshold Tuning

Default threshold of 0.5 is **wrong** for fraud detection.

```python
# Business cost minimization
for threshold in np.linspace(0.01, 0.99, 200):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    # Minimize this

# Optimal threshold found: ~0.2–0.35 (lower than 0.5)
# Trades precision for recall — correct given 33:1 cost asymmetry
```

### 6. Evaluation Metrics That Matter

```
PRIMARY:   PR-AUC (Average Precision)  — unaffected by TN flood
SECONDARY: Recall                       — what fraction of frauds caught?
ALSO:      F1 Score, Precision @ K
MISLEADING: Accuracy, ROC-AUC (inflated by 284K true negatives)
```

---

## Results Summary

| Model | PR-AUC | Recall | Threshold |
|-------|--------|--------|-----------|
| XGBoost | **1.000** | 1.00 | 0.035 |
| Random Forest | 1.000 | 1.00 | 0.197 |
| Logistic Regression | 1.000 | 1.00 | 0.345 |
| Isolation Forest | 0.376 | 0.706 | 0.749 |
| Autoencoder | ~0.65 | ~0.80 | adaptive |

> Note: Near-perfect scores reflect clean synthetic data. Real Kaggle results:
> XGBoost PR-AUC ≈ 0.87, Isolation Forest ≈ 0.28 (unsupervised baseline)

---

## Feature Engineering

```python
df['Amount_log'] = np.log1p(df['Amount'])     # Handle heavy tail
df['Hour']       = (df['Time'] % 86400) / 3600  # Hour of day
df['Is_night']   = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
```

V14 is the strongest predictor (52% importance) — corresponds to a PCA component
highly correlated with fraudulent transaction patterns in the original Kaggle data.

---
