# Model Card — Credit Card Fraud Detector

## Model Overview

| Property | Value |
|----------|-------|
| Model type | XGBoost classifier (primary) |
| Task | Binary classification — fraud vs legitimate |
| Version | 1.0 |
| Training date | See reports/results.json |
| Primary metric | PR-AUC (Average Precision) |

---

## Training Data

- **Source:** Kaggle Credit Card Fraud Detection (ULB Machine Learning Group)
- **Size:** 284,807 transactions (492 fraud, 284,315 legitimate)
- **Time period:** 2 days of European cardholder transactions (September 2013)
- **Features:** V1–V28 (PCA-anonymised), Time, Amount + engineered features
- **Preprocessing:** RobustScaler (handles Amount outliers), SMOTE on training set only

---

## Performance (Test Set)

| Metric | Value | Notes |
|--------|-------|-------|
| PR-AUC | ~0.87 | Primary metric — unaffected by class imbalance |
| ROC-AUC | ~0.98 | Inflated by TN count — not primary metric |
| Recall | ~0.83 | 83% of frauds caught |
| Precision | ~0.77 | 77% of flagged transactions are genuine fraud |
| F1 Score | ~0.80 | |

> Results vary slightly between runs due to random seeds.
> Scores on synthetic data will appear higher (cleaner separability).

---

## Decision Threshold

**Chosen threshold: ~0.30** (not the default 0.5)

Tuned by minimising business cost:
```
Total cost = FN × $500 + FP × $15
```

A lower threshold increases recall at the cost of precision.
This is correct given the 33:1 FN/FP cost asymmetry.

To change the threshold, edit `predict.py`:
```python
THRESHOLD = 0.30   # Adjust based on your business context
```

---

## Intended Use

- **Designed for:** Real-time transaction screening at point of purchase
- **Output:** Fraud probability score [0, 1] + binary flag above threshold
- **Recommended deployment:** Asynchronous scoring with human review queue for flagged transactions

---

## Limitations

1. **Novel fraud patterns:** The model was trained on 2013 data. Fraud tactics evolve; periodic retraining is required (recommended: monthly).

2. **Small probe transactions:** Fraudsters often test cards with small amounts before large purchases. These can be harder to catch.

3. **Geographic bias:** Original data is European cardholders only. Performance on other regions may degrade.

4. **PCA anonymisation:** V1–V28 are anonymised — feature engineering is limited. A production system with raw features (merchant category, IP geolocation, device fingerprint) would perform better.

5. **Class distribution shift:** If fraud rate changes significantly from 0.17%, recalibrate the threshold.

---

## Ethical Considerations

- False positives block legitimate transactions — disproportionate impact on customers with atypical spending patterns
- Model should be one signal in a system, not the sole arbiter
- Decisions should be reviewable and reversible (customer dispute process)
- Avoid using demographic features as proxies for fraud risk

---

## Files

```
models/
├── xgboost_fraud.pkl    # Trained model (joblib)
├── scaler.pkl           # Fitted RobustScaler (must use with model)
└── model_card.md        # This file
```

**Important:** Always load `scaler.pkl` alongside `xgboost_fraud.pkl`.
The scaler must transform features before passing to the model.
See `predict.py` for correct loading pattern.
