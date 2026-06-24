# Data Folder

## Real Dataset (Recommended)

Download `creditcard.csv` from Kaggle and place it here:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**The CSV is NOT committed to Git** (150MB — too large, and Kaggle requires account login).

### Quick download via Kaggle CLI
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

## Dataset Details

| Property | Value |
|----------|-------|
| Rows | 284,807 transactions |
| Fraud | 492 (0.172%) |
| Features | V1–V28 (PCA-transformed), Time, Amount |
| Target | Class (0 = legitimate, 1 = fraud) |
| Source | ULB Machine Learning Group |

V1–V28 are anonymised via PCA to protect cardholder privacy.
Time = seconds elapsed since first transaction.
Amount = transaction value in euros.

## Synthetic Data (No Download Needed)

The pipeline works out of the box using `generate_synthetic_fraud_data()`.
Synthetic data mimics the Kaggle structure: same features, same 0.17% fraud rate,
same statistical signatures in V1, V4, V11, V14, V17.

To switch to real data, edit `src/fraud_detector.py` line in `run()`:
```python
# Replace:
df = generate_synthetic_fraud_data()
# With:
df = pd.read_csv('data/creditcard.csv')
```
