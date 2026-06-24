"""
predict.py — Score new transactions using the saved model
=========================================================
Usage:
    python predict.py                         # scores a demo transaction
    python predict.py --amount 250 --hour 3   # custom transaction
    python predict.py --csv data/sample.csv   # score a batch CSV
"""
import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

THRESHOLD = 0.30          # Business-cost-optimised threshold (tune in model_card.md)
MODELS_DIR = Path(__file__).parent / 'models'

def load_model():
    model_path  = MODELS_DIR / 'xgboost_fraud.pkl'
    scaler_path = MODELS_DIR / 'scaler.pkl'

    if not model_path.exists():
        print("Model not found. Run: python models/save_models.py")
        sys.exit(1)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must match the feature engineering in fraud_detector.py"""
    df = df.copy()
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])
    if 'Time' in df.columns:
        df['Hour']     = (df['Time'] % 86400) / 3600
        df['Is_night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    # Drop raw columns used only for engineering
    drop_cols = ['Class', 'Time', 'Amount']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def score_transaction(model, scaler, transaction: dict) -> dict:
    """Score a single transaction dict and return result."""
    df = pd.DataFrame([transaction])
    df = engineer_features(df)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    prob = model.predict_proba(df_scaled)[0, 1]
    label = 'FRAUD' if prob >= THRESHOLD else 'LEGITIMATE'
    return {
        'fraud_probability': round(float(prob), 4),
        'label': label,
        'threshold': THRESHOLD,
        'risk_level': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW',
    }


def score_csv(model, scaler, csv_path: str):
    """Score a batch of transactions from a CSV file."""
    df = pd.read_csv(csv_path)
    has_labels = 'Class' in df.columns
    true_labels = df['Class'].values if has_labels else None

    df_feat = engineer_features(df)
    df_scaled = pd.DataFrame(scaler.transform(df_feat), columns=df_feat.columns)
    probs = model.predict_proba(df_scaled)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    results = pd.DataFrame({
        'fraud_probability': np.round(probs, 4),
        'prediction': ['FRAUD' if p else 'LEGITIMATE' for p in preds],
    })

    if has_labels:
        from sklearn.metrics import classification_report, average_precision_score
        print("\nEvaluation against ground truth:")
        print(classification_report(true_labels, preds, target_names=['Legitimate', 'Fraud']))
        print(f"PR-AUC: {average_precision_score(true_labels, probs):.4f}")

    return results


def demo_transaction():
    """Simulate one legitimate and one fraudulent transaction."""
    legit = {f'V{i}': np.random.randn() for i in range(1, 29)}
    legit.update({'Time': 50000, 'Amount': 45.00})

    fraud = {f'V{i}': np.random.randn() for i in range(1, 29)}
    fraud.update({'V14': -7.5, 'V17': -5.2, 'V4': 5.1,  # Strong fraud signals
                  'Time': 3600, 'Amount': 1.00})            # Small probe amount

    return legit, fraud


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Scorer')
    parser.add_argument('--amount', type=float, help='Transaction amount')
    parser.add_argument('--hour',   type=float, help='Hour of day (0–23)')
    parser.add_argument('--csv',    type=str,   help='Path to CSV file for batch scoring')
    args = parser.parse_args()

    model, scaler = load_model()
    print(f"Model loaded. Threshold: {THRESHOLD}")

    if args.csv:
        results = score_csv(model, scaler, args.csv)
        print(f"\nScored {len(results)} transactions:")
        print(f"  Flagged as fraud: {(results['prediction']=='FRAUD').sum()}")
        print(results.head(10).to_string())

    elif args.amount or args.hour:
        txn = {f'V{i}': np.random.randn() for i in range(1, 29)}
        txn['Amount'] = args.amount or 50.0
        txn['Time']   = (args.hour or 12) * 3600
        result = score_transaction(model, scaler, txn)
        print(f"\nTransaction score:")
        for k, v in result.items():
            print(f"  {k:<25} {v}")

    else:
        # Demo mode
        print("\n--- DEMO MODE ---")
        legit_txn, fraud_txn = demo_transaction()

        print("\nLegitimate-looking transaction:")
        r = score_transaction(model, scaler, legit_txn)
        for k, v in r.items(): print(f"  {k:<25} {v}")

        print("\nSuspicious transaction (small amount, late night, fraud features):")
        r = score_transaction(model, scaler, fraud_txn)
        for k, v in r.items(): print(f"  {k:<25} {v}")


if __name__ == '__main__':
    main()
