"""
save_models.py — Train pipeline and save models to disk
Run this once to generate models/xgboost_fraud.pkl and models/scaler.pkl
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib
from pathlib import Path
from fraud_detector import FraudDetectionPipeline, generate_synthetic_fraud_data

def save_models():
    models_dir = Path(__file__).parent

    pipeline = FraudDetectionPipeline(output_dir='../reports')
    df = generate_synthetic_fraud_data()
    pipeline.load_and_explore(df)
    pipeline.preprocess()
    X_res, y_res = pipeline.apply_resampling()
    pipeline.train_supervised_models(X_res, y_res)

    # Save best supervised model (XGBoost or fallback)
    model_name = 'XGBoost' if 'XGBoost' in pipeline.models else 'Random Forest'
    model = pipeline.models[model_name]
    scaler = pipeline.scaler

    joblib.dump(model,  models_dir / 'xgboost_fraud.pkl')
    joblib.dump(scaler, models_dir / 'scaler.pkl')

    print(f"Saved {model_name} → models/xgboost_fraud.pkl")
    print(f"Saved scaler      → models/scaler.pkl")
    print(f"\nThreshold used: {pipeline.results[model_name]['threshold']:.3f}")
    print("Remember to update THRESHOLD in predict.py if needed.")

if __name__ == '__main__':
    save_models()
