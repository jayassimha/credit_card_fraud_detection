"""
Credit Card Fraud Detection - Core ML Pipeline
==============================================
Demonstrates: Class imbalance handling, threshold tuning,
anomaly detection, and business-cost-aware evaluation.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Resampling
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Note: imbalanced-learn not installed. SMOTE disabled.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Note: XGBoost not installed. Using GradientBoosting as fallback.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import json
import time


# ─── Business Cost Parameters ───────────────────────────────────────────────
# These are the KEY interview talking points
COST_FALSE_NEGATIVE = 500   # Missing a fraud: avg transaction loss ($)
COST_FALSE_POSITIVE = 15    # Blocking legit transaction: customer friction cost ($)


def generate_synthetic_fraud_data(n_samples=50000, fraud_ratio=0.0017, random_state=42):
    """
    Generates synthetic data mimicking the Kaggle Credit Card Fraud dataset structure.
    Real dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    Features V1-V28: PCA-transformed (anonymized), Time, Amount
    Target: Class (0=legitimate, 1=fraud)
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    print(f"Generating {n_samples:,} transactions ({n_fraud:,} fraud, {n_legit:,} legitimate)...")
    print(f"Fraud rate: {fraud_ratio*100:.2f}% — this is EXTREME class imbalance")
    
    # Legitimate transactions
    legit_features = np.random.randn(n_legit, 28)
    legit_time = np.random.uniform(0, 172800, n_legit)          # 48h window
    legit_amount = np.abs(np.random.lognormal(3.5, 1.5, n_legit))
    legit_amount = np.clip(legit_amount, 0.5, 5000)
    
    # Fraudulent transactions — different statistical signature
    fraud_features = np.random.randn(n_fraud, 28)
    # Fraud has distinct patterns in certain PCA components (V1, V4, V11, V14, V17)
    fraud_features[:, 0]  += np.random.normal(-3.5, 1.0, n_fraud)   # V1
    fraud_features[:, 3]  += np.random.normal(4.0, 1.2, n_fraud)    # V4
    fraud_features[:, 10] += np.random.normal(2.5, 0.8, n_fraud)    # V11
    fraud_features[:, 13] += np.random.normal(-5.0, 1.5, n_fraud)   # V14 (strongest signal)
    fraud_features[:, 16] += np.random.normal(-4.0, 1.0, n_fraud)   # V17
    fraud_features[:, 11] += np.random.normal(3.0, 1.0, n_fraud)    # V12
    
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    # Fraud amounts tend to be smaller (testing) or very large
    fraud_amount = np.where(
        np.random.random(n_fraud) < 0.7,
        np.abs(np.random.lognormal(2.0, 1.0, n_fraud)),    # Small probe transactions
        np.abs(np.random.lognormal(5.5, 1.0, n_fraud))     # Large fraud attempts
    )
    fraud_amount = np.clip(fraud_amount, 0.5, 10000)
    
    # Combine
    features = np.vstack([legit_features, fraud_features])
    time_col = np.concatenate([legit_time, fraud_time])
    amount_col = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    
    col_names = [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(features, columns=col_names)
    df['Time'] = time_col
    df['Amount'] = amount_col
    df['Class'] = labels.astype(int)
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


class FraudDetectionPipeline:
    """
    End-to-end fraud detection pipeline demonstrating:
    1. EDA on imbalanced data
    2. Multiple resampling strategies
    3. Supervised ML (LR, RF, XGBoost)
    4. Anomaly detection (Isolation Forest, One-Class SVM)
    5. Threshold optimization using business costs
    6. Proper evaluation with PR-AUC (not ROC-AUC alone)
    """
    
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.models = {}
        self.scaler = RobustScaler()   # More robust to outliers than StandardScaler
        
    def load_and_explore(self, df):
        """Step 1: EDA — understand the problem before modeling"""
        self.df = df.copy()
        self.feature_cols = [c for c in df.columns if c != 'Class']
        
        fraud = df[df['Class'] == 1]
        legit = df[df['Class'] == 0]
        fraud_pct = len(fraud) / len(df) * 100
        
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        print(f"Total transactions:    {len(df):>10,}")
        print(f"Legitimate (Class=0):  {len(legit):>10,} ({100-fraud_pct:.3f}%)")
        print(f"Fraudulent (Class=1):  {len(fraud):>10,} ({fraud_pct:.3f}%)")
        print(f"\nImbalance ratio: {len(legit)/len(fraud):.0f}:1")
        print("\n⚠ WHY ACCURACY IS MEANINGLESS HERE:")
        naive_accuracy = len(legit) / len(df) * 100
        print(f"  A model that predicts ALL transactions as legitimate")
        print(f"  achieves {naive_accuracy:.2f}% accuracy — but catches ZERO fraud!")
        print(f"  → Use Precision, Recall, F1, PR-AUC instead.")
        
        stats = {
            'total': len(df),
            'fraud': len(fraud),
            'legit': len(legit),
            'fraud_pct': fraud_pct,
            'imbalance_ratio': len(legit)/len(fraud),
            'naive_accuracy': naive_accuracy,
            'amount_fraud_mean': fraud['Amount'].mean(),
            'amount_legit_mean': legit['Amount'].mean(),
        }
        self.eda_stats = stats
        return stats
    
    def preprocess(self, test_size=0.2):
        """Step 2: Feature engineering and train/test split"""
        df = self.df.copy()
        
        # Feature engineering
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Hour'] = (df['Time'] % 86400) / 3600   # Hour of day
        df['Is_night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
        
        feature_cols = [c for c in df.columns if c not in ['Class', 'Time', 'Amount']]
        
        X = df[feature_cols]
        y = df['Class']
        
        # Stratified split — preserves class ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features (RobustScaler handles outliers better)
        self.scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        self.X_train = X_train_scaled
        self.X_test  = X_test_scaled
        self.y_train = y_train.reset_index(drop=True)
        self.y_test  = y_test.reset_index(drop=True)
        self.feature_cols = feature_cols
        
        print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"Train fraud: {y_train.sum():,} | Test fraud: {y_test.sum():,}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def apply_resampling(self, strategy='smote'):
        """
        Step 3: Handle class imbalance
        
        Interview talking points:
        - SMOTE creates synthetic minority samples (not duplicates)
        - Only apply resampling to TRAINING data, never test data
        - Resampling changes the decision boundary, not the threshold
        """
        X_train, y_train = self.X_train.copy(), self.y_train.copy()
        
        print(f"\nResampling strategy: {strategy.upper()}")
        print(f"Before: {y_train.sum():,} fraud / {(y_train==0).sum():,} legit")
        
        if not IMBLEARN_AVAILABLE or strategy == 'none':
            print("No resampling applied (using class_weight='balanced' in models)")
            return X_train, y_train
        
        if strategy == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif strategy == 'smotetomek':
            sampler = SMOTETomek(random_state=42)
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)
        
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(f"After:  {y_res.sum():,} fraud / {(y_res==0).sum():,} legit")
        
        self.X_train_res = pd.DataFrame(X_res, columns=self.X_train.columns)
        self.y_train_res = pd.Series(y_res)
        return self.X_train_res, self.y_train_res
    
    def train_supervised_models(self, X_train_res, y_train_res):
        """Step 4: Train supervised models"""
        print("\n" + "="*60)
        print("SUPERVISED MODELS")
        print("="*60)
        
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', max_iter=1000, C=0.1, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
        }
        
        if XGB_AVAILABLE:
            # scale_pos_weight handles imbalance natively
            fraud_ratio = (y_train_res == 0).sum() / (y_train_res == 1).sum()
            models['XGBoost'] = xgb.XGBClassifier(
                scale_pos_weight=fraud_ratio,
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='aucpr', random_state=42,
                use_label_encoder=False
            )
        else:
            models['Gradient Boosting'] = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start = time.time()
            model.fit(X_train_res, y_train_res)
            elapsed = time.time() - start
            
            # Get probability predictions
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Default threshold evaluation
            y_pred_default = (y_prob >= 0.5).astype(int)
            
            # Optimized threshold (business-cost aware)
            optimal_threshold, cost_savings = self.find_optimal_threshold(y_prob, self.y_test)
            y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
            
            metrics = self.evaluate_model(name, y_prob, y_pred_optimal, optimal_threshold)
            metrics['train_time'] = elapsed
            metrics['optimal_threshold'] = optimal_threshold
            metrics['cost_savings'] = cost_savings
            
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f} (@ threshold={optimal_threshold:.3f})")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.3f} (vs default 0.5)")
            print(f"  Est. cost savings vs no-model: ${cost_savings:,.0f}")
    
    def find_optimal_threshold(self, y_prob, y_true):
        """
        Threshold tuning using business costs — KEY interview talking point.
        
        Business context:
        - False Negative (miss fraud): avg loss = $500
        - False Positive (block legit): customer friction = $15
        
        We minimize: FN * cost_FN + FP * cost_FP
        """
        thresholds = np.linspace(0.01, 0.99, 200)
        min_cost = float('inf')
        optimal_threshold = 0.5
        
        # Cost of catching nothing (naive baseline)
        n_actual_fraud = y_true.sum()
        baseline_cost = n_actual_fraud * COST_FALSE_NEGATIVE
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_threshold = thresh
        
        cost_savings = baseline_cost - min_cost
        return optimal_threshold, cost_savings
    
    def evaluate_model(self, name, y_prob, y_pred, threshold):
        """Comprehensive evaluation metrics for imbalanced classification"""
        y_true = self.y_test
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'name': name,
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'tp': int(tp), 'fp': int(fp),
            'fn': int(fn), 'tn': int(tn),
            'threshold': threshold,
            'y_prob': y_prob.tolist(),
        }
    
    def train_anomaly_detectors(self):
        """
        Step 5: Unsupervised anomaly detection
        
        Interview talking points — Why Isolation Forest over One-Class SVM?
        - Isolation Forest: O(n log n), works on high-dim data, no kernel choice needed
        - One-Class SVM: O(n²) — doesn't scale to 284K transactions
        - IF isolates anomalies by random partitioning; frauds are easier to isolate
        - Doesn't need labels during training → useful for novel fraud patterns
        """
        print("\n" + "="*60)
        print("ANOMALY DETECTION (Unsupervised)")
        print("="*60)
        print("Training on legitimate transactions ONLY (no labels used)")
        
        # Train only on legitimate transactions
        X_legit = self.X_train[self.y_train == 0]
        
        detectors = {
            'Isolation Forest': IsolationForest(
                n_estimators=100, contamination=0.001,
                random_state=42, n_jobs=-1
            ),
        }
        
        # One-class SVM only on small sample (too slow for full data)
        print("\nNote: One-Class SVM is O(n²) — impractical for 284K rows.")
        print("This is exactly why Isolation Forest is preferred in production!")
        
        sample_size = min(5000, len(X_legit))
        X_legit_sample = X_legit.sample(n=sample_size, random_state=42)
        
        detectors['One-Class SVM (sample)'] = OneClassSVM(
            kernel='rbf', nu=0.01, gamma='auto'
        )
        
        for name, detector in detectors.items():
            print(f"\nTraining {name}...")
            start = time.time()
            
            if 'SVM' in name:
                detector.fit(X_legit_sample)
                # Predict on test set
                raw_scores = detector.decision_function(self.X_test)
            else:
                detector.fit(X_legit)
                raw_scores = detector.decision_function(self.X_test)
            
            elapsed = time.time() - start
            
            # Convert: IF outputs negative for anomalies; flip so higher = more anomalous
            anomaly_scores = -raw_scores
            
            # Normalize to [0, 1]
            s_min, s_max = anomaly_scores.min(), anomaly_scores.max()
            anomaly_prob = (anomaly_scores - s_min) / (s_max - s_min + 1e-9)
            
            pr_auc = average_precision_score(self.y_test, anomaly_prob)
            roc_auc = roc_auc_score(self.y_test, anomaly_prob)
            
            # Find threshold via business cost
            optimal_threshold, cost_savings = self.find_optimal_threshold(anomaly_prob, self.y_test)
            y_pred = (anomaly_prob >= optimal_threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            
            metrics = {
                'name': name,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'tp': int(tp), 'fp': int(fp),
                'fn': int(fn), 'tn': int(tn),
                'threshold': optimal_threshold,
                'cost_savings': cost_savings,
                'train_time': elapsed,
                'y_prob': anomaly_prob.tolist(),
                'type': 'anomaly_detection',
            }
            
            self.models[name] = detector
            self.results[name] = metrics
            
            print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f} (how many frauds caught?)")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Train time: {elapsed:.2f}s")
    
    def get_feature_importance(self):
        """Extract feature importances for interpretability"""
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = dict(zip(
                    self.feature_cols,
                    model.feature_importances_
                ))
        
        if importances:
            # Average across tree models
            avg_imp = {}
            for feat in self.feature_cols:
                vals = [importances[m].get(feat, 0) for m in importances]
                avg_imp[feat] = np.mean(vals)
            
            # Sort and return top 15
            sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
            self.feature_importances = sorted_imp[:15]
        else:
            self.feature_importances = []
        
        return self.feature_importances
    
    def generate_report(self):
        """Compile comprehensive results report"""
        report = {
            'eda': self.eda_stats,
            'models': {},
            'feature_importances': self.feature_importances,
            'business_costs': {
                'cost_fn': COST_FALSE_NEGATIVE,
                'cost_fp': COST_FALSE_POSITIVE,
            }
        }
        
        for name, metrics in self.results.items():
            report['models'][name] = {
                k: v for k, v in metrics.items() if k != 'y_prob'
            }
        
        # Save JSON report
        report_path = self.output_dir / 'results.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=float)
        
        print(f"\n✓ Results saved to {report_path}")
        return report
    
    def run(self):
        """Execute the full pipeline"""
        print("\n" + "█"*60)
        print("  CREDIT CARD FRAUD DETECTION PIPELINE")
        print("█"*60)
        
        # 1. Generate/load data
        df = pd.read_csv('data/creditcard.csv')
        
        # 2. EDA
        self.load_and_explore(df)
        
        # 3. Preprocessing
        self.preprocess()
        
        # 4. Resampling
        X_res, y_res = self.apply_resampling(strategy='smote' if IMBLEARN_AVAILABLE else 'none')
        
        # 5. Supervised models
        self.train_supervised_models(X_res, y_res)
        
        # 6. Anomaly detection
        self.train_anomaly_detectors()
        
        # 7. Feature importance
        self.get_feature_importance()
        
        # 8. Report
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print("\nModel Comparison (PR-AUC):")
        for name, m in sorted(self.results.items(), key=lambda x: x[1]['pr_auc'], reverse=True):
            bar = "█" * int(m['pr_auc'] * 40)
            print(f"  {name:<30} {m['pr_auc']:.4f}  {bar}")
        
        return report


if __name__ == "__main__":
    pipeline = FraudDetectionPipeline(output_dir="reports")
    report = pipeline.run()
