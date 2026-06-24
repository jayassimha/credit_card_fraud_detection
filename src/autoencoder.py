"""
Autoencoder for Fraud Detection
================================
Deep learning anomaly detection: Train autoencoder on normal transactions.
High reconstruction error → anomalous (potential fraud).

Key insight: The autoencoder learns to reconstruct "normal" patterns.
When it sees a fraudulent transaction, it cannot reconstruct it well,
resulting in high reconstruction error = fraud signal.
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FraudAutoencoder(nn.Module):
    """
    Bottleneck autoencoder architecture.
    Encoder compresses → Bottleneck forces meaningful representation → Decoder reconstructs.
    Trained ONLY on legitimate transactions (unsupervised).
    """
    def __init__(self, input_dim, encoding_dim=8):
        super().__init__()
        
        # Encoder: compress from input_dim → encoding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU(),
        )
        
        # Decoder: reconstruct from encoding_dim → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def reconstruction_error(self, x):
        """MSE reconstruction error per sample — higher = more anomalous"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return errors


class AutoencoderDetector:
    """
    Trains and evaluates the autoencoder fraud detector.
    
    Architecture decision rationale (for interviews):
    - Bottleneck (8 dims) forces the model to learn compact normal representations
    - Dropout prevents overfitting to noise
    - MSE loss on output: minimized for normal patterns, high for anomalies
    - Alternative: Variational Autoencoder (VAE) gives probabilistic interpretation
    """
    
    def __init__(self, encoding_dim=8, epochs=30, batch_size=256, lr=1e-3):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.threshold = None
        
    def fit(self, X_legit: np.ndarray):
        """Train on legitimate transactions only"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Autoencoder skipped.")
            return self
        
        input_dim = X_legit.shape[1]
        self.model = FraudAutoencoder(input_dim, self.encoding_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_legit)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        train_losses = []
        
        print(f"\nTraining Autoencoder ({self.epochs} epochs)...")
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}: loss={avg_loss:.6f}")
        
        self.train_losses = train_losses
        
        # Set anomaly threshold at 95th percentile of legitimate reconstruction errors
        self.model.eval()
        with torch.no_grad():
            recon_errors = self.model.reconstruction_error(X_tensor).numpy()
        self.threshold = np.percentile(recon_errors, 95)
        print(f"  Anomaly threshold (95th pct): {self.threshold:.6f}")
        
        return self
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Returns normalized reconstruction errors as anomaly scores"""
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            errors = self.model.reconstruction_error(X_tensor).numpy()
        
        # Normalize to [0, 1]
        e_min, e_max = errors.min(), errors.max()
        return (errors - e_min) / (e_max - e_min + 1e-9)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions using threshold"""
        scores = self.anomaly_scores(X)
        normalized_threshold = (self.threshold - 0) / (scores.max() + 1e-9)
        return (scores >= normalized_threshold).astype(int)


if __name__ == "__main__":
    # Quick test
    print("Autoencoder module ready.")
    if TORCH_AVAILABLE:
        print(f"PyTorch available: {torch.__version__}")
        # Test with dummy data
        X = np.random.randn(1000, 30).astype(np.float32)
        ae = AutoencoderDetector(epochs=5)
        ae.fit(X)
        scores = ae.anomaly_scores(X[:10])
        print(f"Sample scores: {scores}")
    else:
        print("PyTorch not installed. Autoencoder will be skipped.")
