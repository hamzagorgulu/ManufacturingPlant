import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional
from .base_model import BaseAnomalyDetector

class Encoder(nn.Module):
    """Encoder part of autoencoder."""
    
    def __init__(self, input_dim: int, encoding_dims: List[int], dropout_rate: float = 0.1):
        """
        Initialize encoder.
        
        Args:
            input_dim (int): Input dimension
            encoding_dims (List[int]): Dimensions of encoding layers
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.encoder(x)

class Decoder(nn.Module):
    """Decoder part of autoencoder."""
    
    def __init__(self, encoded_dim: int, decoding_dims: List[int], output_dim: int):
        """
        Initialize decoder.
        
        Args:
            encoded_dim (int): Input dimension (encoded)
            decoding_dims (List[int]): Dimensions of decoding layers
            output_dim (int): Output dimension
        """
        super().__init__()
        
        layers = []
        prev_dim = encoded_dim
        
        for dim in decoding_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ])
            prev_dim = dim
            
        # Final layer to reconstruct input
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.decoder(x)

class AutoencoderModel(nn.Module):
    """Complete autoencoder model."""
    
    def __init__(self, input_dim: int, encoding_dims: List[int], dropout_rate: float = 0.1):
        """Initialize autoencoder."""
        super().__init__()
        self.encoder = Encoder(input_dim, encoding_dims, dropout_rate)
        self.decoder = Decoder(encoding_dims[-1], encoding_dims[-2::-1], input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderDetector(BaseAnomalyDetector):
    """PyTorch Autoencoder implementation for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dims: List[int] = [64, 32, 16],
                 learning_rate: float = 0.001, dropout_rate: float = 0.1,
                 device: str = None):
        """
        Initialize Autoencoder detector.
        
        Args:
            input_dim (int): Input dimension
            encoding_dims (List[int]): Dimensions of encoding layers
            learning_rate (float): Learning rate for optimization
            dropout_rate (float): Dropout rate for regularization
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = AutoencoderModel(
            input_dim=input_dim,
            encoding_dims=encoding_dims,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.threshold = None
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32,
            validation_split: float = 0.1, patience: int = 10, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X (np.ndarray): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation data ratio
            patience (int): Early stopping patience
            **kwargs: Additional training parameters
        """
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                x = batch[0]
                output = self.model(x)
                loss = self.criterion(output, x)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.threshold = self.get_threshold(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error as anomaly scores.
        
        Returns:
            np.ndarray: Anomaly scores (higher scores indicate higher anomaly probability)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            scores = torch.mean(torch.square(X_tensor - reconstructed), dim=1)
            return scores.cpu().numpy()
    
    def get_threshold(self, X: np.ndarray, contamination: float = 0.1) -> float:
        """Calculate anomaly threshold based on reconstruction errors."""
        scores = self.score_samples(X)
        return np.percentile(scores, (1 - contamination) * 100)
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'dropout_rate': self.dropout_rate
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint['threshold'] 