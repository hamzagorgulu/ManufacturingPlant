import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from .base_model import BaseAnomalyDetector

class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest implementation for anomaly detection."""
    
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, 
                 random_state: int = 42, **kwargs):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators (int): Number of base estimators
            contamination (float): Expected proportion of anomalies
            random_state (int): Random state for reproducibility
            **kwargs: Additional parameters for IsolationForest
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            **kwargs
        )
        self.threshold = None
    
    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X (np.ndarray): Training data
            **kwargs: Additional training parameters
        """
        self.model.fit(X)
        # Calculate threshold based on training data
        self.threshold = self.get_threshold(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Binary predictions (1 for anomaly, 0 for normal)
        """
        scores = self.score_samples(X)
        return (scores < self.threshold).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Anomaly scores (lower scores indicate higher anomaly probability)
        """
        return self.model.score_samples(X)
    
    def get_threshold(self, X: np.ndarray, contamination: float = 0.1) -> float:
        """
        Calculate anomaly threshold based on training data.
        
        Args:
            X (np.ndarray): Training data
            contamination (float): Expected proportion of anomalies
            
        Returns:
            float: Threshold value
        """
        scores = self.score_samples(X)
        return np.percentile(scores, contamination * 100) 