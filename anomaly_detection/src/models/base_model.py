from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detection models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Train the anomaly detection model.
        
        Args:
            X (np.ndarray): Training data
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels (1 for anomaly, 0 for normal).
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Binary predictions
        """
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Anomaly scores
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.__dict__
    
    @abstractmethod
    def get_threshold(self, X: np.ndarray, contamination: float = 0.1) -> float:
        """
        Calculate anomaly threshold based on training data.
        
        Args:
            X (np.ndarray): Training data
            contamination (float): Expected proportion of anomalies
            
        Returns:
            float: Threshold value
        """
        pass 