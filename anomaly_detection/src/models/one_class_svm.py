from sklearn.svm import OneClassSVM
from typing import Dict, Any, Optional, Tuple
import numpy as np
from .base_model import BaseAnomalyDetector

class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM implementation for anomaly detection."""
    
    def __init__(self, kernel: str = 'rbf', nu: float = 0.1, gamma: str = 'scale', 
                 random_state: int = 42, **kwargs):
        """
        Initialize One-Class SVM detector.
        
        Args:
            kernel (str): Kernel type for SVM (e.g., 'rbf', 'linear', 'poly')
            nu (float): An upper bound on the fraction of training errors and a lower bound 
                       on the fraction of support vectors. Should be in the range (0, 1].
            gamma (str): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
            random_state (int): Random state for reproducibility.
            **kwargs: Additional parameters for OneClassSVM.
        """
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            **kwargs
        )
        self.threshold = None
    
    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X (np.ndarray): Training data.
            **kwargs: Additional training parameters.
        """
        self.model.fit(X)
        # Calculate threshold based on training data
        self.threshold = self.get_threshold(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Binary predictions (1 for anomaly, 0 for normal).
        """
        scores = self.score_samples(X)
        return (scores < self.threshold).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            X (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Anomaly scores (lower scores indicate higher anomaly probability).
        """
        return -self.model.decision_function(X)  # Negative decision function for consistency
    
    def get_threshold(self, X: np.ndarray, contamination: float = 0.1) -> float:
        """
        Calculate anomaly threshold based on training data.
        
        Args:
            X (np.ndarray): Training data.
            contamination (float): Expected proportion of anomalies.
            
        Returns:
            float: Threshold value.
        """
        scores = self.score_samples(X)
        return np.percentile(scores, contamination * 100)
    
    def get_individual_prediction_intervals(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate individual prediction intervals for each sample using the decision function.
        
        Args:
            X (np.ndarray): Input data.
            confidence (float): Confidence level for the interval (e.g., 0.95 for 95% confidence).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of the prediction intervals for each sample.
        """
        scores = self.score_samples(X)
        lower_bound = np.percentile(scores, (1 - confidence) / 2 * 100)
        upper_bound = np.percentile(scores, (1 + confidence) / 2 * 100)
        
        return np.full_like(scores, lower_bound), np.full_like(scores, upper_bound)