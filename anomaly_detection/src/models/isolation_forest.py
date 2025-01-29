import numpy as np
from typing import Dict, Any, Optional, Tuple
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
    
    def get_individual_prediction_intervals(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate individual prediction intervals for each sample using the ensemble of trees.
        
        Args:
            X (np.ndarray): Input data
            confidence (float): Confidence level for the interval (e.g., 0.95 for 95% confidence)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of the prediction intervals for each sample
        """
        # Get the anomaly scores from each tree in the ensemble
        tree_scores = np.array([tree.score_samples(X) for tree in self.model.estimators_])
        
        # Calculate the lower and upper percentiles for each sample
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_bounds = np.percentile(tree_scores, lower_percentile, axis=0)
        upper_bounds = np.percentile(tree_scores, upper_percentile, axis=0)
        
        return lower_bounds, upper_bounds