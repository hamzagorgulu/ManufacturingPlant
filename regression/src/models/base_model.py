from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        pass 