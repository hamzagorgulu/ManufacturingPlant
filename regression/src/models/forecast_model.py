from typing import Dict, List, Union, Any
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from .base_model import BaseModel

class ForecasterModel(BaseModel):
    """Model wrapper for multi-step forecasting."""
    
    def __init__(self, base_model: BaseModel, method: str = 'recursive', 
                 forecast_horizon: int = 5):
        """
        Initialize forecaster.
        
        Args:
            base_model: Base model to use (XGBoost, CatBoost, etc.)
            method: Forecasting method ('recursive', 'direct', or 'multiple')
            forecast_horizon: Number of steps to forecast
        """
        self.base_model = base_model
        self.method = method
        self.horizon = forecast_horizon
        self.models = {}
        self.feature_names = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_test: np.ndarray = None, y_test: np.ndarray = None) -> None:
        """Train the forecasting model."""
        self.feature_names = X_train.columns if hasattr(X_train, 'columns') else None
        
        if self.method == 'recursive':
            self.models[1] = self.base_model
            self.models[1].train(X_train, y_train, X_test, y_test)
            
        elif self.method == 'direct':
            for h in range(1, self.horizon + 1):
                self.models[h] = self.base_model.__class__()  # Create new instance
                self.models[h].train(X_train, y_train[:, h-1], X_test, y_test[:, h-1])
                
        elif self.method == 'multiple':
            multi_model = MultiOutputRegressor(self.base_model)
            multi_model.fit(X_train, y_train)
            self.models['multi'] = multi_model
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the selected forecasting method."""
        if self.method == 'recursive':
            predictions = np.zeros((len(X), self.horizon))
            current_X = X.copy()
            
            for h in range(self.horizon):
                pred = self.models[1].predict(current_X)
                predictions[:, h] = pred
                
                # Update features for next prediction
                if h < self.horizon - 1:
                    current_X = self._update_features(current_X, pred)
                    
            return predictions
            
        elif self.method == 'direct':
            return np.column_stack([
                self.models[h].predict(X) 
                for h in range(1, self.horizon + 1)
            ])
            
        elif self.method == 'multiple':
            return self.models['multi'].predict(X)
            
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the underlying model(s)."""
        if self.method == 'recursive':
            return self.models[1].get_feature_importance()
        elif self.method == 'direct':
            # Average feature importance across all horizons
            importances = {}
            for h in range(1, self.horizon + 1):
                h_importance = self.models[h].get_feature_importance()
                for feature, importance in h_importance.items():
                    importances[feature] = importances.get(feature, 0) + importance / self.horizon
            return importances
        elif self.method == 'multiple':
            # For multiple output, get importance from the first estimator
            return self.models['multi'].estimators_[0].get_feature_importance()
            
    def _update_features(self, X: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """Update features for recursive forecasting."""
        # Implement feature updating logic based on your feature engineering
        # This is highly dependent on your feature set
        X_new = X.copy()
        # Update relevant features with new prediction
        return X_new 