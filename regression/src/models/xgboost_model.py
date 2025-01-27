import numpy as np
from typing import Dict, Any
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost implementation with hyperparameter optimization."""
    
    def __init__(self, n_trials: int = 50):
        """
        Initialize XGBoost model.
        
        Args:
            n_trials (int): Number of optimization trials
        """
        self.n_trials = n_trials
        self.model = None
        self.feature_names = None
        
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Returns:
            Dict[str, Any]: Best parameters
        """
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10)
            }
            
            model = XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            return np.sqrt(mean_squared_error(y_test, predictions))
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None) -> None:
        """Train the model with optional hyperparameter optimization."""
        self.feature_names = X_train.columns if hasattr(X_train, 'columns') else None
        
        if X_test is not None and y_test is not None:
            best_params = self.optimize(X_train, y_train, X_test, y_test)
            self.model = XGBRegressor(**best_params, random_state=42)
        else:
            self.model = XGBRegressor(random_state=42)
            
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X_test)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.feature_importances_
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        return dict(enumerate(importance)) 