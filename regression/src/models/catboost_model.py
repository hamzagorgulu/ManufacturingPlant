import numpy as np
from typing import Dict, Any
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """CatBoost implementation with hyperparameter optimization."""
    
    def __init__(self, n_trials: int = 50):
        """
        Initialize CatBoost model.
        
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
                "iterations": trial.suggest_int("iterations", 50, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                "depth": trial.suggest_int("depth", 2, 15),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_strength": trial.suggest_float("random_strength", 1e-6, 10, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "border_count": trial.suggest_int("border_count", 16, 255)
            }
            
            model = CatBoostRegressor(**params, random_state=42, verbose=0)
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
            self.model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
        else:
            self.model = CatBoostRegressor(random_state=42, verbose=0)
            
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