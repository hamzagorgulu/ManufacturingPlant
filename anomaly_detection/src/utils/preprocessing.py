import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler

class AnomalyPreprocessor:
    """Class for preprocessing time series data for anomaly detection."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame, 
                       windows: List[int] = [5, 10, 20, 50],
                       freq_components: int = 5) -> pd.DataFrame:
        """
        Create features for anomaly detection.
        
        Args:
            df (pd.DataFrame): Input dataframe
            windows (List[int]): Window sizes for rolling statistics
            freq_components (int): Number of frequency components to use
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)
        signal = df['Value'].values  # Convert to numpy array for faster operations
        features['value'] = signal
        
        # Vectorized operations for rolling statistics
        signal_series = pd.Series(signal, index=df.index)
        for window in windows:
            # Calculate rolling statistics with min_periods=1
            roll = signal_series.rolling(window=window, min_periods=1)
            features[f'rolling_mean_{window}'] = roll.mean()
            features[f'rolling_std_{window}'] = roll.std()
            
            # For the first window-1 values, use expanding statistics
            expanding = signal_series.expanding()
            mask = features[f'rolling_mean_{window}'].isna()
            features.loc[mask, f'rolling_mean_{window}'] = expanding.mean()[mask]
            features.loc[mask, f'rolling_std_{window}'] = expanding.std()[mask]
            
            # Calculate differences
            features[f'mean_diff_{window}'] = signal - features[f'rolling_mean_{window}']
            std_vals = features[f'rolling_std_{window}']
            features[f'std_diff_{window}'] = np.abs(features[f'mean_diff_{window}']) / np.where(std_vals == 0, 1, std_vals)
        
        # Rate of change features
        features['diff'] = signal_series.diff()
        features['diff'].iloc[0] = 0  # Set first difference to 0
        
        features['diff2'] = features['diff'].diff()
        features['diff2'].iloc[0:2] = 0  # Set first two second differences to 0
        
        # Time-based features
        features['hour'] = df.index.hour
        features['minute'] = df.index.minute
        
        # EWM calculations with minimum periods
        features['ewm_mean'] = signal_series.ewm(span=20, min_periods=1, adjust=False).mean()
        features['ewm_std'] = signal_series.ewm(span=20, min_periods=1, adjust=False).std()
        
        # Rolling quantiles with expanding window for initial values
        for q in [0.25, 0.75]:
            roll_q = signal_series.rolling(window=20, min_periods=1).quantile(q)
            exp_q = signal_series.expanding().quantile(q)
            mask = roll_q.isna()
            features[f'rolling_q{int(q*100)}'] = roll_q
            features.loc[mask, f'rolling_q{int(q*100)}'] = exp_q[mask]
        
        # Range calculation with expanding window for initial values
        roll_max = signal_series.rolling(window=20, min_periods=1).max()
        roll_min = signal_series.rolling(window=20, min_periods=1).min()
        exp_max = signal_series.expanding().max()
        exp_min = signal_series.expanding().min()
        
        features['rolling_range'] = roll_max - roll_min
        mask = features['rolling_range'].isna()
        features.loc[mask, 'rolling_range'] = exp_max[mask] - exp_min[mask]
        
        # Ensure no NaN values remain
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_names = features.columns.tolist()
        return features
    
    def _get_fft_features(self, signal: np.ndarray, component: int) -> np.ndarray:
        """Optimized FFT feature extraction."""
        window_size = 50
        n = len(signal)
        fft_features = np.zeros(n)
        
        # Compute FFT for all windows at once
        for i in range(window_size, n):
            if component < window_size:
                fft_features[i] = np.abs(fft(signal[i-window_size:i]))[component]
        
        return fft_features
    
    def prepare_data(self, features: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Prepare features for modeling."""
        if 'time_to_failure' in features.columns:
            features = features.drop('time_to_failure', axis=1)
            
        if scale:
            return self.scaler.fit_transform(features)
        return features.values
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def get_feature_importance(self, model, feature_names: List[str] = None) -> pd.Series:
        """
        Get feature importance if model supports it.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            pd.Series: Feature importance
        """
        if feature_names is None:
            feature_names = self.feature_names
            
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
            return importance
        return None 