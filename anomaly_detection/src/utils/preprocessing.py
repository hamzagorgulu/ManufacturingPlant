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
    
    def prepare_data(self, features: pd.DataFrame, scale: bool = True, fit_scaler: bool = False) -> np.ndarray:
        """Prepare features for modeling.
        
        Args:
            features: Input features as a DataFrame.
            scale: Whether to scale the features.
            fit_scaler: Whether to fit the scaler on the provided data (use this for training data).
            
        Returns:
            np.ndarray: Prepared features.
        """
        if 'time_to_failure' in features.columns:
            features = features.drop('time_to_failure', axis=1)
            
        if scale:
            if fit_scaler:
                # Fit the scaler on the training data
                scaled_features = self.scaler.fit_transform(features)
            else:
                # Transform the test data using the fitted scaler
                scaled_features = self.scaler.transform(features)
            return scaled_features
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
    
    def extract_datetime_features(self, df):
        #df["minute"] = df.index.minute
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['weekday'] = df.index.weekday
        return df
    
    def extract_statistical_features(self, df, column='motor_current'):
        # moving average
        df['MA'] = df[column].rolling(window=60*20).mean()

        # rolling standard deviation
        df['STD'] = df[column].rolling(window=60*20).std()

        # Fourier transform, use numpy
        df['FT'] = np.fft.fft(df[column])
        # convert complex numbers to real numbers
        df['FT'] = df['FT'].apply(lambda x: np.abs(x))

        return df