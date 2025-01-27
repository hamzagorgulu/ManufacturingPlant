import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class TimeSeriesPreprocessor:
    """Class for preprocessing time series data with feature engineering."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = StandardScaler()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for time series prediction.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        # Sort dataframe chronologically
        df = df.sort_values(['Year', 'Month'])
        
        # Calculate yearly averages without future leakage
        yearly_means = df.groupby('Year')['Value'].mean()
        
        # Add features
        df['Prev_Year_Mean'] = df['Year'].map(yearly_means.shift(1))
        df['Lag1_Inflow_6'] = df['Value'].shift(6)
        df['Lag1_Inflow_12'] = df['Value'].shift(12)
        df['Short_vs_Long_Term'] = (
            df['Value'].shift(5).rolling(window=3).mean() - 
            df['Value'].shift(5).rolling(window=12).mean()
        )
        
        # Cyclical encoding
        df['Month_Sin'] = np.sin(df['Month'] * (2 * np.pi / 12))
        df['Month_Cos'] = np.cos(df['Month'] * (2 * np.pi / 12))
        df['Year_Sin'] = np.sin(df['Year'] * (2 * np.pi / 12))
        df['Year_Cos'] = np.cos(df['Year'] * (2 * np.pi / 12))
        
        return df.dropna()
    
    def prepare_data(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for modeling by dropping unnecessary columns and splitting features/target.
        
        Args:
            train (pd.DataFrame): Training data
            test (pd.DataFrame): Test data
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        # Drop Year and Month columns
        for df in [train, test]:
            if 'Year' in df.columns:
                df.drop(['Year', 'Month'], axis=1, inplace=True)
        
        # Split features and target
        X_train, y_train = train.drop(columns=['Value']), train['Value']
        X_test, y_test = test.drop(columns=['Value']), test['Value']
        
        return X_train, X_test, y_train, y_test 