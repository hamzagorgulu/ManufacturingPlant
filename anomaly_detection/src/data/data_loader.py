import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

class AnomalyDataLoader:
    """Class for loading and basic preprocessing of anomaly detection data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir (str): Directory containing the dataset files
        """
        self.data_dir = Path(data_dir)
        self.failure_dates = {
            'dataset1': pd.Timestamp('2021-12-15'),
            'dataset2': pd.Timestamp('2021-12-30')
        }
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load specific dataset and perform initial preprocessing.
        
        Args:
            dataset_name (str): Name of the dataset to load ('dataset1' or 'dataset2')
            
        Returns:
            pd.DataFrame: Preprocessed dataframe with datetime index
        """
        # Map dataset names to file names
        file_mapping = {
            'dataset1': 'anomaly_detection_01.csv',
            'dataset2': 'anomaly_detection_02.csv'
        }
        
        file_path = self.data_dir / file_mapping[dataset_name]
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Add time to failure information
        failure_date = self.failure_dates[dataset_name]
        df['time_to_failure'] = (failure_date - df.index).total_seconds() / 3600  # hours until failure
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all datasets
        """
        return {
            'dataset1': self.load_data('dataset1'),
            'dataset2': self.load_data('dataset2')
        }
    
    def split_data(self, df: pd.DataFrame, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        Note: For anomaly detection, we typically use only normal data for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing split datasets
        """
        # Find the point where time_to_failure is sufficiently large (normal operation)
        normal_data = df[df['time_to_failure'] > 168]  # More than 1 week from failure
        anomaly_data = df[df['time_to_failure'] <= 168]  # Within 1 week of failure
        
        # Split normal data
        n = len(normal_data)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_data = normal_data.iloc[:train_idx]
        val_data = normal_data.iloc[train_idx:val_idx]
        test_data = pd.concat([normal_data.iloc[val_idx:], anomaly_data])
        test_data = test_data.sort_index()
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def get_sampling_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate the average sampling rate in seconds.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            float: Average sampling rate in seconds
        """
        time_diff = df.index.to_series().diff().mean()
        return time_diff.total_seconds() 