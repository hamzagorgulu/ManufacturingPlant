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

        rename_cols = {
            'Date': 'datetime',
            'Value': 'motor_current'
        }

        df = df.rename(columns=rename_cols)
        
        # Convert Date to datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['datetime'].dt.date)

        # Set datetime as index
        df = df.set_index('datetime')
        
        # Sort by timestamp
        df = df.sort_index()
                
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
                train_end_date: str, 
                test_normal_end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets based on dates.
        
        Args:
            df (pd.DataFrame): Input dataframe
            train_end_date (str): End date for training data (format: 'YYYY-MM-DD')
            val_end_date (str): End date for validation data (format: 'YYYY-MM-DD')
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing split datasets
        """
        # Convert date strings to pandas Timestamp
        train_end_date = pd.Timestamp(train_end_date)
        test_normal_end_date = pd.Timestamp(test_normal_end_date)
        
        # Split data based on dates
        train_data = df[df.index <= train_end_date]
        test_normal_data = df[(df.index > train_end_date) & (df.index <= test_normal_end_date)]
        test_anomaly_data = df[df.index > test_normal_end_date]
        
        return {
            'train': train_data,
            'test_normal': test_normal_data,
            'test_anomaly': test_anomaly_data
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