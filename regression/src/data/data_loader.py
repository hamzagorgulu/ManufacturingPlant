import pandas as pd
from typing import Tuple

class DataLoader:
    """Class for loading and basic preprocessing of time series data."""
    
    def __init__(self, file_path: str):
        """
        Initialize DataLoader with file path.
        
        Args:
            file_path (str): Path to the CSV data file
        """
        self.file_path = file_path
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV and perform initial datetime processing.
        
        Returns:
            pd.DataFrame: Processed dataframe with datetime index
        """
        # Load data
        data = pd.read_csv(self.file_path)
        
        # Convert to datetime index
        data["date"] = data["Year"].astype(str) + "-" + data["Month"].astype(str) + "-01"
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
        
        return data
    
    def split_data(self, data: pd.DataFrame, split_year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets based on year.
        
        Args:
            data (pd.DataFrame): Input dataframe
            split_year (int): Year to split the data on
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
        """
        train = data[data.index.year < split_year]
        test = data[(data.index.year >= split_year) & (data.index.month <= 5)]
        
        return train, test 