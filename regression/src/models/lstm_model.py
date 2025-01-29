import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
import optuna
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce logging output
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

torch.manual_seed(42)  # Set seed for PyTorch
np.random.seed(42)  # Set seed for NumPy

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_column, seq_length):
        self.data = data
        self.target_column = target_column
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data.iloc[index:index + self.seq_length][self.target_column].values
        y = self.data.iloc[index + self.seq_length][self.target_column]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModelWrapper:
    """LSTM implementation with training, evaluation, prediction, and hyperparameter optimization."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output features
            dropout (float): Dropout rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_length = None
        
    def optimize(self, data: pd.DataFrame, target_column: str, seq_length: int, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of the target column
            seq_length (int): Sequence length for LSTM
            n_trials (int): Number of optimization trials
        
        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            
            # Prepare data
            data_scaled = self.scaler.fit_transform(data[[target_column]])
            data[target_column + '_scaled'] = data_scaled
            
            dataset = TimeSeriesDataset(data, target_column=target_column + '_scaled', seq_length=seq_length)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            
            train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
            
            # Initialize model with trial parameters
            model = LSTMModel(
                input_size=self.input_size,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                output_size=self.output_size,
                dropout=params["dropout"]
            )
            
            # Define optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
            criterion = nn.MSELoss()
            
            # Training loop
            for epoch in range(10):  # Short training for hyperparameter tuning
                model.train()
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.unsqueeze(-1)
                    y_batch = y_batch.unsqueeze(-1)
                    
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation on the test set
            model.eval()
            y_test_actual = []
            y_test_predicted = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.unsqueeze(-1)
                    y_batch = y_batch.unsqueeze(-1)
                    
                    predictions = model(x_batch)
                    y_test_actual.extend(y_batch.numpy())
                    y_test_predicted.extend(predictions.numpy())
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_predicted))
            return rmse
        
        # Run Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
    
    def train(self, data: pd.DataFrame, target_column: str, seq_length: int, num_epochs: int = 150, batch_size: int = 32, learning_rate: float = 0.001, use_optuna: bool = False) -> None:
        """
        Train the LSTM model with optional hyperparameter optimization.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of the target column
            seq_length (int): Sequence length for LSTM
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            use_optuna (bool): Whether to use Optuna for hyperparameter optimization
        """
        self.seq_length = seq_length
        
        if use_optuna:
            print("Optimizing hyperparameters with Optuna...")
            best_params = self.optimize(data, target_column, seq_length)
            print(f"Best hyperparameters: {best_params}")
            self.hidden_size = best_params["hidden_size"]
            self.num_layers = best_params["num_layers"]
            self.dropout = best_params["dropout"]
            learning_rate = best_params["learning_rate"]
            batch_size = best_params["batch_size"]
        
        # Prepare data
        data_scaled = self.scaler.fit_transform(data[[target_column]])
        data[target_column + '_scaled'] = data_scaled
        
        dataset = TimeSeriesDataset(data, target_column=target_column + '_scaled', seq_length=seq_length)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout
        )
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.unsqueeze(-1)
                
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the test set."""
        self.model.eval()
        y_test_actual = []
        y_test_predicted = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.unsqueeze(-1)

                predictions = self.model(x_batch)
                y_test_actual.extend(y_batch.numpy())
                y_test_predicted.extend(predictions.numpy())

        # Rescale the predictions back to original values
        y_test_actual_rescaled = self.scaler.inverse_transform(np.array(y_test_actual).reshape(-1, 1))
        y_test_predicted_rescaled = self.scaler.inverse_transform(np.array(y_test_predicted).reshape(-1, 1))

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_actual_rescaled, y_test_predicted_rescaled)
        rmse = np.sqrt(mean_squared_error(y_test_actual_rescaled, y_test_predicted_rescaled))

        print("\nModel Evaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        
        return {"MAE": mae, "RMSE": rmse}
    
    def predict(self, data: pd.DataFrame, target_column: str, num_predictions: int = 5) -> np.ndarray:
        """Make future predictions."""
        if self.seq_length is None:
            raise ValueError("Model not trained yet")
            
        data_scaled = self.scaler.transform(data[[target_column]])
        data[target_column + '_scaled'] = data_scaled
        
        last_sequence = torch.tensor(data[target_column + '_scaled'].values[-self.seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        self.model.eval()
        future_predictions = []
        for _ in range(num_predictions):
            with torch.no_grad():
                next_prediction = self.model(last_sequence).item()
                future_predictions.append(next_prediction)

                # Update the sequence with the predicted value
                last_sequence = torch.cat((last_sequence[:, 1:, :], torch.tensor([[next_prediction]], dtype=torch.float32).unsqueeze(-1)), dim=1)

        # Rescale future predictions back to original values
        future_predictions_rescaled = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        return future_predictions_rescaled.flatten()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the sequence
        return out
