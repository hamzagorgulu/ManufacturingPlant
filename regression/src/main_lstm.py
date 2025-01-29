import pandas as pd
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from utils.preprocessing import TimeSeriesPreprocessor
from models.lstm_model import LSTMModelWrapper  # Import the LSTM model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def plot_predictions(y_test, predictions_dict, title="Actual vs Predictions"):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='green')
    
    colors = ['blue', 'orange', 'red']
    for (name, preds), color in zip(predictions_dict.items(), colors):
        plt.plot(y_test.index, preds, label=name, linestyle='--', color=color)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Water Inflow')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Initialize components
    data_loader = DataLoader('regression/data/multistep_regression.csv')
    preprocessor = TimeSeriesPreprocessor()
    
    # Load and preprocess data
    data = data_loader.load_data()
    data = preprocessor.create_features(data)
    train, test = data_loader.split_data(data)
    
    # Prepare features for LSTM
    target_column = 'Value'  # Replace with your target column name
    seq_length = 12  # Sequence length for LSTM
    
    # Initialize and train models
    models = {
        'LSTM': LSTMModelWrapper(input_size=1, hidden_size=100, num_layers=2, output_size=1, dropout=0.2)
    }
    
    predictions = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Train the model with Optuna hyperparameter optimization
        model.train(
            data, 
            target_column=target_column, 
            seq_length=seq_length, 
            num_epochs=150, 
            batch_size=8, 
            learning_rate=0.001, 
            use_optuna=True  # Enable Optuna for hyperparameter optimization
        )
        
        # Make predictions
        predictions[name] = model.predict(test, target_column=target_column, num_predictions=len(test))
        
        # Calculate metrics
        y_test = test[target_column].values[-len(predictions[name]):]  # Align test data with predictions
        mae = mean_absolute_error(y_test, predictions[name])
        rmse = np.sqrt(mean_squared_error(y_test, predictions[name]))
        mape = mean_absolute_percentage_error(y_test, predictions[name])
        print(f"{name} MAE: {mae:.2f}")
        print(f"{name} RMSE: {rmse:.2f}")
        print(f"{name} MAPE(%): {mape*100:.2f}%")  # Convert to percentage
    
    # Plot results
    plot_predictions(test[target_column].iloc[-len(predictions['LSTM']):], predictions)

if __name__ == "__main__":
    main()