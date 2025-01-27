import pandas as pd
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from utils.preprocessing import TimeSeriesPreprocessor
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
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
    data_loader = DataLoader('data/multistep_regression.csv')
    preprocessor = TimeSeriesPreprocessor()
    
    # Load and preprocess data
    data = data_loader.load_data()
    data = preprocessor.create_features(data)
    train, test = data_loader.split_data(data)
    
    # Prepare features
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(train, test)
     
    # Initialize and train models
    models = {
        'XGBoost': XGBoostModel(n_trials=50),
        'CatBoost': CatBoostModel(n_trials=50)
    }
    
    predictions = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.train(X_train, y_train, X_test, y_test)
        predictions[name] = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions[name])
        rmse = np.sqrt(mean_squared_error(y_test, predictions[name]))
        mape = mean_absolute_percentage_error(y_test, predictions[name])
        print(f"{name} MAE: {mae:.2f}")
        print(f"{name} RMSE: {rmse:.2f}")
        print(f"{name} MAPE(%): {mape*100:.2f}%")  # Convert to percentage
        
        # Print feature importance
        print(f"\n{name} Feature Importance:")
        importance = model.get_feature_importance()
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {imp:.4f}")
    
    # Plot results
    plot_predictions(y_test, predictions)

if __name__ == "__main__":
    main() 