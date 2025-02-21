
import matplotlib.pyplot as plt

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