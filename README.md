# Anomaly Detection and Time Series Forecasting

This repository contains implementations of anomaly detection and multi-step regression tasks using various machine learning models.

## Project Structure

```
anomaly_detection/
├── data/
│   └── multistep_regression.csv
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── catboost_model.py
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py
│       └── evaluation.py
├── notebooks/
│   └── analysis.ipynb
└── requirements.txt
```

## Quick Start

The project includes a Dockerfile for easy reproduction of results. To get started:

1. Build the Docker image:
```bash
docker build -t case_study_image .
```

2. Run the container:
```bash
docker run case_study_image
```

This will automatically:
- Set up the required environment
- Execute the following scripts:
  - `anomaly_detection/src/main_isolation_forest.py`
  - `regression/src/main_xgboost_catboost.py`
  - `regression/src/main_lstm.py`

## Task 1: Anomaly Detection

The anomaly detection pipeline implements feature extraction including:
- Statistical features
- Datetime features

The model uses Isolation Forest for detection, with data splits determined through preliminary data analysis.

### Results

Performance metrics:
- True Positive: 102177
- False Positive: 22686
- True Negative: 54921
- False Negative: 36017

Model evaluation:
- Precision: 0.818
- Recall: 0.739
- F1 Score: 0.777

## Task 2: Multi-step Regression

The regression pipeline implements:
- Statistical feature extraction
- Datetime feature extraction
- 5-month forecast horizon
- Test set: Data after 2020

### Results

#### XGBoost Performance
- MAE: 10.00
- RMSE: 12.11
- MAPE: 23.08%

#### CatBoost Performance
- MAE: 7.49
- RMSE: 10.11
- MAPE: 17.37%

#### LSTM Performance
- MAE: 15.83
- RMSE: 18.87
- MAPE: 31.48%

The results indicate that CatBoost achieves the best performance across all evaluation metrics for the multi-step regression task.
