
## Task 1 Structure
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


## Task 2 Structure
regression/
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
└── requirements.txt# ManufacturingPlant

## Task 1
I implemented a pipeline where I extract both statistical and datetime features from the existin


Anomaly Detection Results
True Positive: 102177
False Positive: 22686
True Negative: 54921
False Negative: 36017
Precision: 0.8183128709065135
Recall: 0.7393736341664616
F1 Score: 0.7768430416221579

Regression Results

XGBoost MAE: 10.00
XGBoost RMSE: 12.11
XGBoost MAPE(%): 23.08%

CatBoost MAE: 7.49
CatBoost RMSE: 10.11
CatBoost MAPE(%): 17.37%

LSTM MAE: 15.83
LSTM RMSE: 18.87
LSTM MAPE(%): 31.48%