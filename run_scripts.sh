#!/bin/bash

# Run the scripts sequentially
python anomaly_detection/src/main_isolation_forest.py
python regression/src/main_xgboost_catboost.py
python regression/src/main_lstm.py