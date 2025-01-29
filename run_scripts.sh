#!/bin/bash

# Exit immediately if any command fails
set -e

# Define log directory
LOG_DIR="/app/logs"
mkdir -p $LOG_DIR

# Run the scripts sequentially and log output
echo "Running anomaly_detection/src/main_isolation_forest.py..."
python anomaly_detection/src/main_isolation_forest.py 2>&1 | tee $LOG_DIR/anomaly_detection.log

echo "Running regression/src/main_xgboost_catboost.py..."
python regression/src/main_xgboost_catboost.py 2>&1 | tee $LOG_DIR/xgboost_catboost.log

echo "Running regression/src/main_lstm.py..."
python regression/src/main_lstm.py 2>&1 | tee $LOG_DIR/lstm.log

echo "All scripts executed successfully. Logs are available in $LOG_DIR."