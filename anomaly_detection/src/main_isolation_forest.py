import pandas as pd
import sys
import numpy as np

from models.isolation_forest import IsolationForestDetector
from models.autoencoder import AutoencoderDetector
from utils.preprocessing import AnomalyPreprocessor
from data.data_loader import AnomalyDataLoader
from utils.evaluation import AnomalyEvaluator

sys.path.append('../src/models')

def main():

    dataloader = AnomalyDataLoader('../data')
    dataset_dict = dataloader.load_all_data()
    df1 = dataset_dict["dataset1"]
    df2 = dataset_dict["dataset2"]

    # Preprocess data
    preprocessor = AnomalyPreprocessor()
    df1 = preprocessor.extract_datetime_features(df1)
    df1 = preprocessor.extract_statistical_features(df1)

    df2 = preprocessor.extract_datetime_features(df2)
    df2 = preprocessor.extract_statistical_features(df2)

    # specify split dates
    df1_train_end_date = pd.to_datetime("2021-01-11")
    df1_test_normal_end_date = pd.to_datetime("2021-01-13")

    df2_train_end_date = pd.to_datetime("2021-01-26")
    df2_test_normal_end_date = pd.to_datetime("2021-01-28")

    # split data
    df1_data_split_dict = dataloader.split_data(df1, df1_train_end_date, df1_test_normal_end_date)
    df2_data_split_dict = dataloader.split_data(df2, df2_train_end_date, df2_test_normal_end_date)

    df1_train_data = df1_data_split_dict['train']
    df1_test_normal_data = df1_data_split_dict['test_normal']
    df1_test_anomaly_data = df1_data_split_dict['test_anomaly']

    df2_train_data = df2_data_split_dict['train']
    df2_test_normal_data = df2_data_split_dict['test_normal']
    df2_test_anomaly_data = df2_data_split_dict['test_anomaly']

    # concatenate datasets
    train_data = pd.concat([df1_train_data, df2_train_data]).drop(columns=["date"])
    test_normal_data = pd.concat([df1_test_normal_data, df2_test_normal_data]).drop(columns=["date"])
    test_anomaly_data = pd.concat([df1_test_anomaly_data, df2_test_anomaly_data]).drop(columns=["date"])

    # scale features
    train_data = preprocessor.prepare_data(train_data, scale=True, fit_scaler=True)
    test_normal_data = preprocessor.prepare_data(test_normal_data)
    test_anomaly_data = preprocessor.prepare_data(test_anomaly_data)

    # train normal set on isolation forest
    params = {
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": "auto", # 0.15
        "max_features": 1.0,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 1,
    }
    clf = IsolationForestDetector(**params)
    clf.fit(train_data)

    # predict on test normal set
    test_normal_scores = clf.score_samples(test_normal_data)
    test_normal_predictions = clf.predict(test_normal_data)

    # predict on test anomaly set
    test_anomaly_scores = clf.score_samples(test_anomaly_data)
    test_anomaly_predictions = clf.predict(test_anomaly_data)

    # evaluate model
    y_true = np.concatenate([np.zeros(len(test_normal_data)), np.ones(len(test_anomaly_data))])
    y_pred = np.concatenate([test_normal_predictions, test_anomaly_predictions])
    scores = np.concatenate([test_normal_scores, test_anomaly_scores])

    evaluator = AnomalyEvaluator()
    metrics = evaluator.calculate_metrics(y_true, y_pred, scores)

    print(f"True Positive: {metrics['true_positive']}")
    print(f"False Positive: {metrics['false_positive']}")
    print(f"True Negative: {metrics['true_negative']}")
    print(f"False Negative: {metrics['false_negative']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")

if __name__ == "__main__":
    main()
