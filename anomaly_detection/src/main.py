import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from data.data_loader import AnomalyDataLoader
from utils.preprocessing import AnomalyPreprocessor
from utils.evaluation import AnomalyEvaluator
from models.isolation_forest import IsolationForestDetector
from models.autoencoder import AutoencoderDetector
from typing import Dict


def setup_experiment_dir() -> Path:
    """Create and return experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def train_and_evaluate_model(model, X_train, X_test, y_test, model_name: str,
                           exp_dir: Path) -> Dict:
    """Train model and evaluate its performance."""
    print(f"\nTraining {model_name}...")
    model.fit(X_train)
    
    # Get predictions and scores
    predictions = model.predict(X_test)
    scores = model.score_samples(X_test)
    
    # Calculate metrics
    metrics = AnomalyEvaluator.calculate_metrics(y_test, predictions, scores)
    print(f"{model_name} metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return {
        'predictions': predictions,
        'scores': scores,
        'metrics': metrics
    }

def main():
    # Setup experiment directory
    exp_dir = setup_experiment_dir()
    print(f"Experiment directory: {exp_dir}")
    
    # Load data
    data_loader = AnomalyDataLoader('data')
    datasets = data_loader.load_all_data()
    
    # Process each dataset
    all_results = {}
    for dataset_name, data in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        breakpoint()
        # Split data
        splits = data_loader.split_data(data)
        
        # Preprocess data
        preprocessor = AnomalyPreprocessor()
        X_train = preprocessor.create_features(splits['train'])
        X_val = preprocessor.create_features(splits['val'])
        X_test = preprocessor.create_features(splits['test'])
        
        # Prepare labels (based on time_to_failure)
        y_test = (splits['test']['time_to_failure'] <= 24).astype(int)  # 24 hours threshold
        
        # Scale features
        X_train_scaled = preprocessor.prepare_data(X_train)
        X_val_scaled = preprocessor.prepare_data(X_val)
        X_test_scaled = preprocessor.prepare_data(X_test)
        
        # Initialize models
        models = {
            'IsolationForest': IsolationForestDetector(contamination=0.1),
            'Autoencoder': AutoencoderDetector(
                input_dim=X_train_scaled.shape[1],
                encoding_dims=[64, 32, 16]
            )
        }
        
        # Train and evaluate models
        dataset_results = {}
        scores_dict = {}
        thresholds_dict = {}
        
        for model_name, model in models.items():
            results = train_and_evaluate_model(
                model, X_train_scaled, X_test_scaled, y_test,
                model_name, exp_dir
            )
            dataset_results[model_name] = results
            scores_dict[model_name] = results['scores']
            thresholds_dict[model_name] = model.threshold
        
        # Plot evaluation curves
        AnomalyEvaluator.plot_roc_curve(
            y_test, scores_dict,
            save_path=exp_dir / f"{dataset_name}_roc_curve.png"
        )
        
        AnomalyEvaluator.plot_precision_recall_curve(
            y_test, scores_dict,
            save_path=exp_dir / f"{dataset_name}_pr_curve.png"
        )
        
        AnomalyEvaluator.plot_anomaly_scores(
            splits['test'].index, scores_dict, thresholds_dict,
            data_loader.failure_dates[dataset_name],
            save_path=exp_dir / f"{dataset_name}_scores.png"
        )
        
        # Compare models
        metrics_comparison = AnomalyEvaluator.compare_models(
            {name: results['metrics'] for name, results in dataset_results.items()}
        )
        metrics_comparison.to_csv(exp_dir / f"{dataset_name}_metrics.csv")
        
        all_results[dataset_name] = dataset_results
    
    print("\nExperiment completed. Results saved in:", exp_dir)

if __name__ == "__main__":
    main() 