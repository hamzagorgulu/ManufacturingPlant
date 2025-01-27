import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class AnomalyEvaluator:
    """Class for evaluating anomaly detection models."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict:
        """
        Calculate various performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            scores: Anomaly scores
            
        Returns:
            Dict containing various metrics
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Calculate AUROC
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = auc(fpr, tpr)
        
        # Calculate AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
        auprc = auc(recall_curve, precision_curve)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auroc': auroc,
            'auprc': auprc
        }
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, scores: Dict[str, np.ndarray], 
                       save_path: str = None) -> None:
        """
        Plot ROC curves for multiple models.
        
        Args:
            y_true: True labels
            scores: Dictionary of model names and their anomaly scores
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        for model_name, model_scores in scores.items():
            fpr, tpr, _ = roc_curve(y_true, model_scores)
            auroc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUROC = {auroc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, scores: Dict[str, np.ndarray],
                                  save_path: str = None) -> None:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            y_true: True labels
            scores: Dictionary of model names and their anomaly scores
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        for model_name, model_scores in scores.items():
            precision, recall, _ = precision_recall_curve(y_true, model_scores)
            auprc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{model_name} (AUPRC = {auprc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_anomaly_scores(timestamps: np.ndarray, scores: Dict[str, np.ndarray],
                           thresholds: Dict[str, float], failure_time: pd.Timestamp,
                           save_path: str = None) -> None:
        """
        Plot anomaly scores over time with thresholds.
        
        Args:
            timestamps: Time indices
            scores: Dictionary of model names and their anomaly scores
            thresholds: Dictionary of model names and their thresholds
            failure_time: Time of actual failure
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 6))
        
        for model_name, model_scores in scores.items():
            plt.plot(timestamps, model_scores, label=f'{model_name} scores')
            threshold = thresholds[model_name]
            plt.axhline(y=threshold, color='r', linestyle='--', 
                       label=f'{model_name} threshold')
        
        plt.axvline(x=failure_time, color='k', linestyle='--', 
                   label='Actual Failure')
        
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models' performance metrics.
        
        Args:
            results: Dictionary of model names and their metrics
            
        Returns:
            DataFrame with comparison results
        """
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(3)
        return comparison_df 