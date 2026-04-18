"""
Comprehensive Metrics Module for PPI Prediction
Includes: Accuracy, Precision, Recall, F1-score, Confusion Matrix, AUC-ROC, AUPR
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate and track various metrics for PPI prediction"""
    
    def __init__(self, num_classes: int = 7, threshold: float = 0.5, logger=None):
        """
        Args:
            num_classes: Number of interaction classes
            threshold: Threshold for binary classification
            logger: Logger instance for logging debug messages
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.logger = logger
    
    def calculate_all_metrics(self, predictions: torch.Tensor, 
                              labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            predictions: Model predictions (logits or probabilities)
            labels: Ground truth labels
        
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Find and remove dimensions that have only one class (all 0s or all 1s)
        valid_dims = []
        removed_dims = []
        for i in range(labels.shape[1]):
            unique_labels = np.unique(labels[:, i])
            if len(unique_labels) > 1:  # Has both 0 and 1
                valid_dims.append(i)
            else:
                removed_dims.append(i)
                if self.logger:
                    self.logger.log(f"类别 {i} 在数据集中只有一种标签 ({unique_labels[0]})，去除此维度")
        
        if removed_dims:
            if self.logger:
                self.logger.log(f"去除的维度: {removed_dims}, 保留的有效维度: {valid_dims}")
            # Filter labels and predictions to keep only valid dimensions
            labels = labels[:, valid_dims]
            predictions = predictions[:, valid_dims]
        
        # Update num_classes to reflect valid dimensions
        original_num_classes = self.num_classes
        self.num_classes = len(valid_dims)
        
        # Apply sigmoid if needed (for logits) - use PyTorch's stable sigmoid
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            # Use PyTorch sigmoid for numerical stability
            if isinstance(predictions, torch.Tensor):
                probs = torch.sigmoid(predictions).numpy()
            else:
                probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        else:
            probs = predictions
        
        # Binary predictions
        preds_binary = (probs > self.threshold).astype(int)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = self._calculate_accuracy(preds_binary, labels)
        metrics['precision_micro'] = self._calculate_precision(preds_binary, labels, average='micro')
        metrics['precision_macro'] = self._calculate_precision(preds_binary, labels, average='macro')
        metrics['recall_micro'] = self._calculate_recall(preds_binary, labels, average='micro')
        metrics['recall_macro'] = self._calculate_recall(preds_binary, labels, average='macro')
        metrics['f1_micro'] = self._calculate_f1(preds_binary, labels, average='micro')
        metrics['f1_macro'] = self._calculate_f1(preds_binary, labels, average='macro')
        
        # AUC metrics
        try:
            metrics['auc_roc_micro'] = self._calculate_auc_roc(probs, labels, average='micro')
            metrics['auc_roc_macro'] = self._calculate_auc_roc(probs, labels, average='macro')
        except Exception as e:
            if self.logger:
                self.logger.log(f"Unexpected error calculating AUC metrics: {str(e)}")
            metrics['auc_roc_micro'] = 0.0
            metrics['auc_roc_macro'] = 0.0
        
        # AUPR (Average Precision)
        try:
            metrics['aupr_micro'] = self._calculate_aupr(probs, labels, average='micro')
            metrics['aupr_macro'] = self._calculate_aupr(probs, labels, average='macro')
        except Exception as e:
            if self.logger:
                self.logger.log(f"Unexpected error calculating AUPR metrics: {str(e)}")
            metrics['aupr_micro'] = 0.0
            metrics['aupr_macro'] = 0.0
        
        # Per-class metrics
        for i in range(self.num_classes):
            unique_labels = np.unique(labels[:, i])
            if len(unique_labels) > 1:  # Has both positive and negative samples
                metrics[f'f1_class_{i}'] = f1_score(labels[:, i], preds_binary[:, i], zero_division=0)
            else:
                if self.logger:
                    self.logger.log(f"类别 {i} 在数据集中只有一种标签 ({unique_labels[0]})，跳过F1计算")
        
        # Restore original num_classes
        self.num_classes = original_num_classes
        
        return metrics
    
    def _calculate_accuracy(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Calculate accuracy"""
        return accuracy_score(labels.flatten(), preds.flatten())
    
    def _calculate_precision(self, preds: np.ndarray, labels: np.ndarray, 
                            average: str = 'micro') -> float:
        """Calculate precision"""
        return precision_score(labels, preds, average=average, zero_division=0)
    
    def _calculate_recall(self, preds: np.ndarray, labels: np.ndarray, 
                         average: str = 'micro') -> float:
        """Calculate recall"""
        return recall_score(labels, preds, average=average, zero_division=0)
    
    def _calculate_f1(self, preds: np.ndarray, labels: np.ndarray, 
                     average: str = 'micro') -> float:
        """Calculate F1 score"""
        return f1_score(labels, preds, average=average, zero_division=0)
    
    def _calculate_auc_roc(self, probs: np.ndarray, labels: np.ndarray, 
                          average: str = 'micro') -> float:
        """Calculate AUC-ROC"""
        try:
            return roc_auc_score(labels, probs, average=average)
        except ValueError as e:
            if self.logger:
                self.logger.log(f"AUC-ROC计算错误 ({average}): {str(e)}")
            return 0.0
    
    def _calculate_aupr(self, probs: np.ndarray, labels: np.ndarray, 
                       average: str = 'micro') -> float:
        """Calculate AUPR (Average Precision)"""
        try:
            return average_precision_score(labels, probs, average=average)
        except ValueError as e:
            if self.logger:
                self.logger.log(f"AUPR计算错误 ({average}): {str(e)}")
            return 0.0
    
    def calculate_confusion_matrix(self, predictions: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   class_idx: int = 0) -> np.ndarray:
        """
        Calculate confusion matrix for a specific class
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            class_idx: Index of class to compute confusion matrix for
        
        Returns:
            Confusion matrix
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Apply sigmoid and threshold - use PyTorch's stable sigmoid
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            # Use PyTorch sigmoid for numerical stability
            if isinstance(predictions, torch.Tensor):
                probs = torch.sigmoid(predictions).numpy()
            else:
                probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        else:
            probs = predictions
        
        preds_binary = (probs[:, class_idx] > self.threshold).astype(int)
        true_labels = labels[:, class_idx].astype(int)
        
        return confusion_matrix(true_labels, preds_binary)
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_name: str = "Class", 
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_name: Name of the class
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, predictions: torch.Tensor, labels: torch.Tensor,
                      class_idx: int = 0, save_path: Optional[str] = None):
        """
        Plot ROC curve for a specific class
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            class_idx: Index of class
            save_path: Path to save figure
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Apply sigmoid - use stable sigmoid
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            # Stable sigmoid implementation to avoid overflow
            probs = np.where(predictions >= 0,
                           1 / (1 + np.exp(-predictions)),
                           np.exp(predictions) / (1 + np.exp(predictions)))
        else:
            probs = predictions
        
        fpr, tpr, _ = roc_curve(labels[:, class_idx], probs[:, class_idx])
        auc = roc_auc_score(labels[:, class_idx], probs[:, class_idx])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {class_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curve(self, predictions: torch.Tensor, labels: torch.Tensor,
                     class_idx: int = 0, save_path: Optional[str] = None):
        """
        Plot Precision-Recall curve for a specific class
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            class_idx: Index of class
            save_path: Path to save figure
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Apply sigmoid - use stable sigmoid
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            # Stable sigmoid implementation to avoid overflow
            probs = np.where(predictions >= 0,
                           1 / (1 + np.exp(-predictions)),
                           np.exp(predictions) / (1 + np.exp(predictions)))
        else:
            probs = predictions
        
        precision, recall, _ = precision_recall_curve(labels[:, class_idx], probs[:, class_idx])
        ap = average_precision_score(labels[:, class_idx], probs[:, class_idx])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Class {class_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def format_metrics_string(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dictionary as a readable string
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for the string (e.g., "Train", "Val", "Test")
    
    Returns:
        Formatted string
    """
    lines = []
    if prefix:
        lines.append(f"=== {prefix} Metrics ===")
    
    # Main metrics
    main_metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 
                   'auc_roc_micro', 'aupr_micro']
    
    for key in main_metrics:
        if key in metrics:
            lines.append(f"{key}: {metrics[key]:.4f}")
    
    # Macro metrics
    lines.append("\nMacro Averages:")
    macro_metrics = ['precision_macro', 'recall_macro', 'f1_macro', 
                    'auc_roc_macro', 'aupr_macro']
    
    for key in macro_metrics:
        if key in metrics:
            lines.append(f"{key}: {metrics[key]:.4f}")
    
    return "\n".join(lines)

