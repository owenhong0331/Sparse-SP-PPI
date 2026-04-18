"""
Logging Module for Training
Supports both TensorBoard and text file logging
"""

import os
import json
import threading
from datetime import datetime
from typing import Dict, Optional, Any
from contextlib import contextmanager


class TrainingLogger:
    """
    Comprehensive logger for training process
    Logs to both TensorBoard and text files
    """
    
    def __init__(self, log_dir: str, experiment_name: str, config: Dict, use_tensorboard: bool = True):
        """
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of the experiment
            config: Configuration dictionary
            use_tensorboard: Whether to use TensorBoard (default True)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.config = config
        self.use_tensorboard = use_tensorboard
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize TensorBoard writer if requested
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.run_dir)
                # Add lock for thread-safe operations
                self.tb_lock = threading.Lock()
            except ImportError:
                print("TensorBoard not available, falling back to text-only logging")
                self.tb_writer = None
                self.tb_lock = None
                self.use_tensorboard = False
        else:
            self.tb_writer = None
            self.tb_lock = None
        
        # Initialize text log file
        self.log_file = os.path.join(self.run_dir, "training.log")
        
        # Save configuration
        self._save_config()
        
        # Log start
        self.log("=" * 80)
        self.log(f"Training started: {timestamp}")
        self.log(f"Experiment: {experiment_name}")
        self.log(f"TensorBoard enabled: {self.use_tensorboard}")
        self.log("=" * 80)
    
    def _save_config(self):
        """Save configuration to JSON file"""
        config_file = os.path.join(self.run_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log(self, message: str, print_console: bool = True):
        """
        Log message to file and optionally print to console
        
        Args:
            message: Message to log
            print_console: Whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        # Print to console with flush to ensure complete output
        if print_console:
            print(log_message, flush=True)
    
    @contextmanager
    def _tb_operation(self):
        """Context manager for safe TensorBoard operations"""
        if self.use_tensorboard and self.tb_writer and self.tb_lock:
            with self.tb_lock:
                yield self.tb_writer
        else:
            yield None
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to TensorBoard and text file
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            prefix: Prefix for metric names (e.g., "train", "val", "test")
        """
        # Log to TensorBoard safely
        if self.use_tensorboard:
            with self._tb_operation() as tb_writer:
                if tb_writer:
                    for key, value in metrics.items():
                        try:
                            tag = f"{prefix}/{key}" if prefix else key
                            tb_writer.add_scalar(tag, float(value), step)
                        except Exception as e:
                            self.log(f"Warning: Could not log metric {tag}: {str(e)}")
        
        # Build metrics string and log once to avoid truncation
        metrics_str = f"\n{prefix.upper()} Metrics at step {step}:\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_str += f"  {key}: {value:.6f}\n"
            else:
                metrics_str += f"  {key}: {value}\n"
        
        # Remove trailing newline and log
        self.log(metrics_str.rstrip())
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Optional[Dict[str, float]] = None,
                  test_metrics: Optional[Dict[str, float]] = None):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            test_metrics: Test metrics (optional)
        """
        self.log("\n" + "=" * 80)
        self.log(f"EPOCH {epoch}")
        self.log("=" * 80)
        
        # Log training metrics
        self.log_metrics(train_metrics, epoch, prefix="train")
        
        # Log validation metrics
        if val_metrics:
            self.log_metrics(val_metrics, epoch, prefix="val")
        
        # Log test metrics
        if test_metrics:
            self.log_metrics(test_metrics, epoch, prefix="test")
    
    def log_confusion_matrix(self, cm, class_name: str, step: int, prefix: str = ""):
        """
        Log confusion matrix to TensorBoard
        
        Args:
            cm: Confusion matrix (2x2 numpy array)
            class_name: Name of the class
            step: Current step/epoch
            prefix: Prefix for tag
        """
        if not self.use_tensorboard:
            return  # Skip if TensorBoard is disabled
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(f'Confusion Matrix - {class_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
            # Log to TensorBoard safely
            with self._tb_operation() as tb_writer:
                if tb_writer:
                    tag = f"{prefix}/confusion_matrix_{class_name}" if prefix else f"confusion_matrix_{class_name}"
                    tb_writer.add_figure(tag, fig, step)
            plt.close(fig)
        except Exception as e:
            self.log(f"Warning: Could not log confusion matrix: {str(e)}")
    
    def log_model_graph(self, model, input_data):
        """
        Log model graph to TensorBoard
        
        Args:
            model: PyTorch model
            input_data: Sample input data
        """
        if not self.use_tensorboard:
            return  # Skip if TensorBoard is disabled
        
        try:
            with self._tb_operation() as tb_writer:
                if tb_writer:
                    tb_writer.add_graph(model, input_data)
        except Exception as e:
            self.log(f"Warning: Could not log model graph: {str(e)}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters and final metrics
        
        Args:
            hparams: Hyperparameters dictionary
            metrics: Final metrics dictionary
        """
        if not self.use_tensorboard:
            return  # Skip if TensorBoard is disabled
        
        try:
            with self._tb_operation() as tb_writer:
                if tb_writer:
                    tb_writer.add_hparams(hparams, metrics)
        except Exception as e:
            self.log(f"Warning: Could not log hyperparameters: {str(e)}")
    
    def save_checkpoint_info(self, epoch: int, metrics: Dict[str, float], 
                            checkpoint_path: str):
        """
        Save checkpoint information
        
        Args:
            epoch: Epoch number
            metrics: Metrics at checkpoint
            checkpoint_path: Path to checkpoint file
        """
        info = {
            'epoch': epoch,
            'metrics': metrics,
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        checkpoint_log = os.path.join(self.run_dir, "checkpoints.json")
        
        # Load existing checkpoints
        if os.path.exists(checkpoint_log):
            with open(checkpoint_log, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []
        
        # Add new checkpoint
        checkpoints.append(info)
        
        # Save
        with open(checkpoint_log, 'w') as f:
            json.dump(checkpoints, f, indent=2)
        
        self.log(f"Checkpoint saved: epoch {epoch}, path: {checkpoint_path}")
    
    def log_best_model(self, epoch: int, metrics: Dict[str, float], 
                      model_path: str, metric_name: str = "val_f1_micro"):
        """
        Log best model information
        
        Args:
            epoch: Epoch number
            metrics: Metrics of best model
            model_path: Path to best model
            metric_name: Name of metric used for selection
        """
        best_info = {
            'epoch': epoch,
            'metrics': metrics,
            'model_path': model_path,
            'selection_metric': metric_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        best_model_file = os.path.join(self.run_dir, "best_model.json")
        with open(best_model_file, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        self.log("\n" + "!" * 80)
        self.log(f"NEW BEST MODEL at epoch {epoch}")
        self.log(f"Metric: {metric_name} = {metrics.get(metric_name, 'N/A')}")
        self.log(f"Model saved to: {model_path}")
        self.log("!" * 80 + "\n")
    
    def log_training_summary(self, total_epochs: int, best_epoch: int, 
                           best_metrics: Dict[str, float], total_time: float, model=None):
        """
        Log training summary at the end
        
        Args:
            total_epochs: Total number of epochs trained
            best_epoch: Epoch with best performance
            best_metrics: Best metrics achieved
            total_time: Total training time in seconds
            model: Optional model to log encoder weights
        """
        self.log("\n" + "=" * 80)
        self.log("TRAINING SUMMARY")
        self.log("=" * 80)
        self.log(f"Total epochs: {total_epochs}")
        self.log(f"Best epoch: {best_epoch}")
        self.log(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        
        # Log encoder weights if model provided
        if model is not None and hasattr(model, 'protein_encoder'):
            encoder = model.protein_encoder
            if hasattr(encoder, 'get_weights'):
                self.log("\nEncoder Weights:")
                weights = encoder.get_weights()
                for etype, weight in weights.items():
                    self.log(f"  {etype}: {weight:.6f}")
        
        self.log("\nBest Metrics:")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.6f}")
            else:
                self.log(f"  {key}: {value}")
        self.log("=" * 80)
    
    def close(self):
        """Close logger and TensorBoard writer"""
        if self.use_tensorboard and self.tb_writer:
            try:
                with self._tb_operation() as tb_writer:
                    if tb_writer:
                        tb_writer.flush()  # Ensure all pending writes are completed
                        tb_writer.close()
            except Exception as e:
                self.log(f"Warning: Error closing TensorBoard writer: {str(e)}")
        self.log("\nLogger closed.")
    
    def get_run_dir(self) -> str:
        """Get the run directory path"""
        return self.run_dir


class SimpleLogger:
    """Simple logger for inference/testing without TensorBoard"""
    
    def __init__(self, log_file: str):
        """
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Log start
        self.log("=" * 80)
        self.log(f"Logging started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 80)
    
    def log(self, message: str, print_console: bool = True):
        """
        Log message to file and optionally print to console
        
        Args:
            message: Message to log
            print_console: Whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        # Print to console with flush to ensure complete output
        if print_console:
            print(log_message, flush=True)