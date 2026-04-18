"""
Checkpoint Management Module
Handles saving and loading of model checkpoints with full training state
"""

import os
import torch
import json
from typing import Dict, Optional, Any
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints during training
    Supports saving/loading model weights, optimizer state, and training configuration
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track saved checkpoints
        self.checkpoint_history = []
        self._load_history()
    
    def _load_history(self):
        """Load checkpoint history from file"""
        history_file = os.path.join(self.checkpoint_dir, "checkpoint_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.checkpoint_history = json.load(f)
    
    def _save_history(self):
        """Save checkpoint history to file"""
        history_file = os.path.join(self.checkpoint_dir, "checkpoint_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       config: Dict[str, Any],
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       extra_state: Optional[Dict] = None,
                       is_best: bool = False) -> str:
        """
        Save a checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Current metrics
            config: Model configuration
            scheduler: Learning rate scheduler (optional)
            extra_state: Additional state to save (optional)
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add extra state if provided
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Generate checkpoint filename
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        else:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_epoch_{epoch:04d}.pth"
            )
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update history (only for regular checkpoints, not best)
        if not is_best:
            self.checkpoint_history.append({
                'epoch': epoch,
                'path': checkpoint_path,
                'metrics': metrics,
                'timestamp': checkpoint['timestamp']
            })
            
            # Remove old checkpoints if limit exceeded
            if self.max_checkpoints > 0 and len(self.checkpoint_history) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_history.pop(0)
                if os.path.exists(old_checkpoint['path']):
                    os.remove(old_checkpoint['path'])
            
            self._save_history()
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       device: str = 'cuda') -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to
        
        Returns:
            Dictionary containing checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
            'extra_state': checkpoint.get('extra_state', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
    
    def load_best_checkpoint(self,
                            model: torch.nn.Module,
                            optimizer: Optional[torch.optim.Optimizer] = None,
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            device: str = 'cuda') -> Dict[str, Any]:
        """
        Load the best checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to
        
        Returns:
            Dictionary containing checkpoint information
        """
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        return self.load_checkpoint(best_checkpoint_path, model, optimizer, scheduler, device)
    
    def load_latest_checkpoint(self,
                              model: torch.nn.Module,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                              device: str = 'cuda') -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to
        
        Returns:
            Dictionary containing checkpoint information, or None if no checkpoints exist
        """
        if len(self.checkpoint_history) == 0:
            return None
        
        latest_checkpoint = self.checkpoint_history[-1]
        return self.load_checkpoint(
            latest_checkpoint['path'], 
            model, 
            optimizer, 
            scheduler, 
            device
        )
    
    def get_checkpoint_list(self) -> list:
        """Get list of all saved checkpoints"""
        return self.checkpoint_history.copy()
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Remove old checkpoints, keeping only the last N
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        while len(self.checkpoint_history) > keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint['path']):
                os.remove(old_checkpoint['path'])
        
        self._save_history()


def save_model_for_inference(model: torch.nn.Module, 
                             config: Dict[str, Any],
                             save_path: str,
                             metadata: Optional[Dict] = None):
    """
    Save model in a format optimized for inference
    
    Args:
        model: Trained model
        config: Model configuration
        save_path: Path to save the model
        metadata: Additional metadata (optional)
    """
    # Prepare save data
    save_data = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if metadata is not None:
        save_data['metadata'] = metadata
    
    # Save
    torch.save(save_data, save_path)
    
    # Also save config as JSON for easy inspection
    config_path = save_path.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def migrate_checkpoint_state_dict(state_dict):
    """Migrate old checkpoint keys for backward compatibility.
    
    Renames:
    - lrr_encoder.* -> sparse_edge_attention_encoder.*
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('lrr_encoder.', 'sparse_edge_attention_encoder.')
        new_state_dict[new_key] = value
    return new_state_dict


def load_model_for_inference(model: torch.nn.Module,
                             checkpoint_path: str,
                             device: str = 'cuda') -> Dict[str, Any]:
    """
    Load model for inference
    
    Args:
        model: Model instance (with correct architecture)
        checkpoint_path: Path to checkpoint
        device: Device to load to
    
    Returns:
        Dictionary with config and metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights (with migration for backward compatibility)
    state_dict = checkpoint['model_state_dict']
    state_dict = migrate_checkpoint_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    
    return {
        'config': checkpoint.get('config', {}),
        'metadata': checkpoint.get('metadata', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }

