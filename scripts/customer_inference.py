"""
Customer PPI Inference Script - 更新版本
支持异质性模型和动态类别映射的蛋白质-蛋白质互作预测
"""

import os
import sys

# Set environment variables to avoid GIL issues with DGL/PyTorch
os.environ['DGL_BACKEND'] = 'pytorch'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json
import numpy as np
import pandas as pd
import torch
import math
import random

# Disable PyTorch internal threading to avoid GIL issues with DGL batching
torch.set_num_threads(1)
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.integrated_high_ppi_simple import ProteinGINModelSimple, ExplainableProteinGINModel
from models.customer_dataloader import PPIDataset, collate_protein_graphs
from models.checkpoint import load_model_for_inference
from models.logger import SimpleLogger
from models.metrics import MetricsCalculator, format_metrics_string
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clear GPU cache if using CUDA
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class PPIPredictor:
    """
    Protein-Protein Interaction Predictor
    """
    
    def __init__(self, checkpoint_path: str, explainable: bool = False, config: dict = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            explainable: Whether to use explainable model
            config: Optional config dict (if provided, overrides checkpoint config)
        """
        self.checkpoint_path = checkpoint_path
        self.explainable = explainable
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                # New format: {'model_state_dict': ..., 'config': ..., ...}
                self.config = checkpoint['config']
                print(f"Loaded checkpoint (new format) from: {checkpoint_path}")
            elif 'model_state_dict' in checkpoint:
                # Format with model_state_dict but no config wrapper
                self.config = checkpoint.get('config', {})
                print(f"Loaded checkpoint (legacy format) from: {checkpoint_path}")
            else:
                # Old format: just state dict
                self.config = {}
                print(f"Loaded checkpoint (old format - state dict only) from: {checkpoint_path}")
                # Store for later use in loading
                self._checkpoint_state_dict = checkpoint
        else:
            # Very old format: just the state dict
            self.config = {}
            print(f"Loaded checkpoint (raw state dict) from: {checkpoint_path}")
            self._checkpoint_state_dict = checkpoint
        
        # Override with provided config if available
        if config is not None:
            print("Using provided config (overriding checkpoint config)")
            self.config = config
        
        # Detect actual output dimension from checkpoint
        # This is critical for compatibility with different class numbers
        actual_output_dim = None
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Look for interaction_head output layer (interaction_head.3.weight or interaction_head.3.bias)
            # This is the final Linear layer that outputs class predictions
            for key in state_dict.keys():
                if 'interaction_head.3' in key and ('.weight' in key or '.bias' in key):
                    shape = state_dict[key].shape
                    if len(shape) >= 1:
                        # For weight [output_dim, hidden_dim] or bias [output_dim]
                        actual_output_dim = shape[0]
                        print(f"Detected actual output dimension from checkpoint key '{key}': {actual_output_dim}")
                        break
        
        # If we detected a different output_dim, update config
        if actual_output_dim is not None:
            config_output_dim = self.config.get('output_dim', actual_output_dim)
            if actual_output_dim != config_output_dim:
                print(f"⚠️  Output dimension mismatch: checkpoint has {actual_output_dim}, config has {config_output_dim}")
                print(f"   Using checkpoint value: {actual_output_dim}")
                self.config['output_dim'] = actual_output_dim
        
        # Detect if we need heterogeneous model
        use_heterogeneous = (self.config.get('encoding', {}).get('peptide_encoder_enabled', False) or 
                           self.config.get('encoding', {}).get('lrr_encoder_enabled', False))
        
        print(f"Heterogeneous model required: {use_heterogeneous}")
        print(f"Peptide encoder enabled: {self.config.get('encoding', {}).get('peptide_encoder_enabled', False)}")
        print(f"LRR encoder enabled: {self.config.get('encoding', {}).get('lrr_encoder_enabled', False)}")
        
        # Initialize model with correct type
        # Use ProteinGINModelSimple for LRR models (matches training architecture)
        lrr_enabled = self.config.get('encoding', {}).get('lrr_encoder_enabled', False)
        
        if explainable:
            self.model = ExplainableProteinGINModel(self.config).to(device)
        else:
            self.model = ProteinGINModelSimple(self.config).to(device)
        
        # Load weights - handle different checkpoint formats
        if hasattr(self, '_checkpoint_state_dict'):
            # Old format: direct state dict
            print("Loading weights from old format checkpoint (state dict only)...")
            try:
                self.model.load_state_dict(self._checkpoint_state_dict)
            except RuntimeError as e:
                print(f"Error loading state dict directly: {e}")
                print("Attempting to load with strict=False...")
                self.model.load_state_dict(self._checkpoint_state_dict, strict=False)
            delattr(self, '_checkpoint_state_dict')
        else:
            # New format: use checkpoint loader with fallback to strict=False
            print("Loading weights from new format checkpoint...")
            try:
                load_model_for_inference(self.model, checkpoint_path, device)
            except RuntimeError as e:
                print(f"Error loading with strict=True: {e}")
                print("Attempting to load with strict=False (architecture mismatch)...")
                # Manual loading with strict=False
                checkpoint = torch.load(checkpoint_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✅ Loaded with strict=False - some weights may not match")
        
        self.model.eval()
        
        # Dynamic class names based on training config
        self.class_names = self._get_actual_class_names()
        self.num_classes = len(self.class_names)
        
        print(f"Using {self.num_classes} classes: {self.class_names}")
    
    def _get_actual_class_names(self):
        """Get actual class names from training configuration"""
        if not self.config:
            # Default config if none loaded from checkpoint
            return ['reaction', 'binding', 'ptmod', 'activation',
                   'inhibition', 'catalysis', 'expression']
        
        actual_class_map = self.config.get('actual_class_map', {})
        if actual_class_map:
            # Use actual class mapping from training
            return list(actual_class_map.keys())
        else:
            # Fallback to default class names
            return ['reaction', 'binding', 'ptmod', 'activation',
                   'inhibition', 'catalysis', 'expression']
    
    def encode_proteins(self, protein_dataset):
        """Encode all proteins with support for heterogeneous models"""
        # Check if protein dataset has graphs
        if not hasattr(protein_dataset, 'graphs') or len(protein_dataset.graphs) == 0:
            raise ValueError(f"Protein dataset has no graphs! Dataset length: {len(protein_dataset)}")

        print(f"Encoding {len(protein_dataset.graphs)} proteins...")

        # Check if we need heterogeneous encoding
        from models.customer_dataloader import is_heterogeneous_dataset, collate_heterogeneous_protein_graphs
        
        use_heterogeneous_encoding = is_heterogeneous_dataset(protein_dataset.graphs)
        
        if use_heterogeneous_encoding:
            print("检测到异质性数据集，使用三类型分类批处理")
            return self._encode_heterogeneous_proteins(protein_dataset)
        else:
            print("使用标准批处理")
            return self._encode_standard_proteins(protein_dataset)
    
    def _encode_standard_proteins(self, protein_dataset):
        """Standard protein encoding with batch processing"""
        # Check if model has encode_proteins method (ProteinGINModelSimple has this)
        if hasattr(self.model, 'encode_proteins'):
            print("Using model's encode_proteins method with built-in batching")
            return self.model.encode_proteins(protein_dataset.graphs)
        
        # Fallback to manual encoding for older models
        protein_loader = DataLoader(
            protein_dataset, 
            batch_size=32, 
            shuffle=False, 
            collate_fn=collate_protein_graphs,
            num_workers=0  # Disable multiprocessing to avoid GIL issues
        )

        self.model.eval()
        protein_embeds = []
        total_batches = len(protein_loader)

        print(f"  Encoding {total_batches} batches of proteins...")

        with torch.no_grad():
            for batch_idx, batch_graph in enumerate(protein_loader):
                # Move graph to device
                batch_graph = batch_graph.to(device)
                
                # Standard model encoding using protein_encoder
                _, graph_embed = self.model.protein_encoder(batch_graph)
                protein_embeds.append(graph_embed.cpu())

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"    Batch {batch_idx + 1}/{total_batches}: {graph_embed.shape}")

        if len(protein_embeds) == 0:
            raise ValueError("No protein embeddings generated! Check if DataLoader is working correctly.")

        print(f"  Generated {len(protein_embeds)} batches of embeddings")
        result = torch.cat(protein_embeds, dim=0).to(device)
        print(f"  Final embeddings shape: {result.shape}")

        return result
    
    def _encode_heterogeneous_proteins(self, protein_dataset):
        """Heterogeneous protein encoding (three-type classification)"""
        from models.customer_dataloader import collate_heterogeneous_protein_graphs
        
        print("开始异质性蛋白质编码...")
        
        # Manually process protein graphs using classified batching
        self.model.eval()
        protein_embeds = []
        
        with torch.no_grad():
            # Process all protein graphs at once using classified batching
            batch_results = collate_heterogeneous_protein_graphs(protein_dataset.graphs)
            
            if not batch_results:
                raise ValueError("分类批处理返回空结果")
            
            print(f"分类批处理结果: {list(batch_results.keys())}")
            
            # Check if model supports new encoding interface
            if hasattr(self.model, 'encode_proteins'):
                # Use new three-type encoding interface
                # First ensure all graphs are on the correct device
                for graph_type, batch_graph in batch_results.items():
                    if batch_graph is not None:
                        batch_graph = batch_graph.to(device)
                        if 'x' in batch_graph.ndata:
                            batch_graph.ndata['x'] = batch_graph.ndata['x'].to(device)
                        batch_results[graph_type] = batch_graph
                
                # Move model to device
                self.model = self.model.to(device)
                
                prot_embed = self.model.encode_proteins(batch_results)
                protein_embeds.append(prot_embed.cpu())
            elif hasattr(self.model, 'protein_encoder'):
                # Use standard encoding interface, process each graph type individually
                for graph_type, batch_graph in batch_results.items():
                    if batch_graph is not None:
                        print(f"  Processing {graph_type} graphs: {batch_graph.batch_size} graphs")
                        
                        # Move graph to device
                        batch_graph = batch_graph.to(device)
                        if 'x' in batch_graph.ndata:
                            batch_graph.ndata['x'] = batch_graph.ndata['x'].to(device)
                        
                        # Move model to device
                        self.model = self.model.to(device)
                        
                        # Use standard encoder to process
                        _, graph_embed = self.model.protein_encoder(batch_graph)
                        protein_embeds.append(graph_embed.cpu())
            else:
                raise ValueError("模型不支持蛋白质编码")
        
        if len(protein_embeds) == 0:
            raise ValueError("No protein embeddings generated!")
        
        # Merge all embeddings
        if len(protein_embeds) > 1:
            result = torch.cat(protein_embeds, dim=0).to(device)
        else:
            result = protein_embeds[0].to(device)
        
        print(f"  Final heterogeneous embeddings shape: {result.shape}")
        print(f"  Processed {result.shape[0]} protein embeddings")
        
        return result
    
    def predict_interactions(self, ppi_g, prot_embed, ppi_list, 
                            indices: List[int], threshold: float = 0.5) -> Dict:
        """
        Predict interactions for given protein pairs with batch processing
        
        Args:
            ppi_g: PPI graph
            prot_embed: Protein embeddings
            ppi_list: List of protein pairs
            indices: Indices of pairs to predict
            threshold: Classification threshold
        
        Returns:
            Dictionary with predictions and probabilities
        """
        self.model.eval()
        
        all_predictions = []
        all_logits = []
        
        # Process in batches to handle large inference sets
        batch_size = 50  # Reduced from 1000 to avoid GPU OOM
        num_batches = math.ceil(len(indices) / batch_size)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Get predictions for this batch
                logits = self.model(ppi_g, prot_embed, ppi_list, batch_indices)
                
                # Convert to probabilities
                probs = torch.sigmoid(logits)
                
                # Binary predictions
                preds = (probs > threshold).int()
                
                all_logits.append(logits.cpu())
                all_predictions.append({
                    'probabilities': probs.cpu().numpy(),
                    'predictions': preds.cpu().numpy()
                })
                
                if batch_idx == 0:
                    print(f"  First batch: {logits.shape} -> {probs.shape}")
        
        # Combine all batches
        if len(all_logits) > 0:
            combined_logits = torch.cat(all_logits, dim=0)
            
            # Combine probabilities and predictions
            combined_probs = np.concatenate([batch['probabilities'] for batch in all_predictions], axis=0)
            combined_preds = np.concatenate([batch['predictions'] for batch in all_predictions], axis=0)
            
            return {
                'logits': combined_logits.numpy(),
                'probabilities': combined_probs,
                'predictions': combined_preds
            }
        else:
            return {
                'logits': np.array([]),
                'probabilities': np.array([]),
                'predictions': np.array([])
            }
    
    def predict_single_pair(self, protein1_id: int, protein2_id: int,
                           ppi_g, prot_embed, threshold: float = 0.5) -> Dict:
        """
        Predict interaction for a single protein pair
        
        Args:
            protein1_id: ID of first protein
            protein2_id: ID of second protein
            ppi_g: PPI graph
            prot_embed: Protein embeddings
            threshold: Classification threshold
        
        Returns:
            Dictionary with prediction results
        """
        # Create temporary PPI list
        ppi_list = [[protein1_id, protein2_id]]
        
        # Predict
        results = self.predict_interactions(ppi_g, prot_embed, ppi_list, [0], threshold)
        
        # Check if results are valid
        if results['probabilities'].size == 0:
            raise ValueError("No prediction results obtained. Check protein IDs and model configuration.")
        
        # Format results with dynamic class names
        prediction_dict = {}
        for i, class_name in enumerate(self.class_names):
            if i < results['probabilities'].shape[1]:  # Ensure we don't exceed actual classes
                prediction_dict[class_name] = {
                    'probability': float(results['probabilities'][0, i]),
                    'prediction': bool(results['predictions'][0, i])
                }
            else:
                # For models with fewer classes than the default 7
                prediction_dict[class_name] = {
                    'probability': 0.0,
                    'prediction': False
                }
        
        return prediction_dict
    
    def batch_predict(self, protein_pairs: List[Tuple[int, int]], 
                     ppi_g, prot_embed, threshold: float = 0.5) -> pd.DataFrame:
        """
        Batch prediction for multiple protein pairs with dynamic class support
        
        Args:
            protein_pairs: List of (protein1_id, protein2_id) tuples
            ppi_g: PPI graph
            prot_embed: Protein embeddings
            threshold: Classification threshold
        
        Returns:
            DataFrame with predictions
        """
        # Predict
        results = self.predict_interactions(
            ppi_g, prot_embed, protein_pairs, 
            list(range(len(protein_pairs))), threshold
        )
        
        # Create DataFrame with dynamic class columns
        rows = []
        for i, (p1, p2) in enumerate(protein_pairs):
            row = {
                'protein1_id': p1,
                'protein2_id': p2
            }
            
            # Add predictions for each class (dynamic based on actual model output)
            for j, class_name in enumerate(self.class_names):
                if j < results['probabilities'].shape[1]:  # Check if class exists in model output
                    row[f'{class_name}_prob'] = results['probabilities'][i, j]
                    row[f'{class_name}_pred'] = results['predictions'][i, j]
                else:
                    # For models with fewer classes than the default 7
                    row[f'{class_name}_prob'] = 0.0
                    row[f'{class_name}_pred'] = 0
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_with_metrics(predictor, ppi_dataset, batch_size=1000, logger=None, split_indices=None):
    """
    Evaluate model with comprehensive metrics (consistent with customer_train.py)
    
    Args:
        predictor: PPIPredictor instance
        ppi_dataset: PPIDataset instance
        batch_size: Batch size for prediction
        logger: Logger instance
        split_indices: Optional indices to evaluate (if None, evaluate all)
    
    Returns:
        Dictionary with metrics, predictions, and labels
    """
    from collections import Counter
    
    # Initialize metrics calculator
    num_classes = predictor.num_classes
    metrics_calc = MetricsCalculator(num_classes=num_classes, logger=logger)
    
    # Get all indices or use provided split
    if split_indices is None:
        split_indices = list(range(len(ppi_dataset.ppi_list)))
    
    logger.log(f"Evaluating on {len(split_indices)} samples")
    
    # Encode proteins
    logger.log("Encoding proteins...")
    prot_embed = predictor.encode_proteins(ppi_dataset.protein_dataset)
    
    # Clear cache after encoding to free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.log(f"  GPU memory after encoding: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Get predictions
    logger.log("Running predictions...")
    results = predictor.predict_interactions(
        ppi_dataset.ppi_graph,
        prot_embed,
        ppi_dataset.ppi_list,
        split_indices,
        threshold=0.5
    )
    
    # Clear cache after predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get labels for the evaluated indices
    if hasattr(ppi_dataset, 'get_labels_tensor'):
        all_labels = ppi_dataset.get_labels_tensor()
        labels = all_labels[split_indices]
    else:
        logger.log("Warning: Cannot get labels from dataset")
        labels = None
    
    # Calculate metrics
    if labels is not None and len(labels) > 0:
        logger.log("Calculating metrics...")
        
        # Convert to tensors for metrics calculation
        logits_tensor = torch.from_numpy(results['logits'])
        labels_tensor = labels.cpu() if torch.is_tensor(labels) else torch.from_numpy(labels)
        
        metrics = metrics_calc.calculate_all_metrics(logits_tensor, labels_tensor)
        
        # Log metrics (consistent with customer_train.py format)
        logger.log("\n" + "="*70)
        logger.log("Evaluation Metrics")
        logger.log("="*70)
        
        # Main metrics
        main_metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 
                       'auc_roc_micro', 'aupr_micro']
        for key in main_metrics:
            if key in metrics:
                logger.log(f"{key}: {metrics[key]:.4f}")
        
        # Macro metrics
        logger.log("\nMacro Averages:")
        macro_metrics = ['precision_macro', 'recall_macro', 'f1_macro', 
                        'auc_roc_macro', 'aupr_macro']
        for key in macro_metrics:
            if key in metrics:
                logger.log(f"{key}: {metrics[key]:.4f}")
        
        # Per-class metrics
        logger.log("\nPer-class F1 Scores:")
        for key in sorted(metrics.keys()):
            if key.startswith('f1_class_'):
                class_idx = int(key.split('_')[-1])
                class_name = predictor.class_names[class_idx] if class_idx < len(predictor.class_names) else f"Class {class_idx}"
                logger.log(f"  {class_name}: {metrics[key]:.4f}")
        
        logger.log("="*70)
        
        # Dataset statistics (consistent with customer_train.py)
        logger.log("\nDataset Statistics:")
        logger.log("="*70)
        logger.log(f"Total samples: {len(split_indices)}")
        
        # Edge type distribution
        protein_graphs = ppi_dataset.protein_dataset.graphs
        total_graphs = len(protein_graphs)
        graph_with_edge_type = Counter()
        
        for graph in protein_graphs:
            if hasattr(graph, 'etypes'):
                etypes = graph.etypes
                if isinstance(etypes, (list, tuple)):
                    for etype in etypes:
                        graph_with_edge_type[etype] += 1
        
        if graph_with_edge_type:
            logger.log(f"\nEdge type distribution (proteins with each edge type):")
            for etype in sorted(graph_with_edge_type.keys()):
                count = graph_with_edge_type[etype]
                percentage = count / total_graphs * 100 if total_graphs > 0 else 0
                logger.log(f"  {etype:15s}: {count:4d} proteins ({percentage:6.2f}%)")
        
        logger.log("="*70)
        
        return {
            'metrics': metrics,
            'predictions': results['predictions'],
            'probabilities': results['probabilities'],
            'logits': results['logits'],
            'labels': labels.cpu().numpy() if torch.is_tensor(labels) else labels
        }
    else:
        logger.log("No labels available for metric calculation")
        return {
            'metrics': {},
            'predictions': results['predictions'],
            'probabilities': results['probabilities'],
            'logits': results['logits'],
            'labels': None
        }


def run_cross_dataset_inference(args, logger):
    """
    Run cross-dataset inference (train on one dataset, test on another)
    
    Args:
        args: Command line arguments
        logger: Logger instance
    
    Returns:
        Evaluation results dictionary
    """
    logger.log("="*70)
    logger.log("Cross-Dataset Inference")
    logger.log("="*70)
    logger.log(f"Training dataset: {args.train_dataset}")
    logger.log(f"Testing dataset: {args.test_dataset}")
    logger.log("")
    
    # Load config from file first (consistent with customer_train.py)
    config = None
    if args.config:
        logger.log(f"Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            config_file = json.load(f)
        
        # Flatten the nested JSON structure (same as customer_train.py)
        config = {
            # Model parameters
            'input_dim': config_file.get('model', {}).get('input_dim', 7),
            'prot_hidden_dim': config_file.get('model', {}).get('prot_hidden_dim', 256),
            'ppi_hidden_dim': config_file.get('model', {}).get('ppi_hidden_dim', 256),
            'prot_num_layers': config_file.get('model', {}).get('prot_num_layers', 3),
            'ppi_num_layers': config_file.get('model', {}).get('ppi_num_layers', 3),
            'output_dim': config_file.get('model', {}).get('output_dim', 7),
            'dropout_ratio': config_file.get('model', {}).get('dropout_ratio', 0.3),
            'use_attention': config_file.get('model', {}).get('use_attention', False),
            'num_heads': config_file.get('model', {}).get('num_heads', 4),
            
            # Encoding parameters
            'encoding_type': config_file.get('encoding', {}).get('encoding_type', 'mape'),
            'encoding_config': {
                'feature_file': config_file.get('encoding', {}).get('feature_file', None),
                'embedding_dir': config_file.get('encoding', {}).get('embedding_dir', None),
                'validate_dims': config_file.get('encoding', {}).get('validate_dims', True)
            },
            
            # Edge construction parameters
            'spatial_threshold': config_file.get('edge_construction', {}).get('spatial_threshold', 10.0),
            'knn_k': config_file.get('edge_construction', {}).get('knn_k', 5),
            'surface_threshold': config_file.get('edge_construction', {}).get('surface_threshold', 0.2),
            'surface_distance': config_file.get('edge_construction', {}).get('surface_distance', 8.0),
            
            # Multi-encoder parameters
            'peptide_encoder_enabled': config_file.get('encoding', {}).get('peptide_encoder_enabled', False),
            'peptide_length_threshold': config_file.get('encoding', {}).get('peptide_length_threshold', 50),
            'lrr_encoder_enabled': config_file.get('encoding', {}).get('lrr_encoder_enabled', False),
            'lrr_annotation_file': config_file.get('encoding', {}).get('lrr_annotation_file', 'customer_ppi/scripts/lrr/lrr_annotation_results.txt'),
            
            # Add encoding section for compatibility
            'encoding': config_file.get('encoding', {}),
            'edge_construction': config_file.get('edge_construction', {})
        }
        
        # Override with command line arguments
        if args.embedding_dir:
            config['encoding_config']['embedding_dir'] = args.embedding_dir
            config['encoding']['embedding_dir'] = args.embedding_dir
    
    # Load model from training dataset
    logger.log(f"Loading model trained on {args.train_dataset}...")
    try:
        predictor = PPIPredictor(args.checkpoint, args.explainable, config)
        logger.log(f"Model loaded successfully")
        
        # If config was not provided and checkpoint has no config, use minimal default
        if not predictor.config:
            logger.log("Warning: Using minimal default config")
            predictor.config = {
                'input_dim': 7,
                'prot_hidden_dim': 256,
                'ppi_hidden_dim': 256,
                'prot_num_layers': 3,
                'ppi_num_layers': 3,
                'output_dim': 7,
                'dropout_ratio': 0.3,
                'use_attention': False,
                'num_heads': 4,
                'encoding_type': 'mape',
                'encoding_config': {},
                'spatial_threshold': 10.0,
                'knn_k': 5,
                'surface_threshold': 0.2,
                'surface_distance': 8.0
            }
            predictor.config['encoding']['embedding_dir'] = args.embedding_dir
            predictor.config['encoding_config']['embedding_dir'] = args.embedding_dir
        
        logger.log(f"Model config: {json.dumps(predictor.config, indent=2)}")
    except Exception as e:
        logger.log(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load test dataset
    logger.log(f"\nLoading test dataset: {args.test_dataset}...")
    
    # Validate test dataset files
    if not os.path.exists(args.ppi_file):
        logger.log(f"❌ Error: PPI file not found: {args.ppi_file}")
        return None
    
    if not os.path.exists(args.protein_seq_file):
        logger.log(f"❌ Error: Protein sequence file not found: {args.protein_seq_file}")
        return None
    
    if not os.path.exists(args.pdb_dir):
        logger.log(f"❌ Error: PDB directory not found: {args.pdb_dir}")
        return None
    
    try:
        ppi_dataset = PPIDataset(
            config=predictor.config,
            ppi_file=args.ppi_file,
            protein_seq_file=args.protein_seq_file,
            pdb_dir=args.pdb_dir,
            cache_dir=args.cache_dir
        )
        logger.log(f"Test dataset loaded: {len(ppi_dataset.ppi_list)} PPI pairs")
        logger.log(f"Number of proteins: {len(ppi_dataset.protein_dataset)}")
        
        # Detect actual classes in test dataset and adjust output dimension if needed
        # (Same logic as in customer_train.py)
        actual_num_classes, actual_class_map = ppi_dataset.detect_actual_classes()
        config_output_dim = predictor.config.get('output_dim', 7)
        
        if actual_num_classes != config_output_dim:
            logger.log(f"🔧 Adjusting model output dimension from {config_output_dim} to {actual_num_classes}")
            logger.log(f"   Config has {config_output_dim} classes, but test dataset has {actual_num_classes} classes")
            
            # Update config with actual number of classes
            predictor.config['output_dim'] = actual_num_classes
            
            # Reinitialize model with adjusted output dimension
            logger.log("Reinitializing model with adjusted output dimension...")
            
            # Re-create the model with new output dimension
            use_heterogeneous = (predictor.config.get('encoding', {}).get('peptide_encoder_enabled', False) or 
                               predictor.config.get('encoding', {}).get('lrr_encoder_enabled', False))
            lrr_enabled = predictor.config.get('encoding', {}).get('lrr_encoder_enabled', False)
            
            if args.explainable:
                predictor.model = ExplainableProteinGINModel(predictor.config).to(device)
            else:
                predictor.model = ProteinGINModelSimple(predictor.config).to(device)
            
            # Try to load weights again with adjusted model
            try:
                from models.checkpoint import load_model_for_inference
                load_model_for_inference(predictor.model, args.checkpoint, device)
                logger.log("✅ Model weights loaded successfully after dimension adjustment")
            except Exception as load_err:
                logger.log(f"⚠️ Warning: Could not load weights after dimension adjustment: {load_err}")
                logger.log("   Proceeding with randomly initialized model...")
            
            predictor.model.eval()
        else:
            logger.log(f"✅ Dataset has {actual_num_classes} classes, matching model output dimension")
        
        # Store the actual class mapping for later use
        predictor.config['actual_class_map'] = actual_class_map
        
    except Exception as e:
        logger.log(f"❌ Error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run evaluation with metrics
    logger.log("\nRunning evaluation...")
    try:
        results = evaluate_with_metrics(
            predictor=predictor,
            ppi_dataset=ppi_dataset,
            batch_size=args.batch_size,
            logger=logger
        )
        
        if results is None or not results.get('metrics'):
            logger.log("❌ Evaluation failed or no metrics calculated")
            return None
        
        # Save results
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        df = predictor.batch_predict(
            ppi_dataset.ppi_list,
            ppi_dataset.ppi_graph,
            predictor.encode_proteins(ppi_dataset.protein_dataset),
            args.threshold
        )
        df.to_csv(args.output, index=False)
        logger.log(f"\nPredictions saved to: {args.output}")
        
        # Save metrics to JSON
        metrics_file = args.output.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        logger.log(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        results_dict = {
            'train_dataset': args.train_dataset,
            'test_dataset': args.test_dataset,
            'checkpoint': args.checkpoint,
            'num_samples': len(ppi_dataset.ppi_list),
            'num_proteins': len(ppi_dataset.protein_dataset),
            'metrics': results['metrics'],
            'class_names': predictor.class_names
        }
        
        summary_file = args.output.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        logger.log(f"Summary saved to: {summary_file}")
        
        # Save predictions and labels as numpy arrays
        np.save(args.output.replace('.csv', '_probs.npy'), results['probabilities'])
        np.save(args.output.replace('.csv', '_preds.npy'), results['predictions'])
        if results['labels'] is not None:
            np.save(args.output.replace('.csv', '_labels.npy'), results['labels'])
        
        logger.log("\n✅ Cross-dataset inference completed successfully!")
        return results
        
    except Exception as e:
        logger.log(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(args):
    """Main inference function with enhanced error handling"""
    
    # Set random seed for reproducibility
    set_seed(42)  # Use fixed seed for consistency
    
    # Initialize logger
    logger = SimpleLogger(args.log_file)
    logger.log(f"Using device: {device}")
    logger.log(f"Loading model from: {args.checkpoint}")
    
    # Cross-dataset inference mode
    if args.cross_dataset or (args.train_dataset and args.test_dataset):
        results = run_cross_dataset_inference(args, logger)
        if results:
            logger.log("\n✅ Cross-dataset inference completed successfully!")
        else:
            logger.log("\n❌ Cross-dataset inference failed")
        return
    
    # Standard inference mode (original code)
    try:
        # Initialize predictor
        predictor = PPIPredictor(args.checkpoint, args.explainable)
        logger.log(f"Model loaded successfully")
        logger.log(f"Configuration: {json.dumps(predictor.config, indent=2)}")
        
        # Validate required files
        if not os.path.exists(args.ppi_file):
            logger.log(f"Error: PPI file not found: {args.ppi_file}")
            return
        
        if not os.path.exists(args.protein_seq_file):
            logger.log(f"Error: Protein sequence file not found: {args.protein_seq_file}")
            return
        
        if not os.path.exists(args.pdb_dir):
            logger.log(f"Error: PDB directory not found: {args.pdb_dir}")
            return
        
        # Load dataset
        logger.log("Loading dataset...")
        ppi_dataset = PPIDataset(
            config=predictor.config,
            ppi_file=args.ppi_file,
            protein_seq_file=args.protein_seq_file,
            pdb_dir=args.pdb_dir,
            cache_dir=args.cache_dir
        )
    except FileNotFoundError as e:
        logger.log(f"❌ File not found error: {e}")
        logger.log("Please check file paths and ensure all required files exist")
        return
    except ValueError as e:
        logger.log(f"❌ Data validation error: {e}")
        logger.log("Please check data format and content")
        return
    except Exception as e:
        logger.log(f"❌ Unexpected error during initialization: {e}")
        logger.log("Please check configuration and model files")
        return
    
    # Encode proteins with enhanced error handling
    logger.log("Encoding proteins...")
    try:
        prot_embed = predictor.encode_proteins(ppi_dataset.protein_dataset)
        logger.log(f"Protein embeddings shape: {prot_embed.shape}")
        
        # Validate dimensions match between PPI graph and protein embeddings
        logger.log("Validating dimensions...")
        ppi_num_nodes = ppi_dataset.ppi_graph.num_nodes()
        prot_embed_num = prot_embed.shape[0]
        
        logger.log(f"PPI graph nodes: {ppi_num_nodes}")
        logger.log(f"Protein embeddings: {prot_embed_num}")
        
        if ppi_num_nodes != prot_embed_num:
            logger.log(f"⚠️ WARNING: Dimension mismatch! PPI graph has {ppi_num_nodes} nodes, but protein embeddings have {prot_embed_num} entries")
            logger.log(f"  Difference: {abs(ppi_num_nodes - prot_embed_num)} proteins")
            
            # Attempt dimension correction
            if ppi_num_nodes < prot_embed_num:
                logger.log("  Truncating protein embeddings to match PPI graph nodes...")
                prot_embed = prot_embed[:ppi_num_nodes]
                logger.log(f"  Adjusted protein embeddings shape: {prot_embed.shape}")
            else:
                logger.log("  ❌ ERROR: PPI graph has more nodes than protein embeddings!")
                logger.log("  This indicates a serious data inconsistency.")
                raise ValueError(f"Dimension mismatch: PPI graph has {ppi_num_nodes} nodes, but only {prot_embed_num} protein embeddings")
        else:
            logger.log("✅ Dimensions match!")
            
    except Exception as e:
        logger.log(f"❌ Error during protein encoding: {e}")
        logger.log("Please check protein graph generation and encoding configuration")
        return
    
    # Prediction mode
    if args.mode == 'all':
        # Predict all interactions
        logger.log("Predicting all interactions...")
        all_indices = list(range(len(ppi_dataset.ppi_list)))
        
        results = predictor.predict_interactions(
            ppi_dataset.ppi_graph,
            prot_embed,
            ppi_dataset.ppi_list,
            all_indices,
            args.threshold
        )
        
        # Save results
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        np.save(args.output.replace('.csv', '_probs.npy'), results['probabilities'])
        np.save(args.output.replace('.csv', '_preds.npy'), results['predictions'])
        
        # Create DataFrame
        df = predictor.batch_predict(
            ppi_dataset.ppi_list,
            ppi_dataset.ppi_graph,
            prot_embed,
            args.threshold
        )
        df.to_csv(args.output, index=False)
        
        logger.log(f"Results saved to: {args.output}")
        logger.log(f"Total predictions: {len(df)}")
        
        # Summary statistics
        for class_name in predictor.class_names:
            positive_count = df[f'{class_name}_pred'].sum()
            logger.log(f"{class_name}: {positive_count} positive predictions "
                      f"({positive_count/len(df)*100:.2f}%)")
    
    elif args.mode == 'pairs':
        # Predict specific pairs from file
        logger.log(f"Loading protein pairs from: {args.pairs_file}")
        
        pairs = []
        with open(args.pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((int(parts[0]), int(parts[1])))
        
        logger.log(f"Loaded {len(pairs)} protein pairs")
        
        # Predict
        df = predictor.batch_predict(
            pairs,
            ppi_dataset.ppi_graph,
            prot_embed,
            args.threshold
        )
        
        # Save results
        df.to_csv(args.output, index=False)
        logger.log(f"Results saved to: {args.output}")
    
    elif args.mode == 'single':
        # Predict single pair
        logger.log(f"Predicting interaction between proteins {args.protein1} and {args.protein2}")
        
        result = predictor.predict_single_pair(
            args.protein1,
            args.protein2,
            ppi_dataset.ppi_graph,
            prot_embed,
            args.threshold
        )
        
        # Print results
        logger.log("\nPrediction Results:")
        logger.log("-" * 60)
        for class_name, values in result.items():
            logger.log(f"{class_name:12s}: prob={values['probability']:.4f}, "
                      f"pred={'YES' if values['prediction'] else 'NO'}")
        logger.log("-" * 60)
        
        # Save to JSON
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.log(f"\nResults saved to: {args.output}")
    
    logger.log("\nInference completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer PPI Inference")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--explainable", action="store_true", help="Use explainable model")
    
    # Data arguments
    parser.add_argument("--ppi_file", type=str, required=True, help="Path to PPI file")
    parser.add_argument("--protein_seq_file", type=str, required=True, help="Path to protein sequence file")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    
    # Cross-dataset arguments
    parser.add_argument("--train_dataset", type=str, default=None, 
                       help="Training dataset name (e.g., SYS30k, SHS27k) for cross-dataset inference")
    parser.add_argument("--test_dataset", type=str, default=None,
                       help="Testing dataset name (e.g., SYS60k, SHS148k) for cross-dataset inference")
    parser.add_argument("--cross_dataset", action="store_true",
                       help="Enable cross-dataset inference mode")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation mode with metrics calculation")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for inference")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (for old checkpoints without config)")
    parser.add_argument("--embedding_dir", type=str, default=None,
                       help="Directory for precomputed embeddings")
    
    # Inference mode
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["all", "pairs", "single"],
                       help="Inference mode")
    parser.add_argument("--pairs_file", type=str, help="File with protein pairs (for 'pairs' mode)")
    parser.add_argument("--protein1", type=int, help="First protein ID (for 'single' mode)")
    parser.add_argument("--protein2", type=int, help="Second protein ID (for 'single' mode)")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--log_file", type=str, default="inference.log", help="Log file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'pairs' and not args.pairs_file:
        parser.error("--pairs_file is required for 'pairs' mode")
    if args.mode == 'single' and (args.protein1 is None or args.protein2 is None):
        parser.error("--protein1 and --protein2 are required for 'single' mode")
    
    # Cross-dataset validation
    if args.cross_dataset:
        if not args.train_dataset or not args.test_dataset:
            parser.error("--train_dataset and --test_dataset are required for cross-dataset inference")
    
    main(args)

