"""
Customer PPI Training Script
Comprehensive training pipeline for protein-protein interaction prediction
"""
import traceback 
import os
import sys
import time
import argparse
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
# from muon import Muon, MuonWithAuxAdam

import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.integrated_high_ppi_simple import ProteinGINModelSimple, ExplainableProteinGINModel
from models.customer_dataloader import PPIDataset
# from models.customer_dataloader_balanced import BalancedPPIDataset, encode_proteins
# from models.customer_dataloader_balanced_new import BalancedPPIDataset, encode_proteins
from models.metrics import MetricsCalculator, format_metrics_string
from models.logger import TrainingLogger
from models.checkpoint import CheckpointManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # Convert logits to probabilities
        prob = torch.sigmoid(inputs)
        
        # Calculate cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', 
                                                    pos_weight=self.pos_weight)
        
        # Calculate p_t
        p_t = prob * targets + (1 - prob) * (1 - targets)
        
        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_ppi_indices_for_lrr_update(ppi_list, use_all=False, max_batch_size=100):
    """
    Get PPI indices for LRR weight update, optionally using all pairs in batches
    
    Args:
        ppi_list: List of all PPI pairs
        use_all: If True, return all indices (will be processed in batches)
        max_batch_size: Maximum number of PPI pairs per batch (default 100)
        
    Returns:
        List of lists: Each sub-list contains up to max_batch_size PPI indices
        Example: [[0,1,2,...,99], [100,101,...,199], ...]
    """
    if use_all:
        all_indices = list(range(len(ppi_list)))
        # Split into batches of max_batch_size
        return [all_indices[i:i+max_batch_size] 
                for i in range(0, len(all_indices), max_batch_size)]
    else:
        # Sample 100 pairs and return as single batch
        sample_size = min(100, len(ppi_list))
        if sample_size == len(ppi_list):
            return [list(range(len(ppi_list)))]
        else:
            sampled = random.sample(range(len(ppi_list)), sample_size)
            return [sampled]


def train_epoch(model, protein_graphs, ppi_g, ppi_list, labels, indices, 
                batch_size, optimizer, loss_fn, metrics_calc):
    """Train for one epoch with integrated encoder (encoding inside model)"""
    model.train()
    
    # 每个epoch开始时编码一次，后续batch复用
    prot_embed = model.encode_proteins(protein_graphs)
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Shuffle indices
    random.shuffle(indices)
    num_batches = math.ceil(len(indices) / batch_size)
    
    for batch_idx in range(num_batches):
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        
        # === [LRRDEBUG] 批次开始前记录参数状态 ===
        if batch_idx == 0:
            print(f"\n[LRRDEBUG] Batch {batch_idx} - 训练前参数状态:")
            for name, param in model.named_parameters():
                if 'lrr_logits' in name or 'lrr_encoder' in name:
                    # print(f"  {name} 值: {param.data.tolist()}")
                    print(f"  {name} requires_grad: {param.requires_grad}")
        
        # Forward pass - 使用预编码的prot_embed
        output = model(ppi_g, prot_embed, ppi_list, batch_indices)
        
        # Use weighted loss function to handle class imbalance
        if hasattr(loss_fn, '__name__') and 'weighted' in loss_fn.__name__.lower():
            loss = loss_fn(output, labels[batch_indices])
        else:
            # Fallback to standard loss if weighted loss not available
            loss = loss_fn(output, labels[batch_indices])
        
        # === [LRRDEBUG] 计算损失后记录状态 ===
        if batch_idx == 0:
            print(f"\n[LRRDEBUG] Batch {batch_idx} - 损失计算后:")
            print(f"  损失值: {loss.item():.6f}")
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
        
        # Backward pass (gradient flows back to encoder weights)
        optimizer.zero_grad()
        
        # === [LRRDEBUG] 反向传播前记录梯度状态 ===
        if batch_idx == 0:
            print(f"\n[LRRDEBUG] Batch {batch_idx} - 反向传播前梯度状态:")
            for name, param in model.named_parameters():
                if 'lrr_logits' in name or 'lrr_encoder' in name:
                    grad_info = param.grad.tolist() if param.grad is not None else "None"
                    print(f"  {name} 梯度 (before backward): {grad_info}")
        
        loss.backward()
        

        

        
        optimizer.step()
        


        
        # Track metrics
        total_loss += loss.item()
        all_predictions.append(output.detach())
        all_labels.append(labels[batch_indices].detach())
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = metrics_calc.calculate_all_metrics(all_predictions, all_labels)
    metrics['loss'] = total_loss / num_batches
    # === [LRRDEBUG] 开始训练前记录初始权重 ===
    print(f"\n[LRRDEBUG] 后训练 - 权重:")
    initial_weights = model.get_weights()
    for etype, weight in initial_weights.items():
        print(f"  {etype}: {weight:.6f}")

    
    return metrics


def evaluate(model, protein_graphs, ppi_g, ppi_list, labels, indices, 
            batch_size, loss_fn, metrics_calc, mode='val'):
    """Evaluate model (integrated encoder)"""
    model.eval()
    
    # 编码一次，后续batch复用
    prot_embed = model.encode_proteins(protein_graphs)
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    num_batches = math.ceil(len(indices) / batch_size)
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Forward pass - 使用预编码的prot_embed
            output = model(ppi_g, prot_embed, ppi_list, batch_indices)
            
            # Use weighted loss function
            if hasattr(loss_fn, '__name__') and 'weighted' in loss_fn.__name__.lower():
                loss = loss_fn(output, labels[batch_indices])
            else:
                # Fallback to standard loss if weighted loss not available
                loss = loss_fn(output, labels[batch_indices])
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(output.detach())
            all_labels.append(labels[batch_indices].detach())
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = metrics_calc.calculate_all_metrics(all_predictions, all_labels)
    metrics['loss'] = total_loss / num_batches
    

    
    return metrics, all_predictions, all_labels


def encode_proteins(model, protein_dataset):
    """Encode all proteins using the model"""
    # Check if protein dataset has graphs
    if not hasattr(protein_dataset, 'graphs') or len(protein_dataset.graphs) == 0:
        raise ValueError(f"Protein dataset has no graphs! Dataset length: {len(protein_dataset)}")

    print(f"Encoding {len(protein_dataset.graphs)} proteins...")

    # 检测是否需要使用三类型分类批处理
    from models.customer_dataloader import is_heterogeneous_dataset, collate_heterogeneous_protein_graphs
    
    use_heterogeneous_encoding = is_heterogeneous_dataset(protein_dataset.graphs)
    
    if use_heterogeneous_encoding:
        print("检测到异质性数据集，使用三类型分类批处理")
        return _encode_heterogeneous_proteins(model, protein_dataset)
    else:
        print("使用标准批处理")
        return _encode_standard_proteins(model, protein_dataset)


def _encode_standard_proteins(model, protein_dataset):
    """标准蛋白质编码"""
    protein_loader = DataLoader(
        protein_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_protein_graphs
    )

    model.eval()
    protein_embeds = []
    
    # 初始化统计变量
    total_input_stats = {
        'min': float('inf'), 'max': float('-inf'),
        'mean': 0.0, 'std': 0.0, 'count': 0
    }
    total_output_stats = {
        'min': float('inf'), 'max': float('-inf'),
        'mean': 0.0, 'std': 0.0, 'count': 0
    }

    with torch.no_grad():
        for batch_idx, batch_graph in enumerate(protein_loader):
            # Move graph to device
            batch_graph = batch_graph.to(device)
            # Explicitly move node features to device
            if 'x' in batch_graph.ndata:
                batch_graph.ndata['x'] = batch_graph.ndata['x'].to(device)
            
            # 1. 编码前：分析输入特征
            input_x = batch_graph.ndata['x']
            
            # 计算输入统计
            input_stats = {
                'min': input_x.min().item(),
                'max': input_x.max().item(),
                'mean': input_x.mean().item(),
                'std': input_x.std().item(),
                'shape': input_x.shape,
                'non_zero': (input_x != 0).sum().item(),
                'zero_ratio': ((input_x == 0).sum().item() / input_x.numel()) * 100
            }
            
            # 更新总体统计
            total_input_stats['min'] = min(total_input_stats['min'], input_stats['min'])
            total_input_stats['max'] = max(total_input_stats['max'], input_stats['max'])
            total_input_stats['count'] += input_x.shape[0]
            
            # 2. 执行编码
            # Check if this is a heterogeneous model with different encoding interface
            if hasattr(model, 'encode_proteins'):
                # For heterogeneous models, we need to be careful with batch processing
                # Process the batch graph directly
                x, graph_embed = model.protein_encoder(batch_graph)
                protein_embeds.append(graph_embed.cpu())
            else:
                # Standard model encoding
                x, graph_embed = model.protein_encoder(batch_graph)
                protein_embeds.append(graph_embed.cpu())
            
            # 3. 编码后：分析输出特征
            # x 是节点级别的嵌入
            output_stats = {
                'min': x.min().item(),
                'max': x.max().item(),
                'mean': x.mean().item(),
                'std': x.std().item(),
                'shape': x.shape,
                'non_zero': (x != 0).sum().item(),
                'zero_ratio': ((x == 0).sum().item() / x.numel()) * 100,
                'negative_ratio': ((x < 0).sum().item() / x.numel()) * 100
            }
            
            # 更新总体输出统计
            total_output_stats['min'] = min(total_output_stats['min'], output_stats['min'])
            total_output_stats['max'] = max(total_output_stats['max'], output_stats['max'])
            total_output_stats['count'] += x.shape[0]
            
            # 4. 计算差异
            if batch_idx == 0:
                print(f"\n{'='*60}")
                print(f"批次 {batch_idx} 的输入输出分析:")
                print(f"{'='*60}")
                
                # 输入特征分析
                print(f"输入特征 (batch_graph.ndata['x']):")
                print(f"  形状: {input_stats['shape']}")
                print(f"  范围: [{input_stats['min']:.6f}, {input_stats['max']:.6f}]")
                print(f"  均值: {input_stats['mean']:.6f}, 标准差: {input_stats['std']:.6f}")
                print(f"  非零值: {input_stats['non_zero']}/{input_x.numel()} ({input_stats['zero_ratio']:.2f}% 为零)")
                
                # 输出特征分析
                print(f"\n输出节点嵌入 (x):")
                print(f"  形状: {output_stats['shape']}")
                print(f"  范围: [{output_stats['min']:.6f}, {output_stats['max']:.6f}]")
                print(f"  均值: {output_stats['mean']:.6f}, 标准差: {output_stats['std']:.6f}")
                print(f"  非零值: {output_stats['non_zero']}/{x.numel()} ({output_stats['zero_ratio']:.2f}% 为零)")
                print(f"  负值比例: {output_stats['negative_ratio']:.2f}%")
                
                # 图级别嵌入分析
                print(f"\n图级别嵌入 (graph_embed):")
                print(f"  形状: {graph_embed.shape}")
                print(f"  范围: [{graph_embed.min().item():.6f}, {graph_embed.max().item():.6f}]")
                print(f"  均值: {graph_embed.mean().item():.6f}, 标准差: {graph_embed.std().item():.6f}")
                
                # 抽样查看几个节点的具体变化
                if x.shape[0] >= 5:  # 如果有足够多的节点
                    print(f"\n抽样节点特征变化 (前5个节点):")
                    print(f"{'节点':<6} {'输入特征(前3维)':<30} {'输出嵌入(前3维)':<30}")
                    for i in range(min(5, x.shape[0])):
                        input_sample = input_x[i][:3].cpu().numpy()
                        output_sample = x[i][:3].cpu().numpy()
                        print(f"{i:<6} {str(input_sample):<30} {str(output_sample):<30}")
            
            elif batch_idx % 10 == 0:  # 每10个批次打印一次进度
                print(f"  批次 {batch_idx}: 输入范围[{input_stats['min']:.3f}, {input_stats['max']:.3f}], "
                      f"输出范围[{output_stats['min']:.3f}, {output_stats['max']:.3f}], "
                      f"图嵌入[{graph_embed.min().item():.3f}, {graph_embed.max().item():.3f}]")

    if len(protein_embeds) == 0:
        raise ValueError("No protein embeddings generated! Check if DataLoader is working correctly.")

    print(f"\n{'='*60}")
    print(f"编码完成总结:")
    print(f"{'='*60}")
    
    if total_input_stats['count'] > 0:
        print(f"输入特征统计 (跨所有批次):")
        print(f"  范围: [{total_input_stats['min']:.6f}, {total_input_stats['max']:.6f}]")
        print(f"  总计节点数: {total_input_stats['count']}")
        
        print(f"\n输出嵌入统计 (跨所有批次):")
        print(f"  范围: [{total_output_stats['min']:.6f}, {total_output_stats['max']:.6f}]")
        print(f"  总计节点数: {total_output_stats['count']}")
    
    # 合并所有嵌入
    if len(protein_embeds) > 1:
        result = torch.cat(protein_embeds, dim=0).to(device)
    else:
        result = protein_embeds[0].to(device)
    
    print(f"  最终图嵌入形状: {result.shape}")
    print(f"  范围: [{result.min().item():.6f}, {result.max().item():.6f}]")
    print(f"  均值: {result.mean().item():.6f}, 标准差: {result.std().item():.6f}")
    print(f"  Processed {result.shape[0]} protein embeddings")
    
    return result

def print_model_parameters(model):
    """
    打印模型的参数统计信息，特别关注 TrainableLrrEncoder 的参数
    """
    print(f"\n{'='*60}")
    print("模型参数统计")
    print(f"{'='*60}")
    
    total_params = 0
    trainable_params = 0
    encoder_params = 0
    encoder_trainable_params = 0
    
    # 统计所有参数
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        # 检查是否是编码器参数
        if 'protein_encoder' in name or 'encoder' in name.lower():
            encoder_params += param_count
            if param.requires_grad:
                encoder_trainable_params += param_count
            
            # 如果是 TrainableLrrEncoder 的权重参数，特别标注
            if 'weight' in name and ('seq_weight' in name or 'str_knn_weight' in name or 
                                   'str_dis_weight' in name or 'surf_weight' in name or 
                                   'lrr_region_weight' in name):
                print(f"  {name}:")
                print(f"    形状: {list(param.shape)}")
                print(f"    数量: {param_count:,}")
                print(f"    可训练: {param.requires_grad}")
                if param.requires_grad:
                    print(f"    当前值: {param.item():.6f} (sigmoid后: {torch.sigmoid(param).item():.6f})")
    
    print(f"\n总体统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  编码器参数: {encoder_params:,}")
    print(f"  编码器可训练参数: {encoder_trainable_params:,}")
    print(f"  编码器冻结参数: {encoder_params - encoder_trainable_params:,}")
    
    # 如果是 TrainableLrrEncoder，打印权重信息
    if hasattr(model, 'protein_encoder') and hasattr(model.protein_encoder, 'get_weights'):
        print(f"\nTrainableLrrEncoder 当前权重:")
        weights = model.protein_encoder.get_weights()
        for etype, weight in weights.items():
            print(f"  {etype}: {weight:.6f}")
    
    print(f"{'='*60}\n")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'encoder_params': encoder_params,
        'encoder_trainable_params': encoder_trainable_params
    }


def analyze_lrr_vs_non_lrr(model, protein_graphs, ppi_g, ppi_list, labels, 
                            split_dict, batch_size, logger, device):
    """
    分析有LRR和无LRR的PPI对的权重和准确率
    
    分类:
    - 单边LRR: 只有一个蛋白质有LRR区域
    - 多边LRR: 两个蛋白质都有LRR区域  
    - 无LRR: 两个蛋白质都没有LRR区域
    
    分别对训练集、验证集、测试集进行分析
    """
    logger.log("\n" + "="*70)
    logger.log("LRR vs Non-LRR PPI 分析")
    logger.log("="*70)
    
    # 1. 首先判断每个蛋白质是否有LRR区域
    protein_has_lrr = {}
    for idx, graph in enumerate(protein_graphs):
        if hasattr(graph, 'etypes'):
            has_lrr_edge = 'LRR_REGION' in graph.etypes and graph.num_edges('LRR_REGION') > 0
            protein_has_lrr[idx] = has_lrr_edge
        else:
            protein_has_lrr[idx] = False
    
    # 统计有LRR和无LRR的蛋白质数量
    num_proteins_with_lrr = sum(protein_has_lrr.values())
    num_proteins_without_lrr = len(protein_has_lrr) - num_proteins_with_lrr
    logger.log(f"\n蛋白质统计:")
    logger.log(f"  有LRR区域的蛋白质: {num_proteins_with_lrr}")
    logger.log(f"  无LRR区域的蛋白质: {num_proteins_without_lrr}")
    
    # 2. 对PPI对进行分类（全局分类）
    single_lrr_all = []  # 单边LRR
    multi_lrr_all = []   # 多边LRR (双边都有)
    no_lrr_all = []      # 无LRR
    
    for idx in range(len(ppi_list)):
        p1, p2 = ppi_list[idx]
        p1_has_lrr = protein_has_lrr.get(p1, False)
        p2_has_lrr = protein_has_lrr.get(p2, False)
        
        if p1_has_lrr and p2_has_lrr:
            multi_lrr_all.append(idx)
        elif p1_has_lrr or p2_has_lrr:
            single_lrr_all.append(idx)
        else:
            no_lrr_all.append(idx)
    
    logger.log(f"\n全局PPI对分类统计:")
    logger.log(f"  单边LRR (一个蛋白质有LRR): {len(single_lrr_all)} 对")
    logger.log(f"  多边LRR (两个蛋白质都有LRR): {len(multi_lrr_all)} 对")
    logger.log(f"  无LRR (两个蛋白质都没有LRR): {len(no_lrr_all)} 对")
    
    # 3. 根据split_dict对每个数据集进行分析
    model.eval()
    
    def process_category(indices, category_name):
        """处理某一类PPI对，返回权重、预测结果和指标"""
        if len(indices) == 0:
            return None, None
        
        # 重置alpha统计
        if hasattr(model, 'lrr_encoder') and hasattr(model.lrr_encoder, 'reset_alpha_stats'):
            model.lrr_encoder.reset_alpha_stats()
        
        all_preds = []
        all_labels_list = []
        
        with torch.no_grad():
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                output = model(ppi_g, protein_graphs, ppi_list, batch_indices)
                all_preds.append(output.detach())
                all_labels_list.append(labels[batch_indices].detach())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels_cat = torch.cat(all_labels_list, dim=0)
        
        # 获取该类PPI对的平均权重
        weights = None
        if hasattr(model, 'get_weights'):
            weights = model.get_weights()
        
        # 计算指标
        preds_binary = (torch.sigmoid(all_preds) > 0.5).float()
        accuracy = (preds_binary == all_labels_cat).float().mean().item()
        
        tp = ((preds_binary == 1) & (all_labels_cat == 1)).sum().item()
        tn = ((preds_binary == 0) & (all_labels_cat == 0)).sum().item()
        fp = ((preds_binary == 1) & (all_labels_cat == 0)).sum().item()
        fn = ((preds_binary == 0) & (all_labels_cat == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'count': len(indices),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        return weights, metrics
    
    # 获取各类PPI对在各个数据集中的索引
    def get_category_indices_in_split(lrr_category_indices, split_indices):
        """获取某一类PPI对在特定split中的索引"""
        return [idx for idx in split_indices if idx in lrr_category_indices]
    
    all_results = {}
    
    # 对每个数据集进行分析
    for split_name, split_indices in [('train', split_dict['train_index']),
                                       ('val', split_dict['val_index']),
                                       ('test', split_dict['test_index'])]:
        logger.log(f"\n{'='*70}")
        logger.log(f"数据集: {split_name.upper()} (共 {len(split_indices)} 对)")
        logger.log(f"{'='*70}")
        
        # 获取三类PPI对在该split中的索引
        single_lrr_split = get_category_indices_in_split(single_lrr_all, split_indices)
        multi_lrr_split = get_category_indices_in_split(multi_lrr_all, split_indices)
        no_lrr_split = get_category_indices_in_split(no_lrr_all, split_indices)
        
        logger.log(f"  单边LRR: {len(single_lrr_split)} 对")
        logger.log(f"  多边LRR: {len(multi_lrr_split)} 对")
        logger.log(f"  无LRR: {len(no_lrr_split)} 对")
        
        split_results = {}
        
        # 处理三类PPI对
        for indices, name in [(single_lrr_split, "单边LRR"),
                              (multi_lrr_split, "多边LRR"),
                              (no_lrr_split, "无LRR")]:
            if len(indices) == 0:
                logger.log(f"\n  {name}: 无数据")
                continue
                
            weights, metrics = process_category(indices, name)
            
            if weights is not None and metrics is not None:
                split_results[name] = {'weights': weights, 'metrics': metrics}
                
                logger.log(f"\n  {name}:")
                logger.log(f"    数量: {metrics['count']}")
                logger.log(f"    准确率: {metrics['accuracy']:.4f}")
                logger.log(f"    Precision: {metrics['precision']:.4f}")
                logger.log(f"    Recall: {metrics['recall']:.4f}")
                logger.log(f"    F1: {metrics['f1']:.4f}")
                logger.log(f"    权重: {', '.join([f'{k}={v:.4f}' for k, v in weights.items()])}")
        
        all_results[split_name] = split_results
    
    # 4. 汇总对比表
    logger.log(f"\n{'='*70}")
    logger.log("权重对比汇总 (按数据集和LRR类别)")
    logger.log(f"{'='*70}")
    
    edge_types = ['SEQ', 'STR_KNN', 'STR_DIS', 'SURF', 'LRR_REGION']
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in all_results:
            continue
        logger.log(f"\n{split_name.upper()}:")
        logger.log(f"  {'类别':<10} " + " ".join([f"{et:<12}" for et in edge_types]))
        logger.log(f"  {'-'*70}")
        
        for lrr_cat in ["单边LRR", "多边LRR", "无LRR"]:
            if lrr_cat in all_results[split_name]:
                weights = all_results[split_name][lrr_cat]['weights']
                row = f"  {lrr_cat:<10} " + " ".join([f"{weights.get(et, 0):<12.4f}" for et in edge_types])
                logger.log(row)
    
    # 准确率汇总表
    logger.log(f"\n{'='*70}")
    logger.log("准确率和F1对比汇总")
    logger.log(f"{'='*70}")
    logger.log(f"  {'数据集':<8} {'类别':<10} {'准确率':<10} {'F1':<10}")
    logger.log(f"  {'-'*50}")
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in all_results:
            continue
        for lrr_cat in ["单边LRR", "多边LRR", "无LRR"]:
            if lrr_cat in all_results[split_name]:
                metrics = all_results[split_name][lrr_cat]['metrics']
                logger.log(f"  {split_name:<8} {lrr_cat:<10} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f}")
    
    logger.log("=" * 70)
    
    return all_results


def main(args):
    """Main training function"""

    # Set random seed
    set_seed(args.seed)
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_file = json.load(f)
        # Flatten the nested JSON structure to match the expected config format
        config = {
            # Model parameters
            'input_dim': config_file.get('model', {}).get('input_dim', args.input_dim),
            'prot_hidden_dim': config_file.get('model', {}).get('prot_hidden_dim', args.prot_hidden_dim),
            'ppi_hidden_dim': config_file.get('model', {}).get('ppi_hidden_dim', args.ppi_hidden_dim),
            'prot_num_layers': config_file.get('model', {}).get('prot_num_layers', args.prot_num_layers),
            'ppi_num_layers': config_file.get('model', {}).get('ppi_num_layers', args.ppi_num_layers),
            'output_dim': config_file.get('model', {}).get('output_dim', args.output_dim),
            'dropout_ratio': config_file.get('model', {}).get('dropout_ratio', args.dropout_ratio),
            'use_attention': config_file.get('model', {}).get('use_attention', args.use_attention),
            'num_heads': config_file.get('model', {}).get('num_heads', args.num_heads),

            # Encoding parameters
            'encoding_type': config_file.get('encoding', {}).get('encoding_type', args.encoding_type),
            'encoding_config': {
                'feature_file': config_file.get('encoding', {}).get('feature_file', args.feature_file),
                'embedding_dir': config_file.get('encoding', {}).get('embedding_dir', args.embedding_dir),
                'validate_dims': config_file.get('encoding', {}).get('validate_dims', True)
            },

            # Edge construction parameters
            'spatial_threshold': config_file.get('edge_construction', {}).get('spatial_threshold', args.spatial_threshold),
            'knn_k': config_file.get('edge_construction', {}).get('knn_k', args.knn_k),
            'surface_threshold': config_file.get('edge_construction', {}).get('surface_threshold', args.surface_threshold),
            'surface_distance': config_file.get('edge_construction', {}).get('surface_distance', args.surface_distance),

            # Multi-encoder parameters
            'peptide_encoder_enabled': config_file.get('encoding', {}).get('peptide_encoder_enabled', False),
            'peptide_length_threshold': config_file.get('encoding', {}).get('peptide_length_threshold', 50),
            'lrr_encoder_enabled': config_file.get('encoding', {}).get('lrr_encoder_enabled', False),
            'lrr_annotation_file': config_file.get('encoding', {}).get('lrr_annotation_file', 'customer_ppi/scripts/lrr/lrr_annotation_results.txt'),
            
            # Add encoding section to config for compatibility
            'encoding': config_file.get('encoding', {}),
            'edge_construction': config_file.get('edge_construction', {})
        }

        # Override with command line arguments if provided
        if args.input_dim != 7:  # If not default
            config['input_dim'] = args.input_dim
        if args.encoding_type != "mape":  # If not default
            config['encoding_type'] = args.encoding_type
            # Also update encoding_config to ensure consistency
            config['encoding_config']['encoding_type'] = args.encoding_type
        if args.feature_file is not None:
            config['encoding_config']['feature_file'] = args.feature_file
        if args.embedding_dir is not None:
            config['encoding_config']['embedding_dir'] = args.embedding_dir
        
        # Override multi-encoder parameters from command line arguments
        if hasattr(args, 'peptide_encoder_enabled') and args.peptide_encoder_enabled:
            config['peptide_encoder_enabled'] = True
        if hasattr(args, 'peptide_length_threshold') and args.peptide_length_threshold != 50:  # If not default
            config['peptide_length_threshold'] = args.peptide_length_threshold
        if hasattr(args, 'lrr_encoder_enabled') and args.lrr_encoder_enabled:
            config['lrr_encoder_enabled'] = True
        if hasattr(args, 'lrr_annotation_file') and args.lrr_annotation_file != "customer_ppi/scripts/lrr/lrr_annotation_results.txt":
            config['lrr_annotation_file'] = args.lrr_annotation_file
        
        # Update training parameters from config file
        if 'training' in config_file:
            print(f"config_file:{config_file}")
            if hasattr(args, 'max_epochs') and args.max_epochs == 500:  # If using default
                args.max_epochs = config_file['training'].get('max_epochs', args.max_epochs)
            if hasattr(args, 'batch_size') and args.batch_size == 10000:  # If using default
                args.batch_size = config_file['training'].get('batch_size', args.batch_size)
            if hasattr(args, 'learning_rate') and args.learning_rate == 0.001:  # If using default
                old_lr = args.learning_rate
                args.learning_rate = config_file['training'].get('learning_rate', args.learning_rate)
                if args.learning_rate != old_lr:
                    print(f"✅ Learning rate updated from {old_lr} to {args.learning_rate} (from config file)")
                else:
                    print(f"ℹ️  Learning rate remains {args.learning_rate}")
            if hasattr(args, 'weight_decay') and args.weight_decay == 0.0005:  # If using default
                args.weight_decay = config_file['training'].get('weight_decay', args.weight_decay)
            if hasattr(args, 'lrr_learning_rate') and args.lrr_learning_rate == 0.001:  # If using default
                old_lrr_lr = args.lrr_learning_rate
                args.lrr_learning_rate = config_file['training'].get('lrr_learning_rate', args.lrr_learning_rate)
                if args.lrr_learning_rate != old_lrr_lr:
                    print(f"✅ LRR learning rate updated from {old_lrr_lr} to {args.lrr_learning_rate} (from config file)")
                else:
                    print(f"ℹ️  LRR learning rate remains {args.lrr_learning_rate}")
            if hasattr(args, 'lrr_weight_decay') and args.lrr_weight_decay == 0.001:  # If using default
                old_lrr_wd = args.lrr_weight_decay
                args.lrr_weight_decay = config_file['training'].get('lrr_weight_decay', args.lrr_weight_decay)
                if args.lrr_weight_decay != old_lrr_wd:
                    print(f"✅ LRR weight decay updated from {old_lrr_wd} to {args.lrr_weight_decay} (from config file)")
                else:
                    print(f"ℹ️  LRR weight decay remains {args.lrr_weight_decay}")
            if hasattr(args, 'scheduler_patience') and args.scheduler_patience == 10:  # If using default
                args.scheduler_patience = config_file['training'].get('scheduler_patience', args.scheduler_patience)
            if hasattr(args, 'early_stopping_patience') and args.early_stopping_patience == 50:  # If using default
                args.early_stopping_patience = config_file['training'].get('early_stopping_patience', args.early_stopping_patience)
                print(f"args.early_stopping_patience:{args.early_stopping_patience}")

        # Update data split parameters from config file
        if 'data_split' in config_file:
            if hasattr(args, 'split_mode') and args.split_mode == "random":  # If using default
                args.split_mode = config_file['data_split'].get('split_mode', args.split_mode)
            if hasattr(args, 'train_ratio') and args.train_ratio == 0.6:  # If using default
                args.train_ratio = config_file['data_split'].get('train_ratio', args.train_ratio)
            if hasattr(args, 'val_ratio') and args.val_ratio == 0.2:  # If using default
                args.val_ratio = config_file['data_split'].get('val_ratio', args.val_ratio)
            if hasattr(args, 'test_ratio') and args.test_ratio == 0.2:  # If using default
                args.test_ratio = config_file['data_split'].get('test_ratio', args.test_ratio)

        # Update logging parameters from config file
        if 'logging' in config_file:
            if hasattr(args, 'log_dir') and args.log_dir == "../logs":  # If using default
                args.log_dir = config_file['logging'].get('log_dir', args.log_dir)
            if hasattr(args, 'experiment_name') and args.experiment_name == "customer_ppi":  # If using default
                args.experiment_name = config_file['logging'].get('experiment_name', args.experiment_name)
            if hasattr(args, 'save_every') and args.save_every == 10:  # If using default
                args.save_every = config_file['logging'].get('save_every', args.save_every)
            if hasattr(args, 'max_checkpoints') and args.max_checkpoints == 5:  # If using default
                args.max_checkpoints = config_file['logging'].get('max_checkpoints', args.max_checkpoints)
            if hasattr(args, 'selection_metric') and args.selection_metric == "f1_micro":  # If using default
                args.selection_metric = config_file['logging'].get('selection_metric', args.selection_metric)

        # Update other parameters from config file
        if 'other' in config_file:
            if hasattr(args, 'seed') and args.seed == 42:  # If using default
                args.seed = config_file['other'].get('seed', args.seed)
                set_seed(args.seed)  # Re-set seed if changed
    else:
        # Default configuration
        config = {
            'input_dim': args.input_dim,
            'prot_hidden_dim': args.prot_hidden_dim,
            'ppi_hidden_dim': args.ppi_hidden_dim,
            'prot_num_layers': args.prot_num_layers,
            'ppi_num_layers': args.ppi_num_layers,
            'output_dim': args.output_dim,
            'dropout_ratio': args.dropout_ratio,
            'use_attention': args.use_attention,
            'num_heads': args.num_heads,
            'encoding_type': args.encoding_type,
            'encoding_config': {
                'feature_file': args.feature_file,
                'embedding_dir': args.embedding_dir,
                'validate_dims': True  # Default to True for safety
            },
            'spatial_threshold': args.spatial_threshold,
            'knn_k': args.knn_k,
            'surface_threshold': args.surface_threshold,
            'surface_distance': args.surface_distance,
            
            # Multi-encoder parameters (defaults)
            'peptide_encoder_enabled': False,
            'peptide_length_threshold': 50,
            'lrr_encoder_enabled': False,
            'lrr_annotation_file': 'customer_ppi/scripts/lrr/lrr_annotation_results.txt'
        }
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        config=config,
        use_tensorboard=False
    )
    
    logger.log(f"Using device: {device}")
    logger.log(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Load dataset
    logger.log("Loading dataset...")
    
    # Check if required files exist before proceeding
    if not os.path.exists(args.ppi_file):
        logger.log(f"Error: PPI file not found: {args.ppi_file}")
        return
    
    if not os.path.exists(args.protein_seq_file):
        logger.log(f"Error: Protein sequence file not found: {args.protein_seq_file}")
        return
    
    if not os.path.exists(args.pdb_dir):
        logger.log(f"Error: PDB directory not found: {args.pdb_dir}")
        return
    
    try:
        logger.log("Using standard PPI Dataset...")
        ppi_dataset = PPIDataset(
                config=config,
                ppi_file=args.ppi_file,
                protein_seq_file=args.protein_seq_file,
                pdb_dir=args.pdb_dir,
                cache_dir=args.cache_dir,
                balance_dataset=args.balance_dataset,
                enable_protein_positive_split=args.enable_protein_positive_split
            )
        # # Initialize dataset class based on balance_dataset flag
        # if args.balance_dataset:
        #     logger.log("Using Balanced PPI Dataset with 1:1 pos:neg ratio...")
        #     ppi_dataset = BalancedPPIDataset(
        #         config=config,
        #         ppi_file=args.ppi_file,
        #         protein_seq_file=args.protein_seq_file,
        #         pdb_dir=args.pdb_dir,
        #         cache_dir=args.cache_dir,
        #         balance_dataset=args.balance_dataset,
        #         encoding_type=args.encoding_type,
        #         feature_file=args.feature_file,
        #         embedding_dir=args.embedding_dir,
        #         spatial_threshold=args.spatial_threshold,
        #         knn_k=args.knn_k,
        #         surface_threshold=args.surface_threshold,
        #         surface_distance=args.surface_distance
        #     )
        # else:
        #     logger.log("Using standard PPI Dataset...")
            
    except FileNotFoundError as e:
        logger.log(f"❌ File not found error: {e}")
        logger.log("Please check file paths and ensure all required files exist")
        return None
    except ValueError as e:
        logger.log(f"❌ Data validation error: {e}")
        logger.log("Please check data format and content")
        return None
    except Exception as e:
        logger.log(f"❌ Unexpected error during dataset initialization: {e}")
        logger.log("Please check configuration and data files")
        return None
    
    # Split dataset
    split_dict = ppi_dataset.split_dataset(
        split_mode=args.split_mode,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed
    )
    
    logger.log(f"Train samples: {len(split_dict['train_index'])}")
    logger.log(f"Val samples: {len(split_dict['val_index'])}")
    logger.log(f"Test samples: {len(split_dict['test_index'])}")
    logger.log(f"Total PPI interactions: {len(ppi_dataset.ppi_list)}")
    
    # Initialize model with initial config
    logger.log("Initializing initial model...")
    
    # Debug: print full encoding config
    logger.log(f"Full encoding config: {config.get('encoding', {})}")
    logger.log(f"Full config keys: {list(config.keys())}")
    
    # Debug: check if config has the necessary sections
    if 'encoding' in config:
        logger.log(f"Encoding section exists: {config['encoding']}")
    else:
        logger.log("WARNING: Encoding section not found in config")
    
    if 'edge_construction' in config:
        logger.log(f"Edge construction section exists: {config['edge_construction']}")
    else:
        logger.log("WARNING: Edge construction section not found in config")
    
    # Initialize ProteinGINModelSimple
    model = ProteinGINModelSimple(config).to(device)
    
    # Encode all proteins to get initial embeddings (tensor)
    # Use model's built-in batch processing to avoid GPU OOM
    logger.log("Encoding all proteins...")
    prot_embed = model.encode_proteins(ppi_dataset.protein_dataset.graphs)
    logger.log(f"Protein embeddings type: {prot_embed}")
    
    # Verify embeddings require grad
    if not prot_embed.requires_grad:
        logger.log("⚠️ WARNING: Protein embeddings do not require grad. Weights may not be trainable.")
    
    logger.log(f"Initial model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 🔄 DYNAMIC OUTPUT DIMENSION ADJUSTMENT
    logger.log("\n🔄 Detecting actual number of classes in dataset...")
    
    # Detect actual classes present in the dataset
    actual_num_classes, actual_class_map = ppi_dataset.detect_actual_classes()
    
    # Only adjust if different from current config
    if actual_num_classes != config['output_dim']:
        logger.log(f"🔧 Adjusting model output dimension from {config['output_dim']} to {actual_num_classes}")
        
        # Update config with actual number of classes
        config['output_dim'] = actual_num_classes
        
        # Reinitialize model with adjusted output dimension
        logger.log("Reinitializing model with adjusted output dimension...")
        model = ProteinGINModelSimple(config).to(device)
        
        logger.log(f"Adjusted model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Re-encode proteins after model reinitialization
        logger.log("Re-encoding proteins after model reinitialization...")
        prot_embed = model.encode_proteins(ppi_dataset.protein_dataset.graphs)
        logger.log(f"Re-encoded protein embeddings shape: {prot_embed.shape}")
    else:
        logger.log(f"✅ Dataset has {actual_num_classes} classes, matching config output_dim")
    
    # Store the actual class mapping for later use
    config['actual_class_map'] = actual_class_map
    config['actual_num_classes'] = actual_num_classes
    
    # Validate dimensions match between PPI graph and protein embeddings
    logger.log("Validating dimensions...")
    ppi_num_nodes = ppi_dataset.ppi_graph.num_nodes()
    prot_embed_num = len(prot_embed)
    
    logger.log(f"PPI graph nodes: {ppi_num_nodes}")
    logger.log(f"Protein embeddings: {prot_embed_num}")
    
    if ppi_num_nodes != prot_embed_num:
        logger.log(f"⚠️ WARNING: Dimension mismatch! PPI graph has {ppi_num_nodes} nodes, but protein embeddings have {prot_embed_num} entries")
        logger.log(f"  Difference: {abs(ppi_num_nodes - prot_embed_num)} proteins")
        
        # Check if we need to regenerate the protein graph cache
        if ppi_num_nodes < prot_embed_num:
            logger.log("🔄 Attempting to regenerate protein graph cache...")
            
            # Clear the protein graph cache
            cache_path = os.path.join(args.cache_dir, "protein_graphs_precomputed.pkl")
            if os.path.exists(cache_path):
                logger.log(f"  Removing outdated cache: {cache_path}")
                os.remove(cache_path)
            
            # Reinitialize the dataset to force regeneration
            logger.log("  Reinitializing dataset with fresh cache...")
            ppi_dataset = PPIDataset(
                config=config,
                ppi_file=args.ppi_file,
                protein_seq_file=args.protein_seq_file,
                pdb_dir=args.pdb_dir,
                cache_dir=args.cache_dir
            )
            
            # Re-encode proteins with the new dataset
            logger.log("  Re-encoding proteins...")
            prot_embed = model.encode_proteins(ppi_dataset.protein_dataset.graphs)
            
            # Re-validate dimensions
            ppi_num_nodes = ppi_dataset.ppi_graph.num_nodes()
            prot_embed_num = len(prot_embed)
            
            logger.log(f"  After cache regeneration:")
            logger.log(f"    PPI graph nodes: {ppi_num_nodes}")
            logger.log(f"    Protein embeddings: {prot_embed_num}")
            
            if ppi_num_nodes == prot_embed_num:
                logger.log("  ✅ Cache regeneration successful! Dimensions now match.")
            else:
                logger.log(f"  ⚠️  Cache regeneration did not resolve the mismatch")
                logger.log(f"     Difference remains: {abs(ppi_num_nodes - prot_embed_num)} proteins")
                
                # Fallback to truncation
                if ppi_num_nodes < prot_embed_num:
                    logger.log("  Truncating protein embeddings to match PPI graph nodes...")
                    prot_embed = prot_embed[:ppi_num_nodes]
                    logger.log(f"  Adjusted protein embeddings shape: {prot_embed.shape}")
                else:
                    logger.log("  ❌ ERROR: PPI graph has more nodes than protein embeddings!")
                    logger.log("  This indicates a serious data inconsistency.")
                    raise ValueError(f"Dimension mismatch: PPI graph has {ppi_num_nodes} nodes, but only {prot_embed_num} protein embeddings")
        else:
            # PPI graph has more nodes than embeddings - try to regenerate cache
            logger.log("🔄 Attempting to regenerate protein graph cache...")
            
            # Clear the protein graph cache
            cache_path = os.path.join(args.cache_dir, "protein_graphs_precomputed.pkl")
            if os.path.exists(cache_path):
                logger.log(f"  Removing outdated cache: {cache_path}")
                os.remove(cache_path)
            
            # Reinitialize the dataset to force regeneration
            logger.log("  Reinitializing dataset with fresh cache...")
            ppi_dataset = PPIDataset(
                config=config,
                ppi_file=args.ppi_file,
                protein_seq_file=args.protein_seq_file,
                pdb_dir=args.pdb_dir,
                cache_dir=args.cache_dir
            )
            
            # Re-encode proteins with the new dataset
            logger.log("  Re-encoding proteins...")
            prot_embed = model.encode_proteins(ppi_dataset.protein_dataset.graphs)
            
            # Re-validate dimensions
            ppi_num_nodes = ppi_dataset.ppi_graph.num_nodes()
            prot_embed_num = prot_embed.shape[0]
            
            logger.log(f"  After cache regeneration:")
            logger.log(f"    PPI graph nodes: {ppi_num_nodes}")
            logger.log(f"    Protein embeddings: {prot_embed_num}")
            
            if ppi_num_nodes == prot_embed_num:
                logger.log("  ✅ Cache regeneration successful! Dimensions now match.")
            else:
                logger.log(f"  ⚠️  Cache regeneration did not resolve the mismatch")
                logger.log(f"     Difference remains: {abs(ppi_num_nodes - prot_embed_num)} proteins")
                
                # If PPI graph still has more nodes, this is a critical error
                if ppi_num_nodes > prot_embed_num:
                    logger.log("  ❌ ERROR: PPI graph has more nodes than protein embeddings!")
                    logger.log("  This indicates a serious data inconsistency.")
                    logger.log("  Please check:")
                    logger.log("  1. PPI data file filtering")
                    logger.log("  2. Protein graph generation")
                    logger.log("  3. Protein ID mapping consistency")
                    raise ValueError(f"Dimension mismatch: PPI graph has {ppi_num_nodes} nodes, but only {prot_embed_num} protein embeddings")
                else:
                    # PPI graph now has fewer nodes, truncate embeddings
                    logger.log("  Truncating protein embeddings to match PPI graph nodes...")
                    prot_embed = prot_embed[:ppi_num_nodes]
                    logger.log(f"  Adjusted protein embeddings shape: {prot_embed.shape}")
    else:
        logger.log("✅ Dimensions match!")
    
    # Re-split dataset after potential cache regeneration
    split_dict = ppi_dataset.split_dataset(
        split_mode=args.split_mode,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed
    )
    
    logger.log(f"Train samples: {len(split_dict['train_index'])}")
    logger.log(f"Val samples: {len(split_dict['val_index'])}")
    logger.log(f"Test samples: {len(split_dict['test_index'])}")
    logger.log(f"Total PPI interactions: {len(ppi_dataset.ppi_list)}")
    
    # Initialize training components
    # optimizer = optim.Adam(
    #     model.parameters(), 
    #     lr=args.learning_rate, 
    #     weight_decay=args.weight_decay
    # )

    # 检查编码器参数是否在优化器中
    encoder_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name.lower():
            encoder_params.append((name, param))
    
    print(f"\n{'='*60}")
    print("编码器参数检查:")
    print(f"{'='*60}")
    if encoder_params:
        for name, param in encoder_params:
            print(f"  {name}:")
            print(f"    形状: {list(param.shape)}")
            print(f"    可训练: {param.requires_grad}")
            print(f"    在优化器中: True")
    else:
        print("  ⚠️ 警告: 未找到编码器参数！")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    print(f"✅ Created optimizer with lr={args.learning_rate}, weight_decay={args.weight_decay}")
    args.lrr_weight_decay = 0
    # args.lrr_learning_rate = 0.001
    # Create separate optimizer for LRR weights if LRR encoder is enabled
    lrr_optimizer = None
    if hasattr(model, 'lrr_encoder') and model.lrr_encoder is not None:
        lrr_optimizer = optim.AdamW(
            model.lrr_encoder.parameters(),
            lr=args.lrr_learning_rate,
            weight_decay=args.lrr_weight_decay
        )
        print(f"✅ Created LRR optimizer with lr={args.lrr_learning_rate}, weight_decay={args.lrr_weight_decay}")
    #     # 2. 针对HeterogeneousHIGHPPI的参数分组
    # muon_params = []    # 用Muon优化的高维参数（≥2D）
    # adamw_params = []   # 用AdamW优化的1D参数

    # for name, param in model.named_parameters():
    #     # 筛选规则：维度≥2的权重参数（Conv/全连接权重）用Muon
    #     if param.ndim >= 2:
    #         muon_params.append(param)
    #         print(f"Muon优化参数: {name}, 维度: {param.ndim}, 形状: {param.shape}")
    #     # 1D参数（偏置、BN层参数）用AdamW
    #     else:
    #         adamw_params.append(param)
    #         print(f"AdamW优化参数: {name}, 维度: {param.ndim}, 形状: {param.shape}")

    # # 改用基础Muon类 + AdamW手动组合（避开分布式逻辑）
    #     optimizer = optim.AdamW([
    #         # 1D参数用AdamW原生优化
    #         {
    #             'params': adamw_params,
    #             'lr': args.learning_rate,
    #             'weight_decay': args.weight_decay
    #         },
    #         # 高维参数用Muon优化
    #         {
    #             'params': muon_params,
    #             'optimizer': Muon(
    #                 lr=args.learning_rate,
    #                 params = muon_params,
    #                 momentum=0.9,
    #                 weight_decay=args.weight_decay,
    #                 # nesterov=True
    #             )
    #         }
    #     ])

    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Maximize F1 score
        factor=0.5, 
        patience=args.scheduler_patience,
        verbose=True
    )
    
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    # Get labels using actual class mapping
    if hasattr(ppi_dataset, 'get_actual_labels_tensor'):
        labels = ppi_dataset.get_actual_labels_tensor(config['actual_num_classes'], config['actual_class_map'])
    else:
        # Fallback to original method
        labels = ppi_dataset.get_labels_tensor()
    
    # Calculate class weights to handle imbalance using training labels
    train_labels = labels[split_dict['train_index']]
    class_counts = torch.sum(train_labels, dim=0)
    print(f"Class counts in training data: {class_counts}")
    
    # Calculate pos weights for class imbalance
    pos_weights = (len(train_labels) - class_counts) / (class_counts + 1e-6)
    pos_weights = torch.clamp(pos_weights, max=10.0)  # Cap weights to prevent instability
    pos_weights = pos_weights.to(device)
    print(f"Positive weights: {pos_weights}")
    
    # Create BCE Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights).to(device)
    
    metrics_calc = MetricsCalculator(num_classes=config['output_dim'], logger=logger)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.join(logger.get_run_dir(), "checkpoints"),
        max_checkpoints=args.max_checkpoints
    )
    # # =============== 边类型分布检查：包含各边类型的图占比 ===============
    

    # 简单版本：使用 graph.etypes（DGL标准方式）
    from collections import Counter
    
    graph_with_edge_type = Counter()
    protein_graphs = ppi_dataset.protein_dataset.graphs
    total_graphs = len(protein_graphs)
    
    # 检查每个图的etypes
    for graph in protein_graphs:
        if hasattr(graph, 'etypes'):
            etypes = graph.etypes
            if isinstance(etypes, (list, tuple)):
                for etype in etypes:
                    graph_with_edge_type[etype] += 1
    
    # 打印分布
    print("\n" + "="*60)
    print("边类型分布统计（包含该边类型的蛋白质图占比）")
    print("="*60)
    print(f"总蛋白质图数: {total_graphs}")
    print()
    
    all_found_types = sorted(list(graph_with_edge_type.keys()))
    if all_found_types:
        print("找到的所有边类型:")
        for etype in all_found_types:
            count = graph_with_edge_type.get(etype, 0)
            percentage = count / total_graphs * 100 if total_graphs > 0 else 0
            print(f"  {etype:15s}: {count:4d} 张图 ({percentage:6.2f}%) 包含此边类型")
    else:
        print("未找到任何边类型！")
        print("建议：检查数据生成代码，确认边类型是否正确添加到图中")

    # Training loop with comprehensive error handling
    logger.log("\nStarting training...")
    start_time = time.time()
    
    best_val_metric = 0.0
    best_epoch = 0
    best_test_metrics = {}
    patience_counter = 0
    early_stop_already = True
    try:
        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.time()
            
            try:
                # Train with trainable encoder
                if epoch == 1:  # Print model parameters on first epoch
                    print_model_parameters(model)
                
                # Train and get protein embeddings (same for train/val/test)
                train_metrics = train_epoch(
                    model, ppi_dataset.protein_dataset.graphs, ppi_dataset.ppi_graph,
                    ppi_dataset.ppi_list, labels, split_dict['train_index'],
                    args.batch_size, optimizer, criterion, metrics_calc
                )
                
                # Validate (using same embeddings from training)
                val_metrics, val_preds, val_labels = evaluate(
                    model, ppi_dataset.protein_dataset.graphs, ppi_dataset.ppi_graph,
                    ppi_dataset.ppi_list, labels, split_dict['val_index'],
                    args.batch_size, criterion, metrics_calc, mode='val'
                )
                
                # Test (using same embeddings from training)
                test_metrics, test_preds, test_labels = evaluate(
                    model, ppi_dataset.protein_dataset.graphs, ppi_dataset.ppi_graph,
                    ppi_dataset.ppi_list, labels, split_dict['test_index'],
                    args.batch_size, criterion, metrics_calc, mode='test'
                )
                
                # Update learning rate
                scheduler.step(val_metrics['f1_micro'])
                
                # Log metrics
                logger.log_epoch(epoch, train_metrics, val_metrics, test_metrics)
                
                # 每10个epoch更新一次LRR权重（分离采样逻辑）
                if epoch % 50 == 0 and epoch > 0 :
                    # Get PPI indices (batched)
                    ppi_indices_batches = get_ppi_indices_for_lrr_update(
                        ppi_list=ppi_dataset.ppi_list,
                        use_all=True,  # From command line
                        max_batch_size=100 
                    )
                    
                    total_lrr_loss = 0
                    num_batches = len(ppi_indices_batches)
                    
                    logger.log(f"\n[LRRDEBUG] Epoch {epoch} - Updating LRR weights with {num_batches} batches")
                    
                    # Process each batch of PPI pairs
                    for batch_idx, ppi_indices in enumerate(ppi_indices_batches):
                        try:
                            lrr_loss = model.update_lrr_weights_gradient(
                                protein_graphs=ppi_dataset.protein_dataset.graphs,
                                ppi_g=ppi_dataset.ppi_graph,
                                ppi_list=ppi_dataset.ppi_list,
                                labels=labels,
                                loss_fn=criterion,
                                ppi_indices=ppi_indices,  # Pass this batch of indices
                                lrr_optimizer=lrr_optimizer
                            )
                            total_lrr_loss += lrr_loss
                            logger.log(f"  Batch {batch_idx+1}/{num_batches}: loss = {lrr_loss:.6f}")
                        except Exception as e:
                            logger.log(f"  ⚠️ LRR weight update failed for batch {batch_idx+1}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    avg_lrr_loss = total_lrr_loss / max(num_batches, 1)
                    logger.log(f"  Average LRR loss: {avg_lrr_loss:.6f}")
                    
                    # 记录更新后的权重
                    if hasattr(model, 'get_weights'):
                        weights = model.get_weights()
                        logger.log(f"  Updated LRR weights: {weights}")
                            
                    # except Exception as e:
                    #     logger.log(f"  ⚠️ LRR权重更新失败: {e}")
                    #     import traceback
                    #     traceback.print_exc()
                
                # Save checkpoint
                if epoch % args.save_every == 0:
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, epoch, val_metrics, config, scheduler
                    )
                
                # Check for best model
                current_val_metric = val_metrics[args.selection_metric]
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    best_epoch = epoch
                    best_test_metrics = test_metrics.copy()
                    patience_counter = 0
                    early_stop_already = False
                    
                    # Save best model
                    best_model_path = checkpoint_manager.save_checkpoint(
                        model, optimizer, epoch, val_metrics, config, scheduler, is_best=True
                    )
                    logger.log_best_model(epoch, val_metrics, best_model_path, args.selection_metric)
                    
                    # Save confusion matrices for best model
                    for class_idx in range(config['output_dim']):
                        cm = metrics_calc.calculate_confusion_matrix(test_preds, test_labels, class_idx)
                        logger.log_confusion_matrix(cm, f"class_{class_idx}", epoch, prefix="test")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= args.early_stopping_patience:
                    logger.log(f"\nEarly stopping triggered at epoch {epoch} (patience: {args.early_stopping_patience})")
                    if early_stop_already is False:
                        # for i in range(20):
                        #     # Get PPI indices (batched)
                        #     ppi_indices_batches = get_ppi_indices_for_lrr_update(
                        #         ppi_list=ppi_dataset.ppi_list,
                        #         use_all=True,  # From command line
                        #         max_batch_size=50
                        #     )
                            
                        #     total_lrr_loss = 0
                        #     num_batches = len(ppi_indices_batches)
                            
                        #     logger.log(f"\n[LRRDEBUG] Epoch {epoch} - Updating LRR weights with {num_batches} batches")
                            
                        #     # Process each batch of PPI pairs
                        #     for batch_idx, ppi_indices in enumerate(ppi_indices_batches):
                        #         try:
                        #             lrr_loss = model.update_lrr_weights_gradient(
                        #                 protein_graphs=ppi_dataset.protein_dataset.graphs,
                        #                 ppi_g=ppi_dataset.ppi_graph,
                        #                 ppi_list=ppi_dataset.ppi_list,
                        #                 labels=labels,
                        #                 loss_fn=criterion,
                        #                 ppi_indices=ppi_indices,  # Pass this batch of indices
                        #                 lrr_optimizer=lrr_optimizer
                        #             )
                        #             total_lrr_loss += lrr_loss
                        #             logger.log(f"  Batch {batch_idx+1}/{num_batches}: loss = {lrr_loss:.6f}")
                        #         except Exception as e:
                        #             logger.log(f"  ⚠️ LRR weight update failed for batch {batch_idx+1}: {e}")
                        #             import traceback
                        #             traceback.print_exc()
                        #             continue
                            
                        #     avg_lrr_loss = total_lrr_loss / max(num_batches, 1)
                        #     logger.log(f"  Average LRR loss: {avg_lrr_loss:.6f}")
                            
                        #     # 记录更新后的权重
                        #     if hasattr(model, 'get_weights'):
                        #         weights = model.get_weights()
                        #         logger.log(f"  Updated LRR weights: {weights}")
                        # patience_counter = 0
                        early_stop_already = True
                    else:
                        break
                
                epoch_time = time.time() - epoch_start
                logger.log(f"Epoch time: {epoch_time:.2f}s\n")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.log(f"❌ GPU OOM error at epoch {epoch}: {e}")
                    logger.log("Try reducing batch_size or model dimensions")
                    # Save current state before exiting
                    if epoch > 1:
                        checkpoint_manager.save_checkpoint(
                            model, optimizer, epoch-1, val_metrics, config, scheduler
                        )
                    return None
                else:
                    logger.log(f"❌ Runtime error at epoch {epoch}: {e}")
                    print("Detailed traceback:")
                    traceback.print_exc()
                    logger.log("Continuing to next epoch...")
                    continue
            except Exception as e:
                logger.log(f"❌ Unexpected error at epoch {epoch}: {e}")
                print("Detailed traceback:")
                traceback.print_exc()
                logger.log("Continuing to next epoch...")
                continue
                
    except KeyboardInterrupt:
        logger.log("\n⚠️ Training interrupted by user")
        if 'best_test_metrics' in locals() and best_test_metrics:
            logger.log(f"Best epoch was {best_epoch} with metrics: {best_test_metrics}")
        return None
    except Exception as e:
        logger.log(f"❌ Critical training error: {e}")
        logger.log("Training failed due to unexpected error")
        return None
    
        # Training summary with error handling
    try:
        total_time = time.time() - start_time
        if 'best_test_metrics' in locals() and best_test_metrics:
            logger.log_training_summary(epoch, best_epoch, best_test_metrics, total_time, model)
            logger.log("✅ Training completed successfully!")
            
            # ============ 训练结束后进行LRR分析 ============
            logger.log("\n" + "="*70)
            logger.log("开始 LRR vs Non-LRR PPI 分析...")
            logger.log("="*70)
            
            try:
                lrr_analysis_results = analyze_lrr_vs_non_lrr(
                    model=model,
                    protein_graphs=ppi_dataset.protein_dataset.graphs,
                    ppi_g=ppi_dataset.ppi_graph,
                    ppi_list=ppi_dataset.ppi_list,
                    labels=labels,
                    split_dict=split_dict,
                    batch_size=args.batch_size,
                    logger=logger,
                    device=device
                )
            except Exception as e:
                logger.log(f"⚠️ LRR分析失败: {e}")
                import traceback
                traceback.print_exc()
            # =============================================
            
        else:
            logger.log("⚠️ Training completed but no best metrics recorded")
    
    except Exception as e:
        logger.log(f"❌ Error in training summary: {e}")
    
    # Close logger
    try:
        logger.close()
    except Exception as e:
        print(f"Warning: Error closing logger: {e}")
    
    return best_test_metrics if 'best_test_metrics' in locals() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer PPI Training")
    
    # Data arguments
    parser.add_argument("--ppi_file", type=str, required=True, help="Path to PPI file")
    parser.add_argument("--protein_seq_file", type=str, required=True, help="Path to protein sequence file")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    
    # Model arguments
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--input_dim", type=int, default=7, help="Input feature dimension")
    parser.add_argument("--prot_hidden_dim", type=int, default=128, help="Protein hidden dimension")
    parser.add_argument("--ppi_hidden_dim", type=int, default=512, help="PPI hidden dimension")
    parser.add_argument("--prot_num_layers", type=int, default=4, help="Number of protein encoder layers")
    parser.add_argument("--ppi_num_layers", type=int, default=2, help="Number of PPI encoder layers")
    parser.add_argument("--output_dim", type=int, default=7, help="Output dimension")
    parser.add_argument("--dropout_ratio", type=float, default=0.0, help="Dropout ratio")
    parser.add_argument("--use_attention", action="store_true", help="Use attention mechanism")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    
    # Encoding arguments
    parser.add_argument("--encoding_type", type=str, default="mape", 
                       choices=["mape", "esm2", "precomputed", "onehot"],
                       help="Node encoding type")
    parser.add_argument("--feature_file", type=str, default=None, help="Feature file for MAPE encoding")
    parser.add_argument("--embedding_dir", type=str, default=None, help="Directory for precomputed embeddings")

    # Multi-encoder arguments
    parser.add_argument("--peptide_encoder_enabled", action="store_true", default=False,
                       help="Enable peptide encoder for small proteins")
    parser.add_argument("--peptide_length_threshold", type=int, default=50,
                       help="Length threshold for peptide classification")
    parser.add_argument("--lrr_encoder_enabled", action="store_true", default=False,
                       help="Enable LRR (Leucine-Rich Repeat) encoder")
    parser.add_argument("--lrr_annotation_file", type=str, default="customer_ppi/scripts/lrr/lrr_annotation_results.txt",
                       help="Path to LRR annotation file")
    parser.add_argument("--lrr_learning_rate", type=float, default=0.001,
                       help="Learning rate for LRR weight updates")
    parser.add_argument("--lrr_weight_decay", type=float, default=0.001,
                       help="Weight decay for LRR weight updates")

    # Edge construction arguments
    parser.add_argument("--spatial_threshold", type=float, default=10.0, help="Spatial distance threshold")
    parser.add_argument("--knn_k", type=int, default=5, help="K for KNN edges")
    parser.add_argument("--surface_threshold", type=float, default=0.2, help="Surface SASA threshold")
    parser.add_argument("--surface_distance", type=float, default=8.0, help="Surface distance threshold")
    
    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="Scheduler patience")
    parser.add_argument("--early_stopping_patience", type=int, default=50, help="Early stopping patience")
    
    # Split arguments
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "bfs", "dfs"])
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    
    # Data balancing argument (default: false, enabled with --balance_dataset)
    parser.add_argument("--balance_dataset", action="store_true",
                       default=False, 
                       help="Enable dataset balancing for imbalanced datasets (1:1 pos:neg ratio)")
    
    # Protein positive set split argument (default: false, enabled with --enable_protein_positive_split)
    parser.add_argument("--enable_protein_positive_split", action="store_true",
                       default=False,
                       help="Enable protein-level positive set splitting for balanced datasets")
    
    # Logging and checkpointing
    parser.add_argument("--log_dir", type=str, default="../logs", help="Log directory")
    parser.add_argument("--experiment_name", type=str, default="customer_ppi", help="Experiment name")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--max_checkpoints", type=int, default=5, help="Maximum checkpoints to keep")
    parser.add_argument("--selection_metric", type=str, default="f1_micro", help="Metric for model selection")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    try:
        args = parser.parse_args()
        
        # Validate arguments
        if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
            print("❌ Error: Split ratios must sum to 1.0")
            print(f"Current sum: {args.train_ratio + args.val_ratio + args.test_ratio}")
            sys.exit(1)
        
        if args.batch_size <= 0:
            print("❌ Error: Batch size must be positive")
            sys.exit(1)
        
        if args.max_epochs <= 0:
            print("❌ Error: Max epochs must be positive")
            sys.exit(1)
        
        # Run main training function
        result = main(args)
        
        if result is None:
            print("❌ Training failed or was interrupted")
            sys.exit(1)
        else:
            print("✅ Training completed successfully!")
            print(f"Best test metrics: {result}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(0)
    except argparse.ArgumentError as e:
        print(f"❌ Argument error: {e}")
        parser.print_help()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

