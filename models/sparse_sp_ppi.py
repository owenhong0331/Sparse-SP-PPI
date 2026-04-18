"""
Sparse-SP-PPI Model with ProteinGIN and Gradient Checkpointing
With normalized LRR weights (sum to 1)
Optimized for CUDA memory issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GINConv
import math
import numpy as np
from torch.utils.checkpoint import checkpoint
from dgl.nn.pytorch import GraphConv, GATConv, HeteroGraphConv

# # Disable PyTorch internal threading to avoid GIL issues with DGL batching
# torch.set_num_threads(1)

# Import collate function - will try to import from dataloader, fallback to simple implementation
try:
    from models.dataloader import collate_protein_graphs
except ImportError:
    # Simple fallback implementation
    def collate_protein_graphs(samples):
        """Collate a list of DGL graphs into a batch"""
#         return dgl.batch(samples)


class SparseEdgeAttentionEncoder(nn.Module):
    """
    动态 Relation Attention 版本
    接口完全兼容旧版

    保留:
        - forward(graph, node_features)
        - get_weights()
    """

    def __init__(self, config):
        super().__init__()

        self.edge_types = ["SEQ", "STR_KNN", "STR_DIS", "SURF", "LRR_REGION"]
        # self.edge_types = ["SEQ", "STR_KNN", "STR_DIS", "SURF"]
        # self.edge_types = ["SEQ", "SURF"]
        # self.edge_types = ["SEQ", "STR_KNN", "STR_DIS"]
        # self.edge_types = ["SEQ", "LRR_REGION"]
        # self.edge_types = ["SEQ", "LRR_REGION","SURF"]

        hidden_dim = config["prot_hidden_dim"]

        # 🔥 新 attention 参数
        self.relation_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),  # 降维减少参数量
            nn.LayerNorm(hidden_dim // 4),  # 稳定训练
            nn.ReLU(),
            nn.Dropout(0.3),  # 关键：小数据必须加 Dropout
            nn.Linear(hidden_dim // 4, 1),
        )

        # 【硬编码】初始 attention 偏置（让 LRR_REGION 初始权重约为0.5，其他约为0.2）
        # 计算公式：sigmoid(score / temperature) = target_weight
        # score = temperature * ln(target_weight / (1 - target_weight))
        # 0.2: score = 2 * ln(0.2/0.8) = 2 * (-1.386) = -2.772
        # 0.5: score = 2 * ln(0.5/0.5) = 2 * 0 = 0
        self.register_buffer(
            "attention_bias", torch.tensor([-4, -4, -4, -4, -2.772])
        )
        print(
            f"[LRR] Hard-coded initial attention bias: {self.attention_bias.tolist()}"
        )
        print(f"[LRR] Expected initial weights: LRR_REGION≈0.5, others≈0.2")

        # ===== 硬编码：权重激活函数配置 =====
        # 每个边独立归一化：使用 sigmoid * scale，每个边权重可 >1（增强）或 <1（衰减）
        self.use_weight_clipping = True  # 是否限制权重范围
        self.min_weight = 0.05   # 最小权重限制
        self.weight_scale = 2.0  # 放大系数：sigmoid(0,1) * 2 = (0, 2)
        self.max_weight = self.weight_scale  # 最大权重 = scale（保持一致）
        self.sigmoid_temperature = 2.0  # sigmoid温度系数，值越大输出越平缓，避免饱和

        print(f"[LRR] Weight activation config:")
        print(f"  - activation: sigmoid * {self.weight_scale} (per-edge, can be >1)")
        print(f"  - use_weight_clipping: {self.use_weight_clipping}")
        print(f"  - min_weight: {self.min_weight}")
        print(f"  - max_weight: {self.max_weight} (= weight_scale)")
        print(f"  - sigmoid_temperature: {self.sigmoid_temperature}")
        print(f"  - output range: [{self.min_weight:.2f}, {self.max_weight:.2f}]")
        # =====================================

        self.use_edge_scaling = False

        # 保存最近一次 forward 的 alpha（调试用）
        self.last_alpha = None

        # 累积所有 batch 的 alpha 值
        self.cumulative_alpha = {etype: 0.0 for etype in self.edge_types}

        # 统计 forward 调用的次数
        self.alpha_count = 0

    # =============================
    # 聚合函数（保持原结构）
    # =============================
    def _aggregate_by_edge_safe(self, graph, node_features, etype):
        if graph.num_edges(etype) == 0:
            return torch.zeros_like(node_features)

        with graph.local_scope():
            graph.ndata["h"] = node_features

            subgraph = graph[etype]

            subgraph.update_all(fn.copy_u("h", "m"), fn.mean("m", "h"))

            return subgraph.ndata["h"]

    # =============================
    # 前向传播
    # =============================
    def forward(self, graph, node_features):
        relation_embeddings = []
        graph_vectors = []
        valid_edge_types = []

        for etype in self.edge_types:
            if etype not in graph.etypes:
                continue

            agg = self._aggregate_by_edge_safe(graph, node_features, etype)

            # 边数量缩放（防止 LRR 爆炸）
            if self.use_edge_scaling:
                num_edges = max(1, graph.num_edges(etype))
                agg = agg / math.log1p(num_edges)

            relation_embeddings.append(agg)

            # 图级向量
            g = agg.mean(dim=0)
            graph_vectors.append(g)

            valid_edge_types.append(etype)

        if len(relation_embeddings) == 0:
            return torch.zeros_like(node_features)

        # [T, D]
        graph_vectors = torch.stack(graph_vectors, dim=0)

        # 🔥 attention 计算（不使用 softmax，直接使用原始分数）
        scores = self.relation_attention(graph_vectors).squeeze(-1)  # [T]

        scores = scores + self.attention_bias[: len(valid_edge_types)]

        # ===== 应用权重激活函数（每个边独立归一化）=====
        # 每个边独立归一化：使用 sigmoid * scale，允许 >1 的增强效果
        # 添加温度缩放，避免sigmoid过快饱和
        weights = torch.sigmoid(scores / self.sigmoid_temperature) * self.weight_scale

        # 限制权重范围 [min_weight, max_weight]
        if self.use_weight_clipping:
            weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        # ==========================================

        # 保存当前 batch 的权重（调试用）
        self.last_alpha = {
            valid_edge_types[i]: float(weights[i].item())
            for i in range(len(valid_edge_types))
        }

        # 累积到全局统计
        for i in range(len(valid_edge_types)):
            etype = valid_edge_types[i]
            self.cumulative_alpha[etype] += float(weights[i].item())

        # 计数器加一
        self.alpha_count += 1

        # 融合（使用未归一化的权重）
        fused = 0
        for i in range(len(relation_embeddings)):
            fused = fused + weights[i] * relation_embeddings[i]

        return fused

    # =============================
    # 保持与旧版一致的接口
    # =============================
    def get_weights(self):
        """
        返回 dict:
        {
            'SEQ': x,
            ...
        }

        返回所有数据的平均 alpha 值（而不仅仅是最后一次）
        如果还没有数据，返回均匀分布
        """
        if self.alpha_count > 0:
            # 返回所有数据的平均值
            result = {
                etype: self.cumulative_alpha[etype] / self.alpha_count
                for etype in self.edge_types
            }
            self.reset_alpha_stats()
            return result

        # 如果还没 forward 过，返回均匀分布
        uniform_weight = 1.0 / len(self.edge_types)
        return {etype: 0.0 for etype in self.edge_types}

    def reset_alpha_stats(self):
        """
        重置 alpha 累积统计

        使用场景：
        - 每个 epoch 开始时
        - 验证集和训练集切换时
        - 需要重新统计时

        示例：
            # 在每个 epoch 开始时调用
            for epoch in range(num_epochs):
                model.sparse_edge_attention_encoder.reset_alpha_stats()
                # ... 训练代码 ...
        """
        self.cumulative_alpha = {etype: 0.0 for etype in self.edge_types}
        self.alpha_count = 0
        self.last_alpha = None
        print(f"[LRRDEBUG] Alpha stats reset")


# class SparseEdgeAttentionEncoder(nn.Module):
#     """
#     可训练的LRR编码器，直接使用可训练权重（无归一化）
#     """
#     def __init__(self, config):
#         super(SparseEdgeAttentionEncoder, self).__init__()

#         # 可训练的原始权重参数（直接作为权重使用，不归一化）
#         # 注意：这些初始值现在就是实际的权重值，而非 logits
#         # 如果您希望初始权重相近，可以设置相近的值
#         self.lrr_logits = nn.Parameter(torch.tensor([
#             0.1,   # SEQ: 初始权重 0.2 (或其他期望的初始值)
#             0.05,   # STR_KNN: 初始权重 0.2
#             0.05,   # STR_DIS: 初始权重 0.2
#             0.05,   # SURF: 初始权重 0.2
#             0.85    # LRR_REGION: 初始权重 0.2
#         ], dtype=torch.float)) # 明确指定数据类型

#         # self.lrr_logits = nn.Parameter(torch.tensor([
#         #     1,   # SEQ: 初始权重 0.2 (或其他期望的初始值)
#         #     1,   # STR_KNN: 初始权重 0.2
#         #     1,   # STR_DIS: 初始权重 0.2
#         #     1,   # SURF: 初始权重 0.2
#         #     1    # LRR_REGION: 初始权重 0.2
#         # ], dtype=torch.float)) # 明确指定数据类型

#         # 或者，如果您想保持原来的logits对应的softmax值作为初始权重：
#         # original_logits = torch.tensor([6.157855, -9.960241, -9.960241, 5.464707, 8.297921])
#         # initial_weights = F.softmax(original_logits, dim=0)
#         # self.lrr_logits = nn.Parameter(initial_weights.clone().detach().float())

#         self.edge_types = ['SEQ', 'STR_KNN', 'STR_DIS', 'SURF', 'LRR_REGION']

#     def get_weights(self):
#         """获取当前LRR权重值（未经归一化）"""
#         # 直接返回参数的值
#         weights = {}
#         for i, name in enumerate(self.edge_types):
#             weights[name] = self.lrr_logits[i].item()
#         return weights

#     def forward(self, graph, node_features):
#         """
#         前向传播，直接使用权重参数
#         """
#         # 直接使用权重参数
#         weights_raw = self.lrr_logits # 不再调用 F.softmax

#         # 初始化结果列表（避免原地操作）
#         results = []

#         # 按顺序应用每种边类型的权重
#         for i, etype in enumerate(self.edge_types):
#             if etype in graph.etypes:
#                 weight = weights_raw[i] # 直接取对应的权重值
#                 try:
#                     agg = self._aggregate_by_edge_safe(graph, node_features, etype)
#                     num_edges = graph.num_edges(etype)
#                     # 防止除0
#                     num_edges = max(1, num_edges)

#                     # 推荐使用 log 缩放（更平滑）
#                     scale = math.log1p(num_edges)

#                     # agg = agg / scale
#                     results.append(weight * agg)
#                 except RuntimeError as e:
#                     if "INTERNAL ASSERT FAILED" in str(e) or "CUDA" in str(e):
#                         print(f"CUDA Error in LRR aggregation for {etype}, using fallback...")
#                         # CPU fallback
#                         graph_cpu = graph.cpu()
#                         node_features_cpu = node_features.cpu()
#                         agg_cpu = self._aggregate_by_edge_safe(graph_cpu, node_features_cpu, etype)
#                         num_edges = graph.num_edges(etype)
#                         # 防止除0
#                         num_edges = max(1, num_edges)

#                         # 推荐使用 log 缩放（更平滑）
#                         scale = math.log1p(num_edges)

#                         agg_cpu = agg_cpu / scale
#                         results.append(weight * agg_cpu.to(node_features.device))
#                     else:
#                         raise e

#         # 合并所有结果（使用sum而不是原地加法）
#         if results:
#             # 将所有加权聚合结果相加
#             result = torch.stack(results).sum(dim=0)
#         else:
#             # 如果没有结果，返回零张量
#             result = torch.zeros_like(node_features)
#             if node_features.requires_grad:
#                 result.requires_grad_(True)

#         return result

#     def _aggregate_by_edge_safe(self, graph, node_features, edge_type):
#         """边类型聚合 - 安全版本"""
#         try:
#             subgraph = graph.edge_type_subgraph([edge_type])

#             if subgraph.num_edges() == 0:
#                 result = torch.zeros_like(node_features)
#                 if node_features.requires_grad:
#                     result.requires_grad_(True)
#                 return result

#             # 确保subgraph和node_features在同一设备
#             device = node_features.device
#             if subgraph.device != device:
#                 subgraph = subgraph.to(device)

#             # 消息传递 - 使用更安全的方式
#             src_features = node_features[subgraph.srcnodes()]
#             subgraph.srcdata['h'] = src_features

#             # 使用更安全的聚合方法
#             subgraph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))

#             # 结果分配（保留梯度信息）- 使用非原地操作
#             base_result = torch.zeros_like(node_features)
#             if node_features.requires_grad:
#                 base_result.requires_grad_(True)

#             dst_nodes = subgraph.dstnodes()
#             neigh_features = subgraph.dstdata['h_neigh']

#             # 使用非原地操作index_add（没有下划线）来避免in-place错误
#             if len(dst_nodes) > 0 and len(neigh_features) > 0:
#                 result = base_result.index_add(0, dst_nodes.to(base_result.device), neigh_features.to(base_result.device))
#             else:
#                 result = base_result

#             return result

#         except RuntimeError as e:
#             if "INTERNAL ASSERT FAILED" in str(e) or "CUDA" in str(e):
#                 # 如果CUDA操作失败，返回零张量
#                 print(f"CUDA Error in aggregation for {edge_type}, returning zeros")
#                 return torch.zeros_like(node_features)
#             else:
#                 raise e

# class SparseEdgeAttentionEncoder(nn.Module):
#     """
#     可训练的LRR编码器，权重归一化（总和为1）
#     """
#     def __init__(self, config):
#         super(SparseEdgeAttentionEncoder, self).__init__()

#         # 可训练的原始权重参数（使用Logits）
#         # 所有边类型初始权重均等，让模型从零开始学习各自的 importance
#         # 使用相同的 logit 值 (0.0)，确保 softmax 后所有权重完全相等 (~0.2 each)
#         # LRR_REGION, SURF, STR_DIS, STR_KNN, SEQ 初始权重全部均等
#         self.lrr_logits = nn.Parameter(torch.tensor([
#             0.0,   # SEQ: 初始权重 ~0.2 (与其他边类型均等)
#             0.0,   # STR_KNN: 初始权重 ~0.2 (与SURF、STR_DIS、LRR均等)
#             0.0,   # STR_DIS: 初始权重 ~0.2 (与STR_KNN、SURF、LRR均等)
#             0.0,   # SURF: 初始权重 ~0.2 (与STR_KNN、STR_DIS均等)
#             0.0,    # LRR_REGION: 初始权重 ~0.2 (所有边类型均等)
#         ]))
#         self.lrr_logits = nn.Parameter(torch.tensor([
#             0.1,   # SEQ: 初始权重 ~0.2 (与其他边类型均等)
#             0.05,   # STR_KNN: 初始权重 ~0.2 (与SURF、STR_DIS、LRR均等)
#             0.05,   # STR_DIS: 初始权重 ~0.2 (与STR_KNN、SURF、LRR均等)
#             0.05,   # SURF: 初始权重 ~0.2 (与STR_KNN、STR_DIS均等)
#             0.85,    # LRR_REGION: 初始权重 ~0.2 (所有边类型均等)
#         ]))
#         self.edge_types = ['SEQ', 'STR_KNN', 'STR_DIS', 'SURF', 'LRR_REGION']

#     def get_weights(self):
#         """获取当前LRR权重值（经过softmax归一化）"""
#         # weights_raw = F.softmax(self.lrr_logits, dim=0)
#         weights_raw = self.lrr_logits
#         weights = {}
#         for i, name in enumerate(self.edge_types):
#             weights[name] = weights_raw[i].item()
#         return weights

#     def forward(self, graph, node_features):
#         """
#         前向传播，权重经过softmax归一化
#         """
#         # 计算归一化的权重
#         weights_raw = self.lrr_logits

#         # 初始化结果列表（避免原地操作）
#         results = []

#         # 按顺序应用每种边类型的权重
#         for i, etype in enumerate(self.edge_types):
#             if etype in graph.etypes:
#                 weight = weights_raw[i]
#                 try:
#                     agg = self._aggregate_by_edge_safe(graph, node_features, etype)
#                     results.append(weight * agg)
#                 except RuntimeError as e:
#                     if "INTERNAL ASSERT FAILED" in str(e) or "CUDA" in str(e):
#                         print(f"CUDA Error in LRR aggregation for {etype}, using fallback...")
#                         # CPU fallback
#                         graph_cpu = graph.cpu()
#                         node_features_cpu = node_features.cpu()
#                         agg_cpu = self._aggregate_by_edge_safe(graph_cpu, node_features_cpu, etype)
#                         results.append(weight * agg_cpu.to(node_features.device))
#                     else:
#                         raise e

#         # 合并所有结果（使用sum而不是原地加法）
#         if results:
#             # 将所有加权聚合结果相加
#             result = torch.stack(results).sum(dim=0)
#         else:
#             # 如果没有结果，返回零张量
#             result = torch.zeros_like(node_features)
#             if node_features.requires_grad:
#                 result.requires_grad_(True)

#         return result

#     def _aggregate_by_edge_safe(self, graph, node_features, edge_type):
#         """边类型聚合 - 安全版本"""
#         try:
#             subgraph = graph.edge_type_subgraph([edge_type])

#             if subgraph.num_edges() == 0:
#                 result = torch.zeros_like(node_features)
#                 if node_features.requires_grad:
#                     result.requires_grad_(True)
#                 return result

#             # 确保subgraph和node_features在同一设备
#             device = node_features.device
#             if subgraph.device != device:
#                 subgraph = subgraph.to(device)

#             # 消息传递 - 使用更安全的方式
#             src_features = node_features[subgraph.srcnodes()]
#             subgraph.srcdata['h'] = src_features

#             # 使用更安全的聚合方法
#             subgraph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))

#             # 结果分配（保留梯度信息）- 使用非原地操作
#             base_result = torch.zeros_like(node_features)
#             if node_features.requires_grad:
#                 base_result.requires_grad_(True)

#             dst_nodes = subgraph.dstnodes()
#             neigh_features = subgraph.dstdata['h_neigh']

#             # 使用非原地操作index_add（没有下划线）来避免in-place错误
#             if len(dst_nodes) > 0 and len(neigh_features) > 0:
#                 result = base_result.index_add(0, dst_nodes.to(base_result.device), neigh_features.to(base_result.device))
#             else:
#                 result = base_result

#             return result

#         except RuntimeError as e:
#             if "INTERNAL ASSERT FAILED" in str(e) or "CUDA" in str(e):
#                 # 如果CUDA操作失败，返回零张量
#                 print(f"CUDA Error in aggregation for {edge_type}, returning zeros")
#                 return torch.zeros_like(node_features)
#             else:
#                 raise e


class ProteinGINModelSimple(nn.Module):
    """
    Simplified Protein GIN Model with trainable LRR encoding
    Optimized for GPU memory efficiency using gradient checkpointing
    """

    def __init__(self, config):
        super(ProteinGINModelSimple, self).__init__()

        self.config = config
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["prot_hidden_dim"]
        self.ppi_hidden_dim = config["ppi_hidden_dim"]
        self.num_layers = config["prot_num_layers"]
        self.ppi_num_layers = config["ppi_num_layers"]
        self.dropout = nn.Dropout(config["dropout_ratio"])
        self.output_dim = config["output_dim"]

        # LRR编码器（可训练，权重归一化）
        self.sparse_edge_attention_encoder = SparseEdgeAttentionEncoder(config)

        print(f"ProteinGINModelSimple initialized:")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  PPI hidden dim: {self.ppi_hidden_dim}")
        print(f"  Number of layers: {self.num_layers}")
        print(f"  Output dim: {self.output_dim}")

        # 打印初始权重
        initial_weights = self.sparse_edge_attention_encoder.get_weights()
        print(f"  Initial normalized LRR weights:")
        total_weight = 0.0
        for name, weight in initial_weights.items():
            print(f"    {name}: {weight:.6f}")
            total_weight += weight
        print(f"    Total: {total_weight:.6f}")

        # self.ppi_layers = nn.ModuleList()

        # # 第一层

        # mlp = nn.Sequential(

        #     nn.Linear(self.hidden_dim, self.ppi_hidden_dim),

        #     nn.BatchNorm1d(self.ppi_hidden_dim),

        #     nn.ReLU(),

        #     nn.Linear(self.ppi_hidden_dim, self.ppi_hidden_dim),

        #     nn.BatchNorm1d(self.ppi_hidden_dim),

        #     nn.ReLU()

        # )

        # self.ppi_layers.append(GINConv(mlp, 'sum'))

        # # 后续层

        # for i in range(self.ppi_num_layers - 1):

        #     mlp = nn.Sequential(

        #         nn.Linear(self.ppi_hidden_dim, self.ppi_hidden_dim),

        #         nn.BatchNorm1d(self.ppi_hidden_dim),

        #         nn.ReLU(),

        #         nn.Linear(self.ppi_hidden_dim, self.ppi_hidden_dim),

        #         nn.BatchNorm1d(self.ppi_hidden_dim),

        #         nn.ReLU()

        #     )

        #     self.ppi_layers.append(GINConv(mlp, 'sum'))

        # self.ppi_layers = nn.ModuleList()

        # self.ppi_layers.append(HeteroGraphConv({
        #     'interacts': GraphConv(self.hidden_dim,self.ppi_hidden_dim, allow_zero_in_degree=True)
        # }, aggregate='sum'))
        # # Subsequent layers
        # for i in range(self.ppi_num_layers - 1):
        #     self.ppi_layers.append(HeteroGraphConv({
        #         'interacts': GraphConv(self.ppi_hidden_dim, self.ppi_hidden_dim, allow_zero_in_degree=True)
        #     }, aggregate='sum'))
        # self.norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers)])

        # 初始化部分 (与代码1完全一致)
        self.ppi_layers = nn.ModuleList()

        # 第一层 (参数: input_dim=hidden_dim, output_dim=ppi_hidden_dim)
        self.ppi_layers.append(
            GraphConv(self.hidden_dim, self.ppi_hidden_dim, allow_zero_in_degree=True)
        )

        # 后续层 (参数: input_dim=ppi_hidden_dim, output_dim=ppi_hidden_dim)
        for i in range(self.ppi_num_layers - 1):
            self.ppi_layers.append(
                GraphConv(
                    self.ppi_hidden_dim, self.ppi_hidden_dim, allow_zero_in_degree=True
                )
            )

        # 归一化层 (与代码1完全一致)
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers)]
        )

        # # 交互预测头 (与代码1完全一致)
        # self.interaction_head = nn.Sequential(
        #     nn.Linear(self.ppi_hidden_dim, self.ppi_hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(config['dropout_ratio']),
        #     nn.Linear(self.ppi_hidden_dim, self.output_dim)
        # )

        # Interaction prediction head
        self.interaction_head = nn.Sequential(
            nn.Linear(self.ppi_hidden_dim, self.ppi_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config["dropout_ratio"]),
            nn.Linear(self.ppi_hidden_dim, self.output_dim),
        )

    def get_weights(self):
        """获取当前LRR权重值"""
        return self.sparse_edge_attention_encoder.get_weights()

    def _graph_pooling(self, graph, node_embeds):
        """图级别池化"""
        if hasattr(graph, "batch_num_nodes"):
            # 批处理图
            batch_num_nodes = graph.batch_num_nodes()
            graph_embeds = []
            start_idx = 0

            for num_nodes in batch_num_nodes:
                end_idx = start_idx + num_nodes
                graph_embed = node_embeds[start_idx:end_idx].mean(dim=0)
                graph_embeds.append(graph_embed)
                start_idx = end_idx

            return torch.stack(graph_embeds)
        else:
            # 单图
            return node_embeds.mean(dim=0, keepdim=True)

    def encode_proteins(self, protein_graphs, graph_types=None, retain_grad=False):
        """
        编码所有蛋白质图（仅支持同质图）
        使用CUDA内存优化

        Args:
            protein_graphs: 蛋白质图列表或字典
            graph_types: 图类型（已废弃，保持兼容性）
            retain_grad: 是否保留梯度（用于LRR权重更新，默认False以节省内存）
        """
        if isinstance(protein_graphs, dict):
            # 批处理图字典 - 只处理第一个非空图
            for graph_type, batch_graph in protein_graphs.items():
                if batch_graph is not None:
                    _, graph_embeds = self._encode_proteins_single(
                        batch_graph, graph_type
                    )
                    return graph_embeds
            raise ValueError("No valid protein graphs found")

        elif isinstance(protein_graphs, list):
            # 图列表 - 批量处理以提高速度
            print("encoding 1")
            protein_embeds = []
            batch_size = 64  # 每批处理32个图，可调整

            num_proteins = len(protein_graphs)
            num_batches = (num_proteins + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_proteins)
                batch_graphs = protein_graphs[start_idx:end_idx]

                try:
                    # 将一批图合并成一个大的批处理图
                    device = next(self.parameters()).device
                    batch_graph = dgl.batch(batch_graphs).to(device)

                    # 获取节点特征
                    node_features = batch_graph.ndata["x"].to(device)

                    # 只在需要时保留梯度
                    if retain_grad and not node_features.requires_grad:
                        node_features.requires_grad_(True)

                    # 一次性编码整个批次
                    node_embeds = self.sparse_edge_attention_encoder(batch_graph, node_features)

                    # 批量池化得到图嵌入
                    batch_graph_embeds = self._graph_pooling(batch_graph, node_embeds)

                    # 根据retain_grad参数决定是否detach
                    if retain_grad:
                        # 保留梯度（用于LRR权重更新）
                        graph_embeds_cpu = batch_graph_embeds.cpu()
                    else:
                        # 断开梯度，节省内存（正常训练）
                        graph_embeds_cpu = batch_graph_embeds.detach().cpu()

                    protein_embeds.append(graph_embeds_cpu)

                except RuntimeError as e:
                    if "INTERNAL ASSERT FAILED" in str(e) or "CUDA" in str(e):
                        print(
                            f"CUDA Error in batch {batch_idx}, falling back to individual processing..."
                        )

                        # CPU回退方案：逐个处理这批图
                        fallback_embeds = []
                        target_device = next(self.parameters()).device
                        for graph in batch_graphs:
                            graph_cpu = graph.cpu()
                            node_features_cpu = graph_cpu.ndata["x"].cpu()

                            with torch.no_grad():
                                node_embeds_cpu = self.sparse_edge_attention_encoder(
                                    graph_cpu, node_features_cpu
                                )

                            graph_embed_cpu = self._graph_pooling(
                                graph_cpu, node_embeds_cpu
                            )
                            # 移回目标设备（GPU）以匹配其他批次
                            fallback_embeds.append(graph_embed_cpu.to(target_device))

                        batch_embeds = torch.cat(fallback_embeds, dim=0)
                        # 根据retain_grad参数决定是否detach
                        if not retain_grad:
                            batch_embeds = batch_embeds.detach()
                        protein_embeds.append(batch_embeds)

                    else:
                        raise e
                finally:
                    # 清理GPU内存
                    del batch_graph, node_features
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # if batch_idx % 10 == 0:
                #     print(f"    Batch {batch_idx}: processed {len(batch_graphs)} graphs")

            # 合并所有批次的embeddings
            if len(protein_embeds) == 0:
                raise ValueError("No protein embeddings generated!")

            # 将最终结果移到模型所在设备
            device = next(self.parameters()).device
            result = torch.cat(protein_embeds, dim=0).to(device)

            return result

        else:
            # 单个图 - 直接编码
            _, graph_embed = self._encode_proteins_single(protein_graphs, "standard")
            return graph_embed

    def _encode_single_step(self, graph, node_features):
        """用于检查点的单步编码函数"""
        return self.sparse_edge_attention_encoder(graph, node_features)

    def _encode_proteins_single(self, graph, graph_type):
        """编码单个蛋白质图"""
        # 将graph移到模型所在设备（GPU）进行计算
        device = next(self.parameters()).device
        if graph.device != device:
            graph = graph.to(device)

        node_features = graph.ndata["x"]

        # 直接调用编码器（保持梯度流）
        node_embeds = self.sparse_edge_attention_encoder(graph, node_features)

        # 图级别嵌入：平均池化
        graph_embed = self._graph_pooling(graph, node_embeds)

        return node_embeds, graph_embed

    def encode_ppi(self, ppi_g, node_features):
        """编码PPI网络（仅同质图），无梯度检查点"""
        x = node_features

        for l, layer in enumerate(self.ppi_layers):
            # try:
            #     # 不使用梯度检查点
            #     x = layer(ppi_g, x)
            # except RuntimeError as e:
            #     if "INTERNAL ASSERT FAILED" in str(e) or "CUDA" in str(e):
            #         print(f"CUDA Error in PPI layer {l}, trying CPU fallback...")
            #         # CPU 回退
            #         ppi_g_cpu = ppi_g.cpu()
            #         x_cpu = x.cpu()
            #         x_cpu = layer(ppi_g_cpu, x_cpu)
            #         x = x_cpu.to(x.device)
            #     else:
            #         raise e

            # if l != len(self.ppi_layers) - 1:
            #     x = self.dropout(F.relu(x))
            # For homogeneous graphs, use standard GraphConv
            if hasattr(layer, "mods") and "interacts" in layer.mods:
                # This is a HeteroGraphConv layer, extract the underlying GraphConv
                conv_layer = layer.mods["interacts"]
                x = conv_layer(ppi_g, x)
            else:
                # Fallback: direct GraphConv
                x = layer(ppi_g, x)

            x = self.norms[l](F.relu(x))
            if l != self.num_layers - 1:
                x = self.dropout(x)

        return x

    def _ppi_layer_forward(self, layer, g, x):
        """用于检查点的PPI层前向传播函数"""
        return layer(g, x)

    def predict_interactions(self, prot_embed_updated, ppi_list, idx):
        """预测蛋白质交互（仅同质图）"""
        node_id = np.array(ppi_list)[idx]
        x1 = prot_embed_updated[node_id[:, 0]]
        x2 = prot_embed_updated[node_id[:, 1]]

        # Interaction prediction (element-wise multiplication)
        x = torch.mul(x1, x2)
        output = self.interaction_head(x)

        return output

    def forward(self, ppi_g, prot_embed, ppi_list, idx):
        """
        Forward pass for PPI prediction (only homogeneous graphs)

        Args:
            retain_grad: 是否保留蛋白质编码的梯度（用于LRR权重更新）
        """
        # 如果prot_embed是蛋白质图，则先编码
        if isinstance(prot_embed, (list, dict)) or (
            hasattr(prot_embed, "ndata") and "x" in prot_embed.ndata
        ):
            prot_embed = self.encode_proteins(prot_embed)

        # 获取设备
        device = next(self.parameters()).device

        # 确保所有张量都在正确的设备上
        prot_embed = prot_embed.to(device)

        # 编码PPI网络
        prot_embed_updated = self.encode_ppi(ppi_g, prot_embed)

        # 预测交互
        output = self.predict_interactions(prot_embed_updated, ppi_list, idx)

        return output

    def update_lrr_weights_gradient(
        self,
        protein_graphs,
        ppi_g,
        ppi_list,
        labels,
        loss_fn,
        ppi_indices,
        lrr_optimizer,
        batch_size=32,
        lrr_weight_reg=0.1,
    ):
        """
        Update LRR weights using pre-selected PPI pairs (batched)

        Args:
            lrr_weight_reg: L2正则化系数，防止LRR权重过小
        """

        # 1. 仅启用 LRR 参数的梯度，其他参数冻结
        # 保存原始requires_grad状态
        original_requires_grad = {}
        for name, param in self.named_parameters():
            original_requires_grad[name] = param.requires_grad
            if "relation_attention" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 3. 使用传入的 PPI 索引（不再内部采样）
        sample_indices = ppi_indices

        # 4. 找出涉及的蛋白质（最多 200 个）
        involved_proteins = set()
        for idx in sample_indices:
            p1, p2 = ppi_list[idx]
            involved_proteins.add(p1)
            involved_proteins.add(p2)

        involved_proteins = list(involved_proteins)
        num_proteins = len(involved_proteins)

        # 创建蛋白质 ID 到索引的映射
        protein_id_to_idx = {pid: i for i, pid in enumerate(involved_proteins)}

        # 5. 分批编码（batch_size=1 时实际为逐个处理）
        device = next(self.parameters()).device
        protein_embeds = []  # 保存所有蛋白质嵌入（GPU 张量）

        num_batches = (num_proteins + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_proteins)
            batch_protein_ids = involved_proteins[start_idx:end_idx]

            batch_subgraphs = [protein_graphs[i] for i in batch_protein_ids]

            # ===== 重要修改：所有操作在 GPU 上完成 =====
            batch_embeds = []
            for graph in batch_subgraphs:
                # 确保图在 GPU 上（避免 CPU-GPU 数据迁移）
                graph = graph.to(device)
                node_features = graph.ndata["x"].to(device)  # 保持在 GPU

                # ✅ 关键：无需设置 node_features.requires_grad (输入特征不应有梯度)
                # 原代码错误：设置 node_features.requires_grad 会引入额外计算图
                # 删除此行：node_features.requires_grad_(True)

                # 编码（保留梯度，计算图在 GPU 上）
                node_embeds = self.sparse_edge_attention_encoder(graph, node_features)
                graph_embed = self._graph_pooling(graph, node_embeds)
                batch_embeds.append(graph_embed)

            # 合并批次嵌入（GPU 上操作，无需 .cpu()）
            batch_protein_embeds = torch.cat(batch_embeds, dim=0)
            protein_embeds.append(batch_protein_embeds)

            # 清理临时变量（释放 GPU 显存）
            del batch_subgraphs, batch_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 合并所有嵌入（GPU 张量，无需 .to(device)）
        protein_embeds = torch.cat(protein_embeds, dim=0)  # 保持在 GPU

        # 6. 调整采样 PPI 对的索引
        adjusted_ppi_list = []
        for orig_idx in sample_indices:
            p1, p2 = ppi_list[orig_idx]
            adjusted_ppi_list.append([protein_id_to_idx[p1], protein_id_to_idx[p2]])

        # 6.5 创建子图：只包含涉及的蛋白质节点
        # 从完整的 ppi_g 中提取子图
        involved_protein_indices = torch.tensor(
            involved_proteins, dtype=torch.long, device=device
        )
        sub_ppi_g = dgl.node_subgraph(ppi_g, involved_protein_indices)
        sub_ppi_g = sub_ppi_g.to(device)

        # 在子图上运行 PPI 编码
        protein_embeds = self.encode_ppi(sub_ppi_g, protein_embeds)
        # 7. 计算损失（所有操作在 GPU 上）
        output = self.predict_interactions(
            protein_embeds, adjusted_ppi_list, list(range(len(adjusted_ppi_list)))
        )
        sample_labels = labels[sample_indices]

        # 8. 反向传播更新 LRR 权重
        lrr_optimizer.zero_grad()
        loss = loss_fn(output, sample_labels)

        # 正则化已移除：让 LRR 权重自由学习
        # if lrr_weight_reg > 0 and hasattr(self.sparse_edge_attention_encoder, 'last_alpha'):
        #     current_weights = self.sparse_edge_attention_encoder.last_alpha
        #     if current_weights and 'LRR_REGION' in current_weights:
        #         lrr_weight = current_weights['LRR_REGION']
        #         target_weight = 0.4
        #         reg_loss = lrr_weight_reg * (lrr_weight - target_weight) ** 2
        #         loss = loss + reg_loss

        loss.backward()
        lrr_optimizer.step()
        print(
            f"[LRRDEBUG] Updated LRR weights using {len(sample_indices)} sampled PPI pairs, loss: {loss.item():.6f}"
        )

        # 9. 清理显存（反向传播后）
        del protein_embeds, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 10. 恢复所有参数的原始requires_grad状态
        for name, param in self.named_parameters():
            param.requires_grad = original_requires_grad[name]

        return loss.item()


class ExplainableProteinGINModel(ProteinGINModelSimple):
    """
    ProteinGINModelSimple with explainability features (Grad-WAM)
    """

    def __init__(self, config):
        super(ExplainableProteinGINModel, self).__init__(config)

        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_with_attention(self, ppi_g, prot_embed, ppi_list, idx):
        """
        Forward pass that saves activations for explainability
        """
        # Register hook to save gradients
        if isinstance(prot_embed, torch.Tensor):
            prot_embed.register_hook(self.save_gradient)
            self.activations = prot_embed

        return self.forward(ppi_g, prot_embed, ppi_list, idx)

    def get_grad_wam(self, target_idx):
        """
        Compute Gradient Weighted Activation Mapping
        Returns importance scores for each amino acid
        """
        if self.gradients is None or self.activations is None:
            raise ValueError("Must call forward_with_attention first")

        # Compute weights from gradients
        weights = torch.mean(self.gradients[target_idx], dim=0)

        # Compute weighted activation map
        grad_cam = torch.mul(self.activations[target_idx], weights)

        return grad_cam


def print_model_parameters(model):
    """
    打印模型的参数统计信息，特别关注 LRR 参数
    """
    print(f"\n{'=' * 60}")
    print("模型参数统计")
    print(f"{'=' * 60}")

    total_params = 0
    trainable_params = 0
    lrr_params = 0
    lrr_trainable_params = 0

    # 统计所有参数
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        if param.requires_grad:
            trainable_params += param_count

        # 检查是否是LRR参数
        if "lrr_logits" in name:
            lrr_params += param_count
            if param.requires_grad:
                lrr_trainable_params += param_count

            print(f"  {name}:")
            print(f"    形状: {list(param.shape)}")
            print(f"    数量: {param_count:,}")
            print(f"    可训练: {param.requires_grad}")
            if param.requires_grad:
                # 显示softmax后的权重
                softmax_weights = F.softmax(param, dim=0)
                print(f"    原始值: {param.tolist()}")
                print(f"    Softmax权重: {softmax_weights.tolist()}")

    print(f"\n总体统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  LRR参数: {lrr_params:,}")
    print(f"  LRR可训练参数: {lrr_trainable_params:,}")
    print(f"  LRR冻结参数: {lrr_params - lrr_trainable_params:,}")

    # 如果是 LRR 参数，打印权重信息
    if hasattr(model, "get_weights"):
        print(f"\nLRR 当前权重 (总和为1):")
        weights = model.get_weights()
        total_weight = 0.0
        for etype, weight in weights.items():
            print(f"  {etype}: {weight:.6f}")
            total_weight += weight
        print(f"  总和: {total_weight:.6f}")

    print(f"{'=' * 60}\n")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lrr_params": lrr_params,
        "lrr_trainable_params": lrr_trainable_params,
    }


def test_normalized_weights():
    """测试权重归一化功能"""
    # 创建配置
    config = {
        "input_dim": 21,
        "prot_hidden_dim": 128,
        "ppi_hidden_dim": 64,
        "prot_num_layers": 2,
        "ppi_num_layers": 2,
        "dropout_ratio": 0.1,
        "output_dim": 1,
    }

    # 创建模型
    model = ProteinGINModelSimple(config)

    # 测试权重归一化
    print("测试权重归一化:")
    weights = model.get_weights()
    total = sum(weights.values())
    print(f"权重: {weights}")
    print(f"总和: {total:.6f}")

    # 验证总和确实为1
    assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1, but got {total}"
    print("✓ 权重归一化验证通过!")


if __name__ == "__main__":
    test_normalized_weights()
