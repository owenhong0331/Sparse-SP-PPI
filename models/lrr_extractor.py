"""
LRR Node Extractor for Symmetric LRR Attention Model
Extracts LRR node embeddings and masks from protein graphs
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import dgl


class LRRExtractor:
    """
    Extracts LRR (Leucine-Rich Repeat) nodes from protein graphs for attention mechanisms.

    This class provides utilities to:
    - Extract LRR node embeddings from protein graphs
    - Generate LRR masks for fast indexing
    - Detect LRR presence in protein pairs
    - Handle proteins without LRR annotations
    """

    def __init__(self, lrr_edge_type: str = 'LRR_REGION', lrr_threshold: float = 0.0):
        """
        Initialize LRR Extractor.

        Args:
            lrr_edge_type: Edge type name for LRR regions in the graph
            lrr_threshold: Minimum weight threshold for LRR edges (default: 0.0, include all)
        """
        self.lrr_edge_type = lrr_edge_type
        self.lrr_threshold = lrr_threshold

    def extract_lrr_nodes(self, protein_graph: dgl.DGLGraph, node_embeddings: torch.Tensor) -> \
            Tuple[torch.Tensor, List[int], bool]:
        """
        Extract LRR node embeddings from a protein graph.

        Args:
            protein_graph: DGLGraph containing protein structure with LRR_REGION edges
            node_embeddings: Tensor of shape [num_nodes, hidden_dim] containing all node embeddings

        Returns:
            Tuple containing:
                - lrr_embeddings: Tensor of shape [num_lrr_nodes, hidden_dim] with LRR node embeddings
                - lrr_indices: List of original node indices that are LRR nodes
                - has_lrr: Boolean indicating whether the protein has LRR regions
        """
        # Check if LRR edge type exists in the graph
        if self.lrr_edge_type not in protein_graph.etypes:
            return torch.empty(0, node_embeddings.shape[1], device=node_embeddings.device), [], False

        # Get LRR edges (source, target) for the specified edge type
        lrr_edges = protein_graph.edges(etype=self.lrr_edge_type)

        # If no LRR edges exist, return empty tensors
        if len(lrr_edges[0]) == 0:
            return torch.empty(0, node_embeddings.shape[1], device=node_embeddings.device), [], False

        # Get unique source nodes (residues) involved in LRR edges
        lrr_nodes = torch.unique(lrr_edges[0]).tolist()

        # Extract embeddings for LRR nodes
        lrr_embeddings = node_embeddings[lrr_nodes]

        return lrr_embeddings, lrr_nodes, True

    def extract_lrr_mask(self, protein_graph: dgl.DGLGraph, num_nodes: int) -> \
            Tuple[torch.Tensor, bool]:
        """
        Generate a boolean mask for LRR nodes.

        This is useful for fast indexing and batch processing.

        Args:
            protein_graph: DGLGraph containing protein structure
            num_nodes: Total number of nodes in the graph

        Returns:
            Tuple containing:
                - mask: Boolean tensor of shape [num_nodes], True for LRR nodes
                - has_lrr: Boolean indicating whether LRR nodes exist
        """
        if self.lrr_edge_type not in protein_graph.etypes:
            return torch.zeros(num_nodes, dtype=torch.bool, device=protein_graph.device), False

        lrr_edges = protein_graph.edges(etype=self.lrr_edge_type)

        if len(lrr_edges[0]) == 0:
            return torch.zeros(num_nodes, dtype=torch.bool, device=protein_graph.device), False

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=protein_graph.device)
        lrr_nodes = torch.unique(lrr_edges[0])
        mask[lrr_nodes] = True

        return mask, True

    def detect_lrr_in_pair(self, protein1_graph: dgl.DGLGraph,
                          protein2_graph: dgl.DGLGraph) -> Dict:
        """
        Detect LRR presence in a protein pair.

        Args:
            protein1_graph: DGLGraph for Protein 1
            protein2_graph: DGLGraph for Protein 2

        Returns:
            Dictionary containing LRR detection statistics:
                {
                    'protein1_has_lrr': bool,
                    'protein2_has_lrr': bool,
                    'both_have_lrr': bool,
                    'neither_has_lrr': bool,  # This triggers warnings/fallbacks
                    'protein1_lrr_count': int,
                    'protein2_lrr_count': int
                }
        """
        # Check Protein 1
        if self.lrr_edge_type in protein1_graph.etypes:
            p1_edges = protein1_graph.edges(etype=self.lrr_edge_type)
            p1_has_lrr = len(p1_edges[0]) > 0
            p1_lrr_count = len(torch.unique(p1_edges[0])) if p1_has_lrr else 0
        else:
            p1_has_lrr = False
            p1_lrr_count = 0

        # Check Protein 2
        if self.lrr_edge_type in protein2_graph.etypes:
            p2_edges = protein2_graph.edges(etype=self.lrr_edge_type)
            p2_has_lrr = len(p2_edges[0]) > 0
            p2_lrr_count = len(torch.unique(p2_edges[0])) if p2_has_lrr else 0
        else:
            p2_has_lrr = False
            p2_lrr_count = 0

        return {
            'protein1_has_lrr': p1_has_lrr,
            'protein2_has_lrr': p2_has_lrr,
            'both_have_lrr': p1_has_lrr and p2_has_lrr,
            'neither_has_lrr': not p1_has_lrr and not p2_has_lrr,
            'protein1_lrr_count': p1_lrr_count,
            'protein2_lrr_count': p2_lrr_count
        }

    def get_lrr_statistics(self, protein_graph: dgl.DGLGraph) -> Dict:
        """
        Get detailed LRR statistics for a single protein.

        Args:
            protein_graph: DGLGraph for the protein

        Returns:
            Dictionary with LRR statistics
        """
        if self.lrr_edge_type not in protein_graph.etypes:
            return {
                'has_lrr': False,
                'lrr_node_count': 0,
                'lrr_edge_count': 0,
                'lrr_percentage': 0.0
            }

        lrr_edges = protein_graph.edges(etype=self.lrr_edge_type)
        num_lrr_edges = len(lrr_edges[0])

        if num_lrr_edges == 0:
            return {
                'has_lrr': False,
                'lrr_node_count': 0,
                'lrr_edge_count': 0,
                'lrr_percentage': 0.0
            }

        lrr_nodes = torch.unique(lrr_edges[0])
        num_lrr_nodes = len(lrr_nodes)
        total_nodes = protein_graph.num_nodes()

        return {
            'has_lrr': True,
            'lrr_node_count': num_lrr_nodes,
            'lrr_edge_count': num_lrr_edges,
            'lrr_percentage': (num_lrr_nodes / total_nodes) * 100 if total_nodes > 0 else 0.0
        }
