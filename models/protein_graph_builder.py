"""
Protein Graph Builders with Multiple Encoding Strategies
Supports standard, peptide, and LRR-enhanced graph construction
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import dgl
import torch

from .edge_construction import build_all_edges, read_ca_atoms_from_pdb
from .node_encoding import get_encoder
from .lrr_parser import LRRDatabase


class ProteinGraphBuilder:
    """Base class for building protein graphs"""
    
    def __init__(self, config: Dict, pdb_dir: str):
        """
        Initialize protein graph builder
        
        Args:
            config: Configuration dictionary
            pdb_dir: Directory containing PDB files
        """
        self.config = config
        self.pdb_dir = pdb_dir
        
        # Edge construction parameters
        self.spatial_threshold = config.get('spatial_threshold', 10.0)
        self.knn_k = config.get('knn_k', 5)
        self.surface_threshold = config.get('surface_threshold', 0.2)
        self.surface_distance = config.get('surface_distance', 8.0)
        
        # Encoding configuration
        self.encoding_type = config.get('encoding_type', 'precomputed')
        self.encoding_config = config.get('encoding_config', {})
        
        # Initialize encoder (delay initialization for precomputed encoders)
        self.encoder = None
        self.encoding_config_copy = self.encoding_config.copy()
        self.encoding_config_copy.pop('encoding_type', None)  # Remove if exists
    
    def build_graph(self, protein_id: str, residue_names: List[str], coords: np.ndarray) -> Optional[dgl.DGLGraph]:
        """
        Build protein graph (to be implemented by subclasses)
        
        Args:
            protein_id: Protein identifier
            residue_names: List of residue names
            coords: Coordinates of CA atoms
        
        Returns:
            DGL heterogeneous graph or None if failed
        """
        raise NotImplementedError("Subclasses must implement build_graph()")
    
    def _encode_nodes(self, protein_id: str, residue_names: List[str]) -> np.ndarray:
        """Encode node features"""
        # Lazy initialization of encoder for precomputed encoders
        if self.encoder is None:
            # For precomputed encoders, we need to pass protein_id
            if self.encoding_type in ['precomputed', 'alphafold']:
                self.encoding_config_copy['protein_id'] = protein_id
            
            self.encoder = get_encoder(self.encoding_type, **self.encoding_config_copy)
        
        if self.encoding_type == 'precomputed':
            # For precomputed embeddings, we need special handling
            try:
                return self.encoder.encode(residue_names)
            except Exception as e:
                print(f"Error encoding {protein_id}: {e}")
                # Fallback to zero matrix
                input_dim = self.config.get('input_dim', 1280)
                return np.zeros((len(residue_names), input_dim))
        else:
            return self.encoder.encode(residue_names)
    
    def _build_edges(self, coords: np.ndarray, edge_types: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Build edges of specified types
        
        Args:
            coords: Coordinates of CA atoms
            edge_types: List of edge types to build (e.g., ['SEQ', 'STR_KNN'])
        
        Returns:
            Dictionary mapping edge type to edge list
        """
        # Build all edges first
        all_edges = build_all_edges(
            coords,
            sasa_values=None,
            spatial_threshold=self.spatial_threshold,
            knn_k=self.knn_k,
            surface_threshold=self.surface_threshold,
            surface_distance=self.surface_distance
        )
        
        # Filter to requested edge types
        filtered_edges = {}
        for edge_type in edge_types:
            if edge_type in all_edges and len(all_edges[edge_type]) > 0:
                filtered_edges[edge_type] = all_edges[edge_type]
        
        return filtered_edges


class StandardProteinGraphBuilder(ProteinGraphBuilder):
    """Standard protein graph builder with all edge types"""
    
    def __init__(self, config: Dict, pdb_dir: str):
        super().__init__(config, pdb_dir)
        self.edge_types = ['SEQ', 'STR_KNN', 'STR_DIS', 'SURF']
    
    def build_graph(self, protein_id: str, residue_names: List[str], coords: np.ndarray) -> Optional[dgl.DGLGraph]:
        """Build standard protein graph with all edge types"""
        try:
            # Build edges
            edges_dict = self._build_edges(coords, self.edge_types)
            
            if len(edges_dict) == 0:
                print(f"Warning: No edges found for {protein_id}")
                return None
            
            # Encode nodes
            node_features = self._encode_nodes(protein_id, residue_names)
            
            # Create heterogeneous graph
            graph_data = {}
            for edge_type, edges in edges_dict.items():
                graph_data[('amino_acid', edge_type, 'amino_acid')] = edges
            
            graph = dgl.heterograph(graph_data)
            graph.ndata['x'] = torch.FloatTensor(node_features)
            
            return graph
            
        except Exception as e:
            print(f"Error building graph for {protein_id}: {e}")
            return None


class PeptideGraphBuilder(ProteinGraphBuilder):
    """Peptide graph builder with simplified edge types (SEQ + STR_KNN only)"""
    
    def __init__(self, config: Dict, pdb_dir: str):
        super().__init__(config, pdb_dir)
        self.edge_types = ['SEQ', 'STR_KNN']  # Only sequence and KNN edges
    
    def build_graph(self, protein_id: str, residue_names: List[str], coords: np.ndarray) -> Optional[dgl.DGLGraph]:
        """Build simplified peptide graph"""
        try:
            # Build edges (only SEQ and STR_KNN)
            edges_dict = self._build_edges(coords, self.edge_types)
            
            if len(edges_dict) == 0:
                print(f"Warning: No edges found for peptide {protein_id}")
                return None
            
            # Encode nodes
            node_features = self._encode_nodes(protein_id, residue_names)
            
            # Create heterogeneous graph
            graph_data = {}
            for edge_type, edges in edges_dict.items():
                graph_data[('amino_acid', edge_type, 'amino_acid')] = edges
            
            graph = dgl.heterograph(graph_data)
            graph.ndata['x'] = torch.FloatTensor(node_features)
            
            return graph
            
        except Exception as e:
            print(f"Error building peptide graph for {protein_id}: {e}")
            return None


class LRREnhancedProteinGraphBuilder(ProteinGraphBuilder):
    """LRR-enhanced protein graph builder with additional LRR_REGION edges"""
    
    def __init__(self, config: Dict, pdb_dir: str, lrr_database: LRRDatabase):
        super().__init__(config, pdb_dir)
        self.edge_types = ['SEQ', 'STR_KNN', 'STR_DIS', 'SURF']
        self.lrr_database = lrr_database
        
        # LRR edge construction parameters
        self.lrr_edge_type = config.get('lrr_edge_type', 'LRR_REGION')
        self.lrr_connect_all = config.get('lrr_connect_all', True)
    
    def build_graph(self, protein_id: str, residue_names: List[str], coords: np.ndarray) -> Optional[dgl.DGLGraph]:
        """Build LRR-enhanced protein graph"""
        try:
            print(f"[DEBUG] LRREnhancedProteinGraphBuilder.build_graph() called for protein: {protein_id}")
            print(f"[DEBUG] Number of residues: {len(residue_names)}, number of coordinates: {len(coords)}")
            
            # Build standard edges
            edges_dict = self._build_edges(coords, self.edge_types)
            print(f"[DEBUG] Standard edges built: {list(edges_dict.keys())}")
            
            # Always include LRR_REGION edge type for consistency
            # If protein has LRR annotations, add actual edges; otherwise add empty edges
            print(f"[DEBUG] Checking LRR annotations for protein: {protein_id}")
            
            if self.lrr_database.has_lrr(protein_id):
                print(f"[DEBUG] ✓ Protein {protein_id} has LRR annotations")
                lrr_edges = self.lrr_database.get_lrr_edges(
                    protein_id,
                    connect_all=self.lrr_connect_all,
                    zero_based=True  # Use 0-based indices for graph construction
                )
                
                print(f"[DEBUG] Generated {len(lrr_edges)} LRR edges for {protein_id}")
                
                if len(lrr_edges) > 0:
                    edges_dict[self.lrr_edge_type] = lrr_edges
                    print(f"[DEBUG] ✓ Added {len(lrr_edges)} LRR edges for {protein_id}")
                else:
                    # Protein has LRR annotations but no edges (shouldn't happen, but handle it)
                    edges_dict[self.lrr_edge_type] = ([], [])
                    print(f"[WARNING] Added empty LRR edges for {protein_id} (has LRR annotations but no edges)")
            else:
                # Protein doesn't have LRR annotations - add empty LRR edges for consistency
                edges_dict[self.lrr_edge_type] = ([], [])
                print(f"[DEBUG] LRREnhancedProteinGraphBuilder: Added empty LRR edges for {protein_id} (no LRR annotations)")
            
            print(f"[DEBUG] LRREnhancedProteinGraphBuilder: Final edge types for {protein_id}: {list(edges_dict.keys())}")
            print(f"[DEBUG] Number of edges per type:")
            for edge_type, edges in edges_dict.items():
                if isinstance(edges, tuple) and len(edges) == 2:
                    print(f"[DEBUG]   {edge_type}: {len(edges[0])} edges")
                else:
                    print(f"[DEBUG]   {edge_type}: {len(edges)} edges")
            
            if len(edges_dict) == 0:
                print(f"Warning: No edges found for {protein_id}")
                return None
            
            # Encode nodes
            node_features = self._encode_nodes(protein_id, residue_names)
            
            # Create heterogeneous graph
            graph_data = {}
            for edge_type, edges in edges_dict.items():
                graph_data[('amino_acid', edge_type, 'amino_acid')] = edges
            
            graph = dgl.heterograph(graph_data)
            graph.ndata['x'] = torch.FloatTensor(node_features)
            
            return graph
            
        except Exception as e:
            print(f"Error building LRR-enhanced graph for {protein_id}: {e}")
            return None