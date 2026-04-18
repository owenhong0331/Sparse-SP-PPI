"""
Sparse-SP-PPI Models Package
"""

from .sparse_sp_ppi import ProteinGINModelSimple, ExplainableProteinGINModel, SparseEdgeAttentionEncoder
from .edge_construction import (
    build_sequence_edges,
    build_spatial_distance_edges,
    build_spatial_knn_edges,
    build_surface_edges,
    build_all_edges,
    read_ca_atoms_from_pdb
)
from .node_encoding import (
    NodeEncoder,
    MAPEPPIEncoder,
    PrecomputedEncoder,
    ESM2Encoder,
    OneHotEncoder,
    get_encoder,
    validate_embedding_directory
)
from .dataloader import (
    ProteinGraphDataset,
    PPIDataset,
    collate_protein_graphs
)
from .metrics import MetricsCalculator, format_metrics_string
from .logger import TrainingLogger, SimpleLogger
from .checkpoint import CheckpointManager, save_model_for_inference, load_model_for_inference
from .lrr_extractor import LRRExtractor

__all__ = [
    # Models
    'ProteinGINModelSimple',
    'ExplainableProteinGINModel',
    'SparseEdgeAttentionEncoder',

    # Edge construction
    'build_sequence_edges',
    'build_spatial_distance_edges',
    'build_spatial_knn_edges',
    'build_surface_edges',
    'build_all_edges',
    'read_ca_atoms_from_pdb',

    # Node encoding
    'NodeEncoder',
    'MAPEPPIEncoder',
    'PrecomputedEncoder',
    'ESM2Encoder',
    'OneHotEncoder',
    'get_encoder',
    'validate_embedding_directory',

    # Data loading
    'ProteinGraphDataset',
    'PPIDataset',
    'collate_protein_graphs',

    # Metrics
    'MetricsCalculator',
    'format_metrics_string',

    # Logging
    'TrainingLogger',
    'SimpleLogger',

    # Checkpointing
    'CheckpointManager',
    'save_model_for_inference',
    'load_model_for_inference',

    # LRR
    'LRRExtractor',
]
