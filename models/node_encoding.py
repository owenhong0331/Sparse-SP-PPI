"""
Node Encoding Module for Protein Graphs
Supports multiple encoding methods:
1. MAPE-PPI original 7-dimensional encoding
2. ESM2 pre-trained embeddings
3. ESM3 pre-trained embeddings
4. AlphaFold embeddings
5. Custom pre-computed embeddings from .npy files
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple


# Standard amino acid mapping
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

AA_1_TO_IDX = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}


class NodeEncoder:
    """Base class for node encoders"""
    
    def __init__(self, encoding_dim: int):
        self.encoding_dim = encoding_dim
    
    def encode(self, residue_names: List[str]) -> np.ndarray:
        """
        Encode a list of residue names
        
        Args:
            residue_names: List of 3-letter residue codes
        
        Returns:
            Encoded features as numpy array of shape (num_residues, encoding_dim)
        """
        raise NotImplementedError


class MAPEPPIEncoder(NodeEncoder):
    """
    MAPE-PPI original 7-dimensional encoding
    Based on physicochemical properties
    """
    
    def __init__(self, feature_file: str = None):
        super().__init__(encoding_dim=7)
        
        # Load feature matrix (20 amino acids x 7 features)
        if feature_file and os.path.exists(feature_file):
            self.feature_matrix = np.loadtxt(feature_file)
        else:
            # Default feature matrix (placeholder - should be loaded from all_assign.txt)
            self.feature_matrix = self._get_default_features()
        
        assert self.feature_matrix.shape == (20, 7), "Feature matrix should be 20x7"
    
    def _get_default_features(self) -> np.ndarray:
        """
        Get default physicochemical features
        This is a placeholder - actual features should be loaded from data
        """
        # Random initialization as placeholder
        np.random.seed(42)
        return np.random.randn(20, 7)
    
    def encode(self, residue_names: List[str]) -> np.ndarray:
        """Encode residues using MAPE-PPI features"""
        encoded = np.zeros((len(residue_names), 7))
        
        aa_to_idx = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
        }
        
        for i, res_name in enumerate(residue_names):
            if res_name in aa_to_idx:
                encoded[i] = self.feature_matrix[aa_to_idx[res_name]]
        
        return encoded


class PrecomputedEncoder(NodeEncoder):
    """
    Encoder for pre-computed embeddings from files
    Supports ESM2, ESM3, AlphaFold, and custom embeddings
    """
    
    def __init__(self, embedding_dir: str, protein_id: str, validate_dims: bool = True):
        """
        Args:
            embedding_dir: Directory containing .npy embedding files
            protein_id: Protein identifier (used to find the embedding file)
            validate_dims: Whether to validate embedding dimensions
        """
        self.embedding_dir = embedding_dir
        self.protein_id = protein_id
        self.validate_dims = validate_dims
        
        # Load embedding
        self.embedding = self._load_embedding()
        
        # Set encoding dimension
        super().__init__(encoding_dim=self.embedding.shape[1])
    
    def _load_embedding(self) -> np.ndarray:
        """Load pre-computed embedding from file"""
        # Try different file naming conventions
        possible_names = [
            f"{self.protein_id}.npy",
            f"{self.protein_id}_embedding.npy",
            f"{self.protein_id}_features.npy"
        ]
        
        for name in possible_names:
            filepath = os.path.join(self.embedding_dir, name)
            if os.path.exists(filepath):
                embedding = np.load(filepath)
                
                # Validate shape
                if len(embedding.shape) != 2:
                    raise ValueError(f"Embedding should be 2D (length x dim), got shape {embedding.shape}")
                
                return embedding
        
        raise FileNotFoundError(f"Could not find embedding file for {self.protein_id} in {self.embedding_dir}")
    
    def encode(self, residue_names: List[str]) -> np.ndarray:
        """
        Return pre-computed embeddings
        
        Args:
            residue_names: List of residue names (used for validation)
        
        Returns:
            Pre-computed embeddings
        """
        if self.validate_dims and len(residue_names) != self.embedding.shape[0]:
            raise ValueError(
                f"Number of residues ({len(residue_names)}) does not match "
                f"embedding length ({self.embedding.shape[0]})"
            )
        
        return self.embedding


class ESM2Encoder(NodeEncoder):
    """
    ESM2 (Evolutionary Scale Modeling 2) encoder
    Uses pre-trained protein language model
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = 'cuda'):
        """
        Args:
            model_name: ESM2 model variant
            device: Device to run model on
        """
        try:
            import esm
        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")
        
        self.device = device
        
        # Load ESM2 model
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Get embedding dimension
        super().__init__(encoding_dim=self.model.embed_dim)
    
    def encode(self, residue_names: List[str], sequence: Optional[str] = None) -> np.ndarray:
        """
        Encode using ESM2
        
        Args:
            residue_names: List of 3-letter residue codes
            sequence: Optional 1-letter amino acid sequence (if None, derived from residue_names)
        
        Returns:
            ESM2 embeddings
        """
        # Convert to 1-letter sequence if not provided
        if sequence is None:
            sequence = ''.join([AA_3_TO_1.get(res, 'X') for res in residue_names])
        
        # Prepare data
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
            embeddings = results["representations"][self.model.num_layers]
        
        # Remove batch dimension and special tokens
        embeddings = embeddings[0, 1:len(sequence)+1].cpu().numpy()
        
        return embeddings


class ESM3Encoder(NodeEncoder):
    """
    ESM3 encoder (placeholder - to be implemented when ESM3 is released)
    """
    
    def __init__(self):
        raise NotImplementedError("ESM3 encoder not yet implemented")


class OneHotEncoder(NodeEncoder):
    """Simple one-hot encoding of amino acids"""
    
    def __init__(self):
        super().__init__(encoding_dim=20)
    
    def encode(self, residue_names: List[str]) -> np.ndarray:
        """One-hot encode amino acids"""
        encoded = np.zeros((len(residue_names), 20))
        
        for i, res_name in enumerate(residue_names):
            aa_1 = AA_3_TO_1.get(res_name, 'A')  # Default to Alanine if unknown
            idx = AA_1_TO_IDX.get(aa_1, 0)
            encoded[i, idx] = 1.0
        
        return encoded


def get_encoder(encoding_type: str, **kwargs) -> NodeEncoder:
    """
    Factory function to get appropriate encoder
    
    Args:
        encoding_type: Type of encoding ('mape', 'esm2', 'esm3', 'alphafold', 'precomputed', 'onehot')
        **kwargs: Additional arguments for specific encoders
    
    Returns:
        NodeEncoder instance
    """
    encoding_type = encoding_type.lower()
    
    if encoding_type == 'mape':
        return MAPEPPIEncoder(kwargs.get('feature_file'))
    
    elif encoding_type == 'esm2':
        return ESM2Encoder(
            model_name=kwargs.get('model_name', 'esm2_t33_650M_UR50D'),
            device=kwargs.get('device', 'cuda')
        )
    
    elif encoding_type == 'esm3':
        return ESM3Encoder()
    
    elif encoding_type in ['alphafold', 'precomputed']:
        return PrecomputedEncoder(
            embedding_dir=kwargs['embedding_dir'],
            protein_id=kwargs['protein_id'],
            validate_dims=kwargs.get('validate_dims', True)
        )
    
    elif encoding_type == 'onehot':
        return OneHotEncoder()
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def validate_embedding_directory(embedding_dir: str) -> Tuple[bool, Dict[str, any]]:
    """
    Validate an embedding directory
    
    Args:
        embedding_dir: Path to directory containing embeddings
    
    Returns:
        Tuple of (is_valid, info_dict)
    """
    if not os.path.exists(embedding_dir):
        return False, {"error": "Directory does not exist"}
    
    # Find all .npy files
    npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        return False, {"error": "No .npy files found"}
    
    # Check dimensions consistency
    dims = []
    for npy_file in npy_files[:10]:  # Check first 10 files
        filepath = os.path.join(embedding_dir, npy_file)
        try:
            emb = np.load(filepath)
            if len(emb.shape) == 2:
                dims.append(emb.shape[1])
        except Exception as e:
            continue
    
    if len(dims) == 0:
        return False, {"error": "Could not load any embeddings"}
    
    # Check if all dimensions are the same
    unique_dims = set(dims)
    if len(unique_dims) > 1:
        return False, {"error": f"Inconsistent embedding dimensions: {unique_dims}"}
    
    return True, {
        "num_files": len(npy_files),
        "embedding_dim": dims[0],
        "sample_files": npy_files[:5]
    }

