"""
Customer Data Loader for Protein-Protein Interaction Prediction
Supports multiple encoding methods and heterogeneous graph construction
"""

import os
import csv
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
import psutil
import gc
import hashlib
import hashlib

import dgl
import torch
from torch.utils.data import Dataset, DataLoader

from .node_encoding import get_encoder, validate_embedding_directory
from .edge_construction import build_all_edges, read_ca_atoms_from_pdb
from .lrr_parser import LRRDatabase
from .protein_graph_builder import (
    StandardProteinGraphBuilder,
    PeptideGraphBuilder,
    LRREnhancedProteinGraphBuilder
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProteinGraphDataset(Dataset):
    """
    Dataset for protein graphs with configurable node encodings
    """
    
    def __init__(self, config: Dict, protein_ids: List[str], pdb_dir: str,
                 cache_dir: Optional[str] = None, balance_dataset: bool = False):
        """
        Args:
            config: Configuration dictionary with encoding settings
            protein_ids: List of protein identifiers
            pdb_dir: Directory containing PDB files
            cache_dir: Directory to cache processed graphs
            balance_dataset: Whether this is for a balanced dataset
        """
        self.config = config
        self.protein_ids = protein_ids
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        self.balance_dataset = balance_dataset

        # Encoding configuration
        self.encoding_type = config.get('encoding_type', 'mape')
        self.encoding_config = config.get('encoding_config', {})

        # Edge construction parameters
        self.spatial_threshold = config.get('spatial_threshold', 10.0)
        self.knn_k = config.get('knn_k', 5)
        self.surface_threshold = config.get('surface_threshold', 0.2)
        self.surface_distance = config.get('surface_distance', 8.0)

        # Multi-encoder configuration
        self.peptide_encoder_enabled = config.get('peptide_encoder_enabled', False)
        self.lrr_encoder_enabled = config.get('lrr_encoder_enabled', False)
        self.peptide_length_threshold = config.get('peptide_length_threshold', 50)
        
        # Log multi-encoder configuration status
        print(f"[CONFIG] Peptide encoder enabled: {self.peptide_encoder_enabled}")
        print(f"[CONFIG] LRR encoder enabled: {self.lrr_encoder_enabled}")
        print(f"[CONFIG] Peptide length threshold: {self.peptide_length_threshold}")

        # Initialize LRR database if LRR encoder is enabled
        self.lrr_database = None
        if self.lrr_encoder_enabled:
            lrr_annotation_file = config.get('lrr_annotation_file',
                                            'customer_ppi/scripts/lrr/lrr_annotation_results.txt')
            print(f"[CONFIG] LRR annotation file: {lrr_annotation_file}")
            if os.path.exists(lrr_annotation_file):
                self.lrr_database = LRRDatabase(lrr_annotation_file)
                print(f"[CONFIG] LRR encoder enabled. Loaded annotations for {len(self.lrr_database.protein_lrr_regions)} proteins")
            else:
                print(f"[CONFIG] WARNING: LRR annotation file not found: {lrr_annotation_file}")
                self.lrr_encoder_enabled = False
        else:
            print(f"[CONFIG] LRR encoder disabled")

        # Track protein types (peptide vs protein)
        self.protein_types = {}  # {protein_id: 'peptide' or 'protein'}

        # Load or build graphs
        self.graphs = self._load_or_build_graphs()
    
    def _get_cache_path(self) -> Optional[str]:
        """Get cache file path"""
        if self.cache_dir is None:
            return None
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Add balanced identifier to cache name if this is a balanced dataset
        if self.balance_dataset:
            cache_name = f"protein_graphs_{self.encoding_type}_balanced.pkl"
        else:
            cache_name = f"protein_graphs_{self.encoding_type}.pkl"
            
        return os.path.join(self.cache_dir, cache_name)
    
    def _calculate_optimal_batch_size(self, total_proteins: int) -> int:
        """Calculate optimal batch size based on encoding type and available memory"""
        # Base batch sizes for different encoding types
        base_batch_sizes = {
            'mape': 1000,      # Smallest embeddings (7-20 dimensions)
            'esm2': 200,       # ESM2 embeddings (1280-2560 dimensions)
            'precomputed': 150, # Precomputed embeddings (various sizes)
            'alphafold': 100,  # AlphaFold embeddings (large dimensions)
            'onehot': 800      # One-hot encoding (small)
        }
        
        # Get base batch size for encoding type
        base_size = base_batch_sizes.get(self.encoding_type, 100)
        
        # Adjust for large datasets
        if total_proteins > 5000:
            base_size = max(10, base_size // 2)
        elif total_proteins > 2000:
            base_size = max(20, base_size // 2)
        
        # Check available system memory and adjust accordingly
        available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
        
        if available_memory_gb < 2.0:
            base_size = max(10, base_size // 4)
        elif available_memory_gb < 4.0:
            base_size = max(20, base_size // 2)
        
        # For ESM2 3B specifically, use smaller batches due to large embeddings
        if self.encoding_type in ['precomputed', 'esm2']:
            # Check if this is a high-dimensional model
            input_dim = self.config.get('input_dim', 1280)
            if input_dim >= 2000:  # High-dimensional models like ESM2 3B
                base_size = max(50, base_size // 3)
        
        # Ensure batch size is reasonable
        batch_size = min(max(10, base_size), total_proteins)
        
        return batch_size
    
    def _load_or_build_graphs(self) -> List[dgl.DGLGraph]:
        """Load cached graphs or build new ones"""
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached graphs from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Check if this is a chunked cache file
                if isinstance(cache_data, dict) and 'chunk_files' in cache_data:
                    print(f"Loading chunked cache with {len(cache_data['chunk_files'])} chunks...")
                    cached_graphs = []
                    self.protein_types = {}  # Initialize protein_types from cache
                    
                    for chunk_file in cache_data['chunk_files']:
                        if os.path.exists(chunk_file):
                            with open(chunk_file, 'rb') as f:
                                chunk_data = pickle.load(f)
                            
                            # Load graphs and protein types from chunk
                            if isinstance(chunk_data, dict) and 'graphs' in chunk_data:
                                cached_graphs.extend(chunk_data['graphs'])
                                if 'protein_types' in chunk_data:
                                    self.protein_types.update(chunk_data['protein_types'])
                            else:
                                # Legacy format (just graphs)
                                cached_graphs.extend(chunk_data)
                            
                            # Free memory after loading each chunk
                            del chunk_data
                            gc.collect()
                            
                            current_memory = get_memory_usage()
                            print(f"Loaded chunk {chunk_file}, memory: {current_memory:.2f} GB")
                        else:
                            print(f"WARNING: Chunk file {chunk_file} not found!")
                    
                    # Also load protein types from main cache file if available
                    if 'protein_types' in cache_data:
                        self.protein_types.update(cache_data['protein_types'])
                    
                    print(f"Loaded {len(cached_graphs)} graphs from chunked cache")
                    print(f"Loaded protein types for {len(self.protein_types)} proteins from cache")
                else:
                    # Standard cache file
                    if isinstance(cache_data, dict) and 'graphs' in cache_data:
                        # New format with protein types
                        cached_graphs = cache_data['graphs']
                        if 'protein_types' in cache_data:
                            self.protein_types = cache_data['protein_types']
                            print(f"Loaded protein types for {len(self.protein_types)} proteins from cache")
                    else:
                        # Legacy format (just graphs)
                        cached_graphs = cache_data
                        self.protein_types = {}  # Initialize empty for legacy cache
                    print(f"Loaded {len(cached_graphs)} graphs from standard cache")
                
                if len(cached_graphs) == 0:
                    print("WARNING: Cached graph list is empty! Will rebuild graphs.")
                else:
                    # 新增：验证缓存蛋白质图数量与当前蛋白质集合是否一致
                    if len(cached_graphs) != len(self.protein_ids):
                        print(f"⚠️  缓存不匹配: 缓存有{len(cached_graphs)}个图，当前有{len(self.protein_ids)}个蛋白质")
                        print("将重新构建蛋白质图...")
                        # 删除不匹配的缓存文件
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                            print(f"已删除不匹配的缓存: {cache_path}")
                        # 继续执行构建逻辑
                    else:
                        print(f"✅  缓存匹配: {len(cached_graphs)}个图与{len(self.protein_ids)}个蛋白质一致")
                        return cached_graphs
                    
            except (pickle.PickleError, EOFError, Exception) as e:
                print(f"WARNING: Failed to load cache from {cache_path}: {str(e)}")
                print("Will rebuild graphs from scratch.")
                # Remove corrupted cache file
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    print(f"Removed corrupted cache file: {cache_path}")
        
        # Build graphs
        print(f"Building protein graphs with {self.encoding_type} encoding...")
        print(f"Total proteins to process: {len(self.protein_ids)}")
        
        # Smart batch size calculation based on encoding type and available memory
        batch_size = self._calculate_optimal_batch_size(len(self.protein_ids))
        
        print(f"Using optimized batch size: {batch_size} proteins per batch")
        
        graphs = []
        failed_count = 0
        
        for i in range(0, len(self.protein_ids), batch_size):
            batch_proteins = self.protein_ids[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(self.protein_ids)-1)//batch_size + 1} "
                  f"(proteins {i+1}-{min(i+batch_size, len(self.protein_ids))})")
            
            batch_graphs = []
            for protein_id in tqdm(batch_proteins, desc="Processing proteins"):
                try:
                    graph = self._build_protein_graph(protein_id)
                    if graph is not None:
                        batch_graphs.append(graph)
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Error processing1 {protein_id}: {str(e)}")
                    failed_count += 1
            
            graphs.extend(batch_graphs)
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Monitor memory usage and dynamically adjust batch size if needed
            current_memory = get_memory_usage()
            available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
            
            print(f"Current memory usage: {current_memory:.2f} GB")
            print(f"Available system memory: {available_memory:.2f} GB")
            
            # Dynamic batch size adjustment
            if available_memory < 2.0:  # Less than 2GB available
                new_batch_size = max(10, batch_size // 2)
                if new_batch_size != batch_size:
                    print(f"⚠️  Low memory warning! Reducing batch size from {batch_size} to {new_batch_size}")
                    batch_size = new_batch_size
            
            print(f"Batch completed: {len(batch_graphs)} graphs built, "
                  f"{len(batch_proteins) - len(batch_graphs)} failed")

        print(f"Successfully built {len(graphs)} graphs")
        print(f"Failed to build {failed_count} graphs")

        if len(graphs) == 0:
            print("ERROR: No graphs were successfully built!")
            print("Please check:")
            print(f"  1. PDB directory exists: {self.pdb_dir}")
            print(f"  2. PDB files match protein IDs")
            print(f"  3. PDB files are valid")

        # Save to cache with memory optimization
        if cache_path:
            print(f"Saving {len(graphs)} graphs to cache: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Memory-optimized cache saving
            print("Using memory-optimized cache saving strategy...")
            
            # Create cache data that includes both graphs and protein_types
            cache_data = {
                'graphs': graphs,
                'protein_types': self.protein_types  # Save protein type information
            }
            
            # Option 1: Save in smaller chunks to reduce memory pressure
            chunk_size = min(100, len(graphs))  # Save in chunks of 100 graphs
            
            if len(graphs) > chunk_size:
                print(f"Saving graphs in chunks of {chunk_size}...")
                
                # Calculate number of chunks
                num_chunks = (len(graphs) - 1) // chunk_size + 1
                chunk_files = []
                
                # Save chunks sequentially
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(graphs))
                    chunk_graphs = graphs[start_idx:end_idx]
                    
                    # Create chunk data with protein types for the corresponding proteins
                    chunk_protein_types = {}
                    for protein_id in self.protein_types:
                        protein_idx = self.protein_ids.index(protein_id)
                        if start_idx <= protein_idx < end_idx:
                            chunk_protein_types[protein_id] = self.protein_types[protein_id]
                    
                    chunk_data = {
                        'graphs': chunk_graphs,
                        'protein_types': chunk_protein_types
                    }
                    
                    chunk_file = cache_path.replace('.pkl', f'_chunk_{chunk_idx}.pkl')
                    chunk_files.append(chunk_file)
                    
                    print(f"Saving chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_graphs)} graphs)")
                    
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(chunk_data, f)
                    
                    # Free memory after each chunk
                    del chunk_data, chunk_graphs
                    gc.collect()
                    
                    current_memory = get_memory_usage()
                    print(f"Memory after chunk {chunk_idx + 1}: {current_memory:.2f} GB")
                
                # Save main cache file with chunk references
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'chunk_files': chunk_files, 
                        'total_graphs': len(graphs),
                        'protein_types': self.protein_types  # Save full protein types info
                    }, f)
                
                print("Chunked cache saving completed!")
            else:
                # Small dataset, save normally
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print("Standard cache saving completed!")

        return graphs
    
    def _build_protein_graph(self, protein_id: str) -> Optional[dgl.DGLGraph]:
        """Build heterogeneous graph for a single protein using appropriate encoder"""
        # Find PDB file
        pdb_file = os.path.join(self.pdb_dir, f"{protein_id}.pdb")
        if not os.path.exists(pdb_file):
            print(f"Warning: PDB file not found for {protein_id}")
            return None

        try:
            # Read structure
            coords, residue_names = read_ca_atoms_from_pdb(pdb_file)

            if len(coords) == 0:
                print(f"Warning: No CA atoms found in {protein_id}")
                return None

            residue_count = len(residue_names)

            # Select appropriate graph builder based on configuration
            graph_builder = self._select_graph_builder(protein_id, residue_count)

            # Build graph using selected builder
            graph = graph_builder.build_graph(protein_id, residue_names, coords)

            # Track protein type for PPI network construction
            # Type tracking should match graph builder selection logic
            if self.peptide_encoder_enabled and residue_count < self.peptide_length_threshold:
                self.protein_types[protein_id] = 'peptide'
                print(f"[DEBUG] Protein {protein_id} marked as 'peptide' (residue_count={residue_count})")
            elif self.lrr_encoder_enabled:
                self.protein_types[protein_id] = 'lrr_enhanced'
                print(f"[DEBUG] Protein {protein_id} marked as 'lrr_enhanced' (LRR encoder enabled)")
            else:
                self.protein_types[protein_id] = 'protein'
                print(f"[DEBUG] Protein {protein_id} marked as 'protein' (LRR encoder disabled)")

            return graph

        except Exception as e:
            print(f"Error processing {protein_id}: {str(e)}")
            return None

    def _select_graph_builder(self, protein_id: str, residue_count: int):
        """
        Select appropriate graph builder based on configuration and protein characteristics

        Priority:
        1. If peptide_encoder_enabled and residue_count < threshold -> PeptideGraphBuilder
        2. If lrr_encoder_enabled -> Use LRREnhancedProteinGraphBuilder for NON-PEPTIDE proteins
        3. Otherwise -> StandardProteinGraphBuilder
        
        When LRR encoder is enabled, non-peptide proteins use LRR encoding to ensure no 'standard' type exists.
        """
        # Check if this is a peptide (small protein) - highest priority
        if self.peptide_encoder_enabled and residue_count < self.peptide_length_threshold:
            print(f"Using PeptideGraphBuilder for {protein_id} (residue_count={residue_count})")
            return PeptideGraphBuilder(self.config, self.pdb_dir)

        # If LRR encoder is enabled, use LRREnhancedProteinGraphBuilder for NON-PEPTIDE proteins
        # This ensures no 'standard' protein type exists when LRR is enabled
        if self.lrr_encoder_enabled:
            # LRR数据库必须存在，否则异常退出
            if not self.lrr_database:
                raise RuntimeError(f"LRR encoder enabled but LRR database is not available for {protein_id}. "
                                  "LRR annotation file must exist when LRR encoder is enabled.")
            
            print(f"[DEBUG] Selecting LRREnhancedProteinGraphBuilder for {protein_id} (LRR encoder enabled, non-peptide)")
            return LRREnhancedProteinGraphBuilder(self.config, self.pdb_dir, self.lrr_database)

        # Default: standard protein graph builder (only when LRR is not enabled)
        return StandardProteinGraphBuilder(self.config, self.pdb_dir)
    
    def _encode_nodes(self, protein_id: str, residue_names: List[str]) -> np.ndarray:
        """Encode node features based on configuration"""
        if self.encoding_type == 'precomputed' or self.encoding_type == 'alphafold':
            # For precomputed embeddings, pass protein_id
            # Remove encoding_type from encoding_config to avoid duplicate parameter
            encoding_config_copy = self.encoding_config.copy()
            encoding_config_copy.pop('encoding_type', None)  # Remove if exists
            
            encoder = get_encoder(
                self.encoding_type,
                protein_id=protein_id,
                **encoding_config_copy
            )
            try:
                embedding = encoder.encode(residue_names)
                # If dimensions don't match, handle it gracefully
                if len(embedding) != len(residue_names):
                    validate_dims = self.encoding_config.get('validate_dims', True)
                    if not validate_dims:
                        # Truncate or pad to match residue count
                        if len(embedding) > len(residue_names):
                            embedding = embedding[:len(residue_names)]
                        else:
                            # Pad with zeros
                            padding = np.zeros((len(residue_names) - len(embedding), embedding.shape[1]))
                            embedding = np.vstack([embedding, padding])
                        print(f"Warning: Adjusted embedding dimensions for {protein_id}: "
                              f"original {len(embedding if len(embedding) != len(residue_names) else 'N/A')} vs needed {len(residue_names)}")
                    else:
                        # If validation is enabled, let the original error propagate
                        pass
                return embedding
            except Exception as e:
                print(f"Error processing 3{protein_id}: {str(e)}")
                # Fallback: return zero matrix
                # Try both nested and flat config structures
                if 'model' in self.config and isinstance(self.config['model'], dict):
                    input_dim = self.config['model'].get('input_dim', 1280)
                else:
                    input_dim = self.config.get('input_dim', 1280)

                # Log warning about fallback
                print(f"⚠️  WARNING: Using fallback zero matrix for {protein_id}")
                print(f"    Fallback dimension: {input_dim}")
                print(f"    Residue count: {len(residue_names)}")
                if input_dim == 1280:
                    print(f"    ⚠️  ALERT: Fallback dimension is 1280 (default), may cause dimension mismatch!")
                    print(f"    Expected dimension for ESM3-small: 1536")

                return np.zeros((len(residue_names), input_dim))  # Using config input_dim
        else:
            # For other encodings
            # Remove encoding_type from encoding_config to avoid duplicate parameter
            encoding_config_copy = self.encoding_config.copy()
            encoding_config_copy.pop('encoding_type', None)  # Remove if exists
            
            encoder = get_encoder(self.encoding_type, **encoding_config_copy)
            return encoder.encode(residue_names)
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        return self.graphs[idx]


def collate_protein_graphs(samples: List[dgl.DGLGraph]) -> dgl.DGLGraph:
    """
    向后兼容的批处理函数
    如果检测到异质性（多种边类型），则使用新的分类批处理函数
    """
    # 检查是否需要三类型分类批处理
    if len(samples) > 0:
        # 检测图的异质性
        first_graph_etypes = samples[0].etypes
        has_heterogeneity = False
        
        for i, graph in enumerate(samples):
            if i > 0 and graph.etypes != first_graph_etypes:
                has_heterogeneity = True
                break
        
        # 如果有异质性，使用新的分类批处理
        if has_heterogeneity:
            # print("检测到异质性图，使用三类型分类批处理")
            try:
                # 导入新的分类批处理函数
                from .heterogeneous_high_ppi import collate_protein_graphs_with_classification
                batch_results, graph_types = collate_protein_graphs_with_classification(samples)
                
                # 如果返回的是字典，需要特殊处理（如使用forward_with_classification）
                if isinstance(batch_results, dict):
                    # print(f"返回分类批处理结果: {list(batch_results.keys())}")
                    # 返回第一个批处理图用于向后兼容
                    if batch_results:
                        first_key = list(batch_results.keys())[0]
                        return batch_results[first_key]
                    else:
                        raise RuntimeError("分类批处理返回空结果")
                else:
                    # 应该是单个批处理图
                    return batch_results
            except ImportError as e:
                print(f"无法导入分类批处理函数，使用传统批处理: {e}")
    
    # 传统批处理逻辑
    # print(f"DGL version: {dgl.__version__}")
    # print(f"Available batch functions: batch={hasattr(dgl, 'batch')}, batch_hetero={hasattr(dgl, 'batch_hetero')}")
    
    # Debug: check edge types in each sample
    if len(samples) > 0:
        # print(f"collate_protein_graphs: Processing {len(samples)} graphs")
        
        # Check if all graphs have the same edge types in the same order
        first_graph_etypes = samples[0].etypes
        consistent = True
        
        for i, graph in enumerate(samples):
            # print(f"  Graph {i}: {graph.num_nodes()} nodes, edge types: {graph.etypes}")
            
            # Check edge type order consistency
            if graph.etypes != first_graph_etypes:
                # print(f"    WARNING: Different edge type order! Expected: {first_graph_etypes}")
                consistent = False
            
            # Check if graph has empty edges
            for etype in graph.etypes:
                num_edges = graph.number_of_edges(etype)
                if num_edges == 0:
                    print(f"    {etype}: {num_edges} edges (empty)")
        
        # if consistent:
            # print(f"  All graphs have consistent edge type order: {first_graph_etypes}")
        # else:
        #     print(f"  WARNING: Graphs have inconsistent edge type order!")
            # print(f"  First graph order: {first_graph_etypes}")
    
    # Try different batch strategies based on DGL version
    if hasattr(dgl, 'batch'):
        # print("Using dgl.batch()")
        return dgl.batch(samples)
    elif hasattr(dgl, 'batch_hetero'):
        # print("Using dgl.batch_hetero()")
        return dgl.batch_hetero(samples)
    else:
        raise RuntimeError("DGL does not have batch or batch_hetero function")


def collate_heterogeneous_protein_graphs(samples: List[dgl.DGLGraph]) -> Dict[str, dgl.DGLGraph]:
    """
    专门用于异质性图的三类型分类批处理函数
    返回包含三种批处理图的字典
    """
    try:
        from .heterogeneous_high_ppi import collate_protein_graphs_with_classification
        batch_results, graph_types = collate_protein_graphs_with_classification(samples)
        
        # 确保返回的是字典格式
        if isinstance(batch_results, dict):
            return batch_results
        else:
            # 如果不是字典，创建一个包含所有图的字典
            return {'standard': batch_results} if batch_results else {}
    except ImportError as e:
        # print(f"无法导入分类批处理函数: {e}")
        # 返回空的字典作为退路
        return {}


def is_heterogeneous_dataset(samples: List[dgl.DGLGraph]) -> bool:
    """
    检测数据集是否包含异质性图（多种边类型）
    """
    if len(samples) <= 1:
        return False
    
    # 检查所有图的边类型是否一致
    first_graph_etypes = samples[0].etypes
    
    for i, graph in enumerate(samples):
        if i > 0 and graph.etypes != first_graph_etypes:
            return True
    
    return False


class PPIDataset:
    """
    Dataset for Protein-Protein Interactions
    """
    
    def __init__(self, config: Dict, ppi_file: str, protein_seq_file: str,
                 pdb_dir: str, cache_dir: Optional[str] = None, balance_dataset: bool = False,
                 enable_protein_positive_split: bool = False):
        """
        Args:
            config: Configuration dictionary
            ppi_file: Path to PPI interaction file
            protein_seq_file: Path to protein sequence dictionary
            pdb_dir: Directory containing PDB files
            cache_dir: Directory for caching
            balance_dataset: Whether to balance dataset (1:1 pos:neg ratio)
            enable_protein_positive_split: Whether to enable protein-level positive set splitting
        """
        self.config = config
        self.ppi_file = ppi_file
        self.protein_seq_file = protein_seq_file
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        self.balance_dataset = balance_dataset
        self.enable_protein_positive_split = enable_protein_positive_split
        
        # Load raw data first
        protein_name_to_id, ppi_list, ppi_labels = self._load_raw_ppi_data()
        
        # Store original raw data for validation
        original_protein_name_to_id = protein_name_to_id
        original_ppi_list = ppi_list
        original_ppi_labels = ppi_labels
        
        # Validate data consistency before processing
        protein_ids = list(original_protein_name_to_id.keys())
        valid_proteins, validation_report = validate_data_consistency(
            protein_ids=protein_ids,
            pdb_dir=pdb_dir,
            encoding_config=config.get('encoding_config', {}),
            protein_seq_file=protein_seq_file,
            verbose=True
        )
        
        # Filter PPI data to only include valid proteins
        filtered_ppi_list, filtered_ppi_labels, filtered_protein_name_to_id = filter_ppi_data(
            ppi_list=original_ppi_list,
            labels=original_ppi_labels,
            valid_proteins=valid_proteins,
            protein_id_to_idx=original_protein_name_to_id,
            verbose=True
        )
        
        # Store validated data
        self.protein_name_to_id = filtered_protein_name_to_id
        self.ppi_list = filtered_ppi_list
        self.ppi_labels = filtered_ppi_labels
        
        # Apply dataset balancing if requested BEFORE building protein graphs
        if self.balance_dataset:
            if self.enable_protein_positive_split:
                self._protein_positive_set_balancing()
            else:
                self._apply_dataset_balancing()
            print(f"After balancing: {len(self.ppi_list)} interactions")
            
            # Protein mapping already updated in _protein_positive_set_balancing()
            # No need to update again here
            print(f"After balancing: {len(self.protein_name_to_id)} proteins in balanced dataset")
        
        # Build protein graphs only for valid proteins (after balancing if applicable)
        valid_protein_ids = list(self.protein_name_to_id.keys())
        self.protein_dataset = ProteinGraphDataset(
            config, valid_protein_ids, pdb_dir, cache_dir, self.balance_dataset
        )

        # Build PPI graph
        self.ppi_graph = self._build_ppi_graph()
    
    

    def _build_ppi_graph(self) -> dgl.DGLGraph:
        """Build PPI network graph (heterogeneous if multi-encoder is enabled and actual heterogeneity exists)"""
        # Check if heterogeneous PPI network is needed
        heterogeneous_ppi = self.config.get('peptide_encoder_enabled', False) or \
                           self.config.get('lrr_encoder_enabled', False)
        
        # Log PPI network configuration
        print(f"[PPI_CONFIG] Heterogeneous PPI network: {heterogeneous_ppi}")
        print(f"[PPI_CONFIG] Peptide encoder enabled: {self.config.get('peptide_encoder_enabled', False)}")
        print(f"[PPI_CONFIG] LRR encoder enabled: {self.config.get('lrr_encoder_enabled', False)}")
        
        if heterogeneous_ppi and hasattr(self.protein_dataset, 'protein_types'):
            # Check if actual protein type heterogeneity exists
            protein_types = self.protein_dataset.protein_types
            
            if protein_types:
                # Count unique protein types to determine actual heterogeneity
                unique_types = set(protein_types.values())
                print(f"[PPI_CONFIG] Found {len(unique_types)} unique protein type(s): {list(unique_types)}")
                
                # Only build heterogeneous graph if there are multiple protein types
                if len(unique_types) > 1:
                    print(f"[PPI_CONFIG] Building heterogeneous PPI graph with {len(unique_types)} node types")
                    # Build heterogeneous PPI graph
                    return self._build_heterogeneous_ppi_graph()
                else:
                    single_type = list(unique_types)[0] if unique_types else "unknown"
                    print(f"[PPI_CONFIG] Only {len(unique_types)} protein type found ({single_type}), falling back to homogeneous graph")
                    # Fall back to standard homogeneous graph
                    g = dgl.to_bidirected(dgl.graph(self.ppi_list))
                    g = dgl.add_self_loop(g)
                    return g.to(device)
            else:
                print("[PPI_CONFIG] WARNING: protein_types dictionary is empty, falling back to homogeneous graph")
                # Fall back to standard homogeneous graph
                g = dgl.to_bidirected(dgl.graph(self.ppi_list))
                g = dgl.add_self_loop(g)
                return g.to(device)
        else:
            print("[PPI_CONFIG] Building standard homogeneous PPI graph")
            # Build standard homogeneous PPI graph
            g = dgl.to_bidirected(dgl.graph(self.ppi_list))
            g = dgl.add_self_loop(g)
            return g.to(device)

    def _build_heterogeneous_ppi_graph(self) -> dgl.DGLGraph:
        """Build heterogeneous PPI network with peptide and protein node types"""
        protein_types = self.protein_dataset.protein_types
        
        # Log PPI network node type configuration
        peptide_node_type = self.config.get('peptide_node_type', 'peptide')
        protein_node_type = self.config.get('protein_node_type', 'protein')
        print(f"[PPI_CONFIG] Peptide node type: {peptide_node_type}")
        print(f"[PPI_CONFIG] Protein node type: {protein_node_type}")
        
        # Count protein types
        peptide_count = sum(1 for pt in protein_types.values() if pt == 'peptide')
        protein_count = sum(1 for pt in protein_types.values() if pt == 'protein')
        print(f"[PPI_CONFIG] Peptide nodes: {peptide_count}, Protein nodes: {protein_count}")

        # Separate edges by node type combinations
        peptide_peptide_edges = []
        peptide_protein_edges = []
        protein_peptide_edges = []
        protein_protein_edges = []

        # Create node ID mappings
        peptide_nodes = []
        protein_nodes = []
        global_to_local = {}  # {global_id: (node_type, local_id)}

        for global_id, node_type in protein_types.items():
            if node_type == 'peptide':
                local_id = len(peptide_nodes)
                peptide_nodes.append(global_id)
                global_to_local[global_id] = ('peptide', local_id)
            else:
                local_id = len(protein_nodes)
                protein_nodes.append(global_id)
                global_to_local[global_id] = ('protein', local_id)

        # Classify edges
        for src, dst in self.ppi_list:
            if src not in global_to_local or dst not in global_to_local:
                continue

            src_type, src_local = global_to_local[src]
            dst_type, dst_local = global_to_local[dst]

            if src_type == 'peptide' and dst_type == 'peptide':
                peptide_peptide_edges.append((src_local, dst_local))
                peptide_peptide_edges.append((dst_local, src_local))  # Bidirectional
            elif src_type == 'peptide' and dst_type == 'protein':
                peptide_protein_edges.append((src_local, dst_local))
                protein_peptide_edges.append((dst_local, src_local))  # Reverse
            elif src_type == 'protein' and dst_type == 'peptide':
                protein_peptide_edges.append((src_local, dst_local))
                peptide_protein_edges.append((dst_local, src_local))  # Reverse
            else:  # protein-protein
                protein_protein_edges.append((src_local, dst_local))
                protein_protein_edges.append((dst_local, src_local))  # Bidirectional

        # Build heterogeneous graph
        graph_data = {}

        if len(peptide_peptide_edges) > 0:
            graph_data[('peptide', 'interacts', 'peptide')] = peptide_peptide_edges
        if len(peptide_protein_edges) > 0:
            graph_data[('peptide', 'interacts', 'protein')] = peptide_protein_edges
        if len(protein_peptide_edges) > 0:
            graph_data[('protein', 'interacts', 'peptide')] = protein_peptide_edges
        if len(protein_protein_edges) > 0:
            graph_data[('protein', 'interacts', 'protein')] = protein_protein_edges

        # Create heterogeneous graph
        num_nodes_dict = {
            'peptide': len(peptide_nodes),
            'protein': len(protein_nodes)
        }

        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

        # Add self-loops for each node type
        for ntype in g.ntypes:
            g = dgl.add_self_loop(g, ntype=ntype)

        # Store node mappings for later use
        self.peptide_nodes = peptide_nodes
        self.protein_nodes = protein_nodes
        self.global_to_local = global_to_local

        print(f"Built heterogeneous PPI graph:")
        print(f"  Peptide nodes: {len(peptide_nodes)}")
        print(f"  Protein nodes: {len(protein_nodes)}")
        print(f"  Edge types: {list(graph_data.keys())}")

        return g.to(device)
    
    def _apply_dataset_balancing(self):
        """Apply 1:1 positive:negative sampling to balance dataset"""
        import random
        from collections import defaultdict
        
        print("Applying dataset balancing (1:1 positive:negative ratio)...")
        
        # Separate positive and negative samples
        positive_indices = []
        negative_indices = []
        
        for i, label in enumerate(self.ppi_labels):
            # Check if this is a positive sample (classes 1-6 have value 1)
            # Negative samples have class 7 (inactivate) as negative
            is_positive = any(label[j] == 1 for j in range(0, 7))  # Classes 0-6 are positive
            is_negative = label[7] == 1  # Class 7 (inactivate) is negative
            
            if is_positive:
                positive_indices.append(i)
            elif is_negative:
                negative_indices.append(i)
        
        print(f"Original dataset: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            print("Warning: Cannot balance dataset - one class is empty")
            return
        
        # Calculate target sizes (1:1 ratio)
        target_positive = min(len(positive_indices), len(negative_indices))
        target_negative = target_positive
        
        # Sample negative instances
        if len(negative_indices) > target_negative:
            negative_indices = random.sample(negative_indices, target_negative)
        elif len(positive_indices) > target_positive:
            positive_indices = random.sample(positive_indices, target_positive)
        
        # Combine and shuffle
        balanced_indices = positive_indices + negative_indices
        random.shuffle(balanced_indices)
        
        # Create new dataset
        balanced_ppi_list = []
        balanced_ppi_labels = []
        
        for idx in balanced_indices:
            balanced_ppi_list.append(self.ppi_list[idx])
            balanced_ppi_labels.append(self.ppi_labels[idx])
        
        # Create new protein ID mapping (keep existing for simplicity)
        balanced_protein_names = set()
        for pair in balanced_ppi_list:
            balanced_protein_names.add(pair[0])
            balanced_protein_names.add(pair[1])
        
        balanced_protein_ids = list(balanced_protein_names)
        
        # Update dataset
        self.ppi_list = balanced_ppi_list
        self.ppi_labels = balanced_ppi_labels
        
        print(f"Balanced dataset: {len(balanced_ppi_list)} samples")
        print(f"Balanced proteins: {len(balanced_protein_ids)}")
        
        # Optional: Update split ratios for training (8:1:1 as requested)
        self._update_split_ratios()
    
    def _update_split_ratios(self):
        """Update split ratios to 8:1:1 (train:val:test)"""
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # First create initial indices
        indices = np.arange(len(self.ppi_list))
        
        # Split test set (1/10 = 10%)
        train_val_indices, test_indices = train_test_split(
            indices, test_size=0.1, random_state=self.config.get('other', {}).get('seed', 42)
        )
        
        # Split remaining into train and val (8:1 of 90% = 8:1)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.111,  # Use 11.1% directly for val set (1:8 ratio)
            random_state=self.config.get('other', {}).get('seed', 42)
        )
        
        # Store split information
        self.split_dict = {
            'train_index': train_indices.tolist(),
            'val_index': val_indices.tolist(), 
            'test_index': test_indices.tolist()
        }
        
        print(f"Updated split ratios:")
        print(f"  Train: {len(train_indices)} ({len(train_indices)/len(indices)*100:.1f}%)")
        print(f"  Val: {len(val_indices)} ({len(val_indices)/len(indices)*100:.1f}%)")
    
    def _protein_positive_set_balancing(self):
        """
        Apply protein-level positive set balancing with 8:1:1 split
        Validation set contains proteins not seen in training
        Test set contains all negative interactions
        """
        import random
        from collections import defaultdict
        
        print("Applying protein-level positive set balancing (8:1:1 split)...")
        
        # Create reverse mapping from index to protein ID
        idx_to_protein_id = {v: k for k, v in self.protein_name_to_id.items()}
        
        # Separate positive and negative interactions
        positive_interactions = []
        negative_interactions = []
        
        for i, label in enumerate(self.ppi_labels):
            # Check if this is a positive sample (classes 1-6 have value 1)
            # Negative samples have class 7 (inactivate) as negative
            is_positive = any(label[j] == 1 for j in range(0, 7))  # Classes 0-6 are positive
            is_negative = label[7] == 1  # Class 7 (inactivate) is negative
            
            # Convert protein indices to protein IDs
            protein1_idx, protein2_idx = self.ppi_list[i]
            protein1_id = idx_to_protein_id[protein1_idx]
            protein2_id = idx_to_protein_id[protein2_idx]
            
            interaction = ([protein1_id, protein2_id], label, i)
            
            if is_positive:
                positive_interactions.append(interaction)
            elif is_negative:
                negative_interactions.append(interaction)
        
        print(f"Original dataset: {len(positive_interactions)} positive, {len(negative_interactions)} negative")
        
        if len(positive_interactions) == 0:
            print("Warning: No positive interactions found for protein-level splitting")
            return
        
        # Group positive interactions by protein
        protein_positive_interactions = defaultdict(list)
        
        for interaction in positive_interactions:
            protein_pair, label, idx = interaction
            protein1_id, protein2_id = protein_pair
            
            # Add interaction to both proteins' lists
            protein_positive_interactions[protein1_id].append(interaction)
            protein_positive_interactions[protein2_id].append(interaction)
        
        print(f"Found {len(protein_positive_interactions)} unique proteins with positive interactions")
        
        # Sort proteins by residue count from smallest to largest
        sorted_proteins = self._sort_proteins_by_residue_count(protein_positive_interactions.keys())
        
        # Calculate split sizes
        n_proteins = len(sorted_proteins)
        train_size = int(n_proteins * 0.8)
        val_size = int(n_proteins * 0.1)
        test_size = n_proteins - train_size - val_size
        
        # Split proteins into train, val, test
        train_proteins = sorted_proteins[:train_size]
        val_proteins = sorted_proteins[train_size:train_size + val_size]
        test_proteins = sorted_proteins[train_size + val_size:]
        
        print(f"Protein split: train={len(train_proteins)}, val={len(val_proteins)}, test={len(test_proteins)}")
        
        # Collect interactions for each split
        train_interactions = set()
        val_interactions = set()
        test_interactions = set()
        
        # Add positive interactions based on protein membership
        for protein in train_proteins:
            for interaction in protein_positive_interactions[protein]:
                train_interactions.add(interaction[2])  # Store original index
        
        for protein in val_proteins:
            for interaction in protein_positive_interactions[protein]:
                val_interactions.add(interaction[2])
        
        for protein in test_proteins:
            for interaction in protein_positive_interactions[protein]:
                test_interactions.add(interaction[2])
        
        # Remove any overlaps (interactions appearing in multiple sets)
        # This ensures strict separation by protein
        train_interactions = train_interactions - val_interactions - test_interactions
        val_interactions = val_interactions - train_interactions - test_interactions
        test_interactions = test_interactions - train_interactions - val_interactions
        
        print(f"Positive interactions by split: train={len(train_interactions)}, val={len(val_interactions)}, test={len(test_interactions)}")
        
        # Sample negative interactions to match positive counts
        target_negative = min(len(train_interactions), len(val_interactions), len(test_interactions))
        
        if len(negative_interactions) > target_negative:
            # Sample negative interactions for each split
            negative_indices = [interaction[2] for interaction in negative_interactions]
            
            # Calculate how many negatives to add to each split
            train_neg_count = min(len(train_interactions), target_negative)
            val_neg_count = min(len(val_interactions), target_negative)
            test_neg_count = min(len(test_interactions), target_negative)
            
            # Sample negative interactions
            if len(negative_indices) >= train_neg_count + val_neg_count + test_neg_count:
                sampled_negatives = random.sample(negative_indices, train_neg_count + val_neg_count + test_neg_count)
                
                train_negatives = sampled_negatives[:train_neg_count]
                val_negatives = sampled_negatives[train_neg_count:train_neg_count + val_neg_count]
                test_negatives = sampled_negatives[train_neg_count + val_neg_count:]
                
                # Add negatives to each split
                train_interactions.update(train_negatives)
                val_interactions.update(val_negatives)
                test_interactions.update(test_negatives)
                
                print(f"Added negatives: train={len(train_negatives)}, val={len(val_negatives)}, test={len(test_negatives)}")
            else:
                print("Warning: Not enough negative interactions for balanced sampling")
                # Use all negatives and distribute them
                all_negatives = set(negative_indices)
                train_interactions.update(all_negatives)
        else:
            # Not enough negatives, just use all
            negative_indices = [interaction[2] for interaction in negative_interactions]
            train_interactions.update(negative_indices)
            print(f"Added all {len(negative_indices)} negative interactions to training set")
        
        # Convert back to lists and sort
        train_indices = sorted(list(train_interactions))
        val_indices = sorted(list(val_interactions))
        test_indices = sorted(list(test_interactions))
        
        # Create new balanced dataset with protein IDs converted back to indices
        balanced_ppi_list = []
        balanced_ppi_labels = []
        
        # Create new protein ID mapping for the balanced dataset
        balanced_protein_name_to_id = {}
        
        for idx in train_indices + val_indices + test_indices:
            # Convert protein IDs back to indices using the new mapping
            protein_pair, label, _ = positive_interactions[idx] if idx < len(positive_interactions) else negative_interactions[idx - len(positive_interactions)]
            protein1_id, protein2_id = protein_pair
            
            # Add proteins to the new mapping if they don't exist
            if protein1_id not in balanced_protein_name_to_id:
                balanced_protein_name_to_id[protein1_id] = len(balanced_protein_name_to_id)
            if protein2_id not in balanced_protein_name_to_id:
                balanced_protein_name_to_id[protein2_id] = len(balanced_protein_name_to_id)
            
            # Get new indices for the proteins
            protein1_idx = balanced_protein_name_to_id[protein1_id]
            protein2_idx = balanced_protein_name_to_id[protein2_id]
            
            balanced_ppi_list.append([protein1_idx, protein2_idx])
            balanced_ppi_labels.append(label)
        
        # Update dataset
        self.ppi_list = balanced_ppi_list
        self.ppi_labels = balanced_ppi_labels
        self.protein_name_to_id = balanced_protein_name_to_id
        
        # Store split information (indices relative to the new balanced dataset)
        n_train = len(train_indices)
        n_val = len(val_indices)
        n_test = len(test_indices)
        
        self.split_dict = {
            'train_index': list(range(n_train)),
            'val_index': list(range(n_train, n_train + n_val)),
            'test_index': list(range(n_train + n_val, n_train + n_val + n_test))
        }
        
        print(f"Final balanced dataset: {len(balanced_ppi_list)} samples")
        print(f"Final split: train={n_train}, val={n_val}, test={n_test}")
        
        # Verify protein separation
        self._verify_protein_separation(train_proteins, val_proteins, test_proteins)
    
    def _sort_proteins_by_residue_count(self, protein_ids):
        """
        Sort proteins by residue count from smallest to largest
        
        Args:
            protein_ids: List of protein IDs to sort
            
        Returns:
            List of protein IDs sorted by residue count
        """
        # Dictionary to store protein ID and residue count
        protein_residue_counts = {}
        
        for protein_id in protein_ids:
            try:
                # Get PDB file path
                pdb_file = os.path.join(self.pdb_dir, f"{protein_id}.pdb")
                
                if os.path.exists(pdb_file):
                    # Read CA atoms from PDB to get residue count
                    coords, residue_names = read_ca_atoms_from_pdb(pdb_file)
                    residue_count = len(residue_names)
                    protein_residue_counts[protein_id] = residue_count
                else:
                    # If PDB file doesn't exist, use a default value
                    protein_residue_counts[protein_id] = 0
                    print(f"Warning: PDB file not found for {protein_id}, using residue count 0")
            except Exception as e:
                # If there's an error reading the PDB file, use default value
                protein_residue_counts[protein_id] = 0
                print(f"Warning: Error reading PDB file for {protein_id}: {str(e)}, using residue count 0")
        
        # Sort proteins by residue count (ascending order)
        sorted_proteins = sorted(protein_residue_counts.keys(), key=lambda x: protein_residue_counts[x])
        
        # Print some debug information
        print(f"Sorted {len(sorted_proteins)} proteins by residue count:")
        for i, protein_id in enumerate(sorted_proteins[:5]):  # Show first 5
            print(f"  {i+1}. {protein_id}: {protein_residue_counts[protein_id]} residues")
        if len(sorted_proteins) > 5:
            print(f"  ... and {len(sorted_proteins) - 5} more proteins")
        
        return sorted_proteins
    
    def _verify_protein_separation(self, train_proteins, val_proteins, test_proteins):
        """Verify that proteins are properly separated between splits"""
        train_set = set(train_proteins)
        val_set = set(val_proteins)
        test_set = set(test_proteins)
        
        # Check for overlaps
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap:
            print(f"⚠️  WARNING: {len(train_val_overlap)} proteins overlap between train and val sets")
        else:
            print("✅ Train and validation sets have no protein overlap")
        
        if train_test_overlap:
            print(f"⚠️  WARNING: {len(train_test_overlap)} proteins overlap between train and test sets")
        else:
            print("✅ Train and test sets have no protein overlap")
        
        if val_test_overlap:
            print(f"⚠️  WARNING: {len(val_test_overlap)} proteins overlap between val and test sets")
        else:
            print("✅ Validation and test sets have no protein overlap")
        
        # Check if validation set contains proteins not seen in training
        val_unseen = val_set - train_set
        if val_unseen:
            print(f"✅ Validation set contains {len(val_unseen)} proteins not seen in training")
        else:
            print("⚠️  WARNING: Validation set contains no new proteins")
        
        # Check protein counts in each split
        print(f"Protein distribution: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    def _load_raw_ppi_data(self) -> Tuple[Dict, List, List]:
        """Load PPI data from files"""
        protein_name_to_id = {}
        ppi_list = []
        ppi_labels = []
        
        # Class mapping
        class_map = {
            'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3,
            'inhibition': 4, 'catalysis': 5, 'expression': 6 , 'inactivate': 7
        }
        
        # Load protein names
        with open(self.protein_seq_file, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if len(row) >= 1:
                    protein_name_to_id[row[0]] = idx
        
        # Load PPI interactions
        ppi_dict = {}
        ppi_name = 0

        with open(self.ppi_file, 'r') as f:
            skip_header = True
            for i, line in enumerate(f):
                if skip_header:
                    skip_header = False
                    continue

                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue

                # Format: item_id_a  item_id_b  mode  action  ...
                prot1, prot2 = parts[0], parts[1]
                
                # Get interaction type from mode (column 2) or action (column 3)
                # Try mode first, then action
                interaction_type = parts[2].strip() if parts[2].strip() else parts[3].strip()
                
                # If mode is empty, use action
                if not interaction_type or interaction_type == '':
                    interaction_type = parts[3].strip() if len(parts) > 3 else ''
                
                # Skip if proteins not in dictionary
                if prot1 not in protein_name_to_id or prot2 not in protein_name_to_id:
                    continue
                pair_name = f"{prot1}__{prot2}"
                

                # Add or update interaction
                if pair_name not in ppi_dict:
                    ppi_dict[pair_name] = ppi_name
                    label = [0] * 8  # Initialize all zeros (negative sample)
                    if interaction_type in class_map:
                        label[class_map[interaction_type]] = 1  # Set positive class
                    elif interaction_type == '' or interaction_type is None:
                        # Empty action field - treat as negative sample (class 0 = reaction)
                        label[class_map["inactivate"]] = 1  # Set reaction class as negative

                    ppi_labels.append(label)
                    ppi_name += 1
                else:
                    # Update existing label
                    idx = ppi_dict[pair_name]
                    if interaction_type in class_map:
                        ppi_labels[idx][class_map[interaction_type]] = 1
        
        # Convert to list format
        for pair_name in ppi_dict.keys():
            prot1, prot2 = pair_name.split('__')
            ppi_list.append([protein_name_to_id[prot1], protein_name_to_id[prot2]])
        
        return protein_name_to_id, ppi_list, ppi_labels
    
    def detect_actual_classes(self) -> Tuple[int, Dict[str, int]]:
        """
        Detect actual number of classes present in the dataset
        
        Returns:
            Tuple of (actual_num_classes, actual_class_map)
        """
        # Get labels as numpy array for analysis
        labels_array = np.array(self.ppi_labels)
        
        # Check which classes actually have samples
        class_presence = np.any(labels_array == 1, axis=0)
        
        # Get indices of classes that are present
        present_class_indices = np.where(class_presence)[0]
        
        # Original class mapping for reference
        original_class_map = {
            'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3,
            'inhibition': 4, 'catalysis': 5, 'expression': 6, 'inactivate': 7
        }
        
        # Create new mapping for actual present classes
        actual_class_map = {}
        for i, class_idx in enumerate(present_class_indices):
            # Find the class name for this index
            for class_name, idx in original_class_map.items():
                if idx == class_idx:
                    actual_class_map[class_name] = i
                    break
        
        actual_num_classes = len(present_class_indices)
        
        # Log detection results
        print(f"\n🔍 Dataset Class Detection Results:")
        print(f"   Original classes: 8")
        print(f"   Actual classes present: {actual_num_classes}")
        print(f"   Present class indices: {present_class_indices.tolist()}")
        
        if actual_num_classes > 0:
            print(f"   Actual class mapping:")
            for class_name, new_idx in actual_class_map.items():
                print(f"     {class_name} (original idx: {original_class_map[class_name]}) -> new idx: {new_idx}")
        
        # For edge case: if no classes are present, use single class (binary classification)
        if actual_num_classes == 0:
            print("⚠️  Warning: No classes detected in dataset. Defaulting to binary classification.")
            actual_num_classes = 1
            actual_class_map = {'interaction': 0}
        
        return actual_num_classes, actual_class_map
    
    def get_actual_labels_tensor(self, actual_num_classes: int, actual_class_map: Dict[str, int]) -> torch.Tensor:
        """
        Convert labels to new tensor with actual number of classes
        
        Args:
            actual_num_classes: Number of classes actually present
            actual_class_map: Mapping from original class names to new indices
        
        Returns:
            Labels tensor with adjusted dimensions
        """
        if actual_num_classes == 8:
            # No need to change, return original labels
            return torch.FloatTensor(np.array(self.ppi_labels)).to(device)
        
        # Convert labels to new format
        new_labels = []
        original_labels = np.array(self.ppi_labels)
        
        # Create reverse mapping from original index to new index
        reverse_class_map = {}
        original_class_map = {
            'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3,
            'inhibition': 4, 'catalysis': 5, 'expression': 6, 'inactivate': 7
        }
        
        for class_name, new_idx in actual_class_map.items():
            original_idx = original_class_map[class_name]
            reverse_class_map[original_idx] = new_idx
        
        for i in range(len(original_labels)):
            new_label = [0] * actual_num_classes
            
            # Copy existing labels to new positions
            for original_idx, new_idx in reverse_class_map.items():
                if original_labels[i][original_idx] == 1:
                    new_label[new_idx] = 1
            
            new_labels.append(new_label)
        
        return torch.FloatTensor(np.array(new_labels)).to(device)
    
    
    def split_dataset(self, split_mode: str = 'random', 
                     split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                     seed: int = 42) -> Dict[str, List[int]]:
        """
        Split dataset into train/val/test
        
        Args:
            split_mode: 'random', 'bfs', or 'dfs'
            split_ratio: (train, val, test) ratios
            seed: Random seed
        
        Returns:
            Dictionary with 'train_index', 'val_index', 'test_index'
        """
        # If split_dict already exists (from balancing), return it directly
        if hasattr(self, 'split_dict') and self.split_dict is not None:
            print("Using existing split from dataset balancing")
            return self.split_dict
        
        random.seed(seed)
        np.random.seed(seed)
        
        ppi_num = len(self.ppi_list)
        
        # Ensure minimum samples for val/test sets
        min_val_test = min(5, ppi_num // 10)  # At least 5 or 10% of data
        min_val_test = max(1, min_val_test)   # At least 1 sample
        
        if split_mode == 'random':
            indices = list(range(ppi_num))
            random.shuffle(indices)
            
            # Calculate split sizes
            val_size = max(int(ppi_num * split_ratio[1]), min_val_test)
            test_size = max(int(ppi_num * split_ratio[2]), min_val_test)
            train_size = ppi_num - val_size - test_size
            
            # Ensure train_size is positive
            if train_size <= 0:
                print("Warning: Dataset too small for three-way split. Using 50/25/25 split")
                train_size = max(1, ppi_num // 2)
                val_size = max(1, (ppi_num - train_size) // 2)
                test_size = ppi_num - train_size - val_size
            
            train_end = train_size
            val_end = train_size + val_size
            
            return {
                'train_index': indices[:train_end],
                'val_index': indices[train_end:val_end],
                'test_index': indices[val_end:]
            }
        
        elif split_mode in ['bfs', 'dfs']:
            # Define minimum validation/test sizes for graph-based splits too
            min_val_test = max(10, ppi_num // 10)  # At least 10 or 10% of data
            
            result = self._graph_based_split(split_mode, split_ratio, seed)
            print(f"split_mode:{split_mode},_graph_based_split:{result}")
            
            # Check if graph-based split is reasonable
            val_samples = len(result['val_index'])
            test_samples = len(result['test_index'])
            
            if val_samples < min_val_test or test_samples < min_val_test:
                print(f"Warning: Graph-based split resulted in too few samples (Val: {val_samples}, Test: {test_samples})")
                print(f"Falling back to random split for better data distribution")
                
                # Fall back to random split
                indices = list(range(ppi_num))
                random.shuffle(indices)
                
                train_end = int(ppi_num * split_ratio[0])
                val_end = int(ppi_num * (split_ratio[0] + split_ratio[1]))
                
                # Ensure minimum sizes
                min_size = max(min_val_test, int(ppi_num * 0.1))
                if train_end < min_size:
                    val_end = int(ppi_num * (1 - 0.2))  # Ensure at least 20% for test
                    train_end = min_size
                    
                result = {
                    'train_index': indices[:train_end],
                    'val_index': indices[train_end:val_end],
                    'test_index': indices[val_end:]
                }
                
                print(f"Random split - Train: {len(result['train_index'])}, Val: {len(result['val_index'])}, Test: {len(result['test_index'])}")
            
            return result
        else:
            raise ValueError(f"Split mode {split_mode} not supported. Use 'random', 'bfs', or 'dfs'")
    
    def _graph_based_split(self, split_mode: str, 
                         split_ratio: Tuple[float, float, float], 
                         seed: int) -> Dict[str, List[int]]:
        """
        Graph-based splitting using BFS or DFS
        
        Args:
            split_mode: 'bfs' or 'dfs'
            split_ratio: (train, val, test) ratios
            seed: Random seed
        
        Returns:
            Dictionary with split indices
        """
        random.seed(seed)
        np.random.seed(seed)
        
        ppi_num = len(self.ppi_list)
        
        # Build node-to-edge mapping
        node_to_edge_index = {}
        for i, edge in enumerate(self.ppi_list):
            node1, node2 = edge[0], edge[1]
            
            if node1 not in node_to_edge_index:
                node_to_edge_index[node1] = []
            if node2 not in node_to_edge_index:
                node_to_edge_index[node2] = []
                
            node_to_edge_index[node1].append(i)
            node_to_edge_index[node2].append(i)
        
        node_num = len(node_to_edge_index)
        
        # Calculate subgraph size (40% of edges for test/val, following original implementation)
        sub_graph_size = int(ppi_num * 0.4)
        
        if split_mode == 'bfs':
            selected_edge_indices = self._get_bfs_subgraph(ppi_num, node_to_edge_index, sub_graph_size)
        elif split_mode == 'dfs':
            selected_edge_indices = self._get_dfs_subgraph(ppi_num, node_to_edge_index, sub_graph_size)
        
        # Remaining edges are for training
        all_edge_indices = set(range(ppi_num))
        unselected_edge_indices = list(all_edge_indices.difference(set(selected_edge_indices)))
        
        # Shuffle selected edges for val/test split
        selected_list = list(selected_edge_indices)
        random.shuffle(selected_list)
        
        val_size = int(len(selected_list) * (split_ratio[1] / (split_ratio[1] + split_ratio[2])))
        
        return {
            'train_index': unselected_edge_indices,
            'val_index': selected_list[:val_size],
            'test_index': selected_list[val_size:]
        }
    
    def _get_bfs_subgraph(self, ppi_num: int, 
                         node_to_edge_index: Dict[int, List[int]], 
                         sub_graph_size: int) -> List[int]:
        """
        Breadth-First Search subgraph selection
        
        Args:
            ppi_num: Total number of PPIs
            node_to_edge_index: Node to edge mapping
            sub_graph_size: Target subgraph size
        
        Returns:
            List of selected edge indices
        """
        candidate_nodes = []
        selected_edge_indices = []
        selected_nodes = []
        
        # Start with a random node that has reasonable connectivity
        # Use actual protein nodes instead of PPI indices
        node_keys = list(node_to_edge_index.keys())
        if not node_keys:
            return []
        random_node = random.choice(node_keys)
        while len(node_to_edge_index.get(random_node, [])) > 20:
            random_node = random.choice(node_keys)
        
        candidate_nodes.append(random_node)
        
        while len(selected_edge_indices) < sub_graph_size and candidate_nodes:
            current_node = candidate_nodes.pop(0)
            selected_nodes.append(current_node)
            
            # Process edges connected to current node
            for edge_index in node_to_edge_index.get(current_node, []):
                if edge_index not in selected_edge_indices:
                    selected_edge_indices.append(edge_index)
                    
                    # Find connected node
                    edge = self.ppi_list[edge_index]
                    if edge[0] == current_node:
                        connected_node = edge[1]
                    else:
                        connected_node = edge[0]
                    
                    # Add to candidate list if not already selected
                    if connected_node not in selected_nodes and connected_node not in candidate_nodes:
                        candidate_nodes.append(connected_node)
        
        return selected_edge_indices
    
    def _get_dfs_subgraph(self, ppi_num: int, 
                         node_to_edge_index: Dict[int, List[int]], 
                         sub_graph_size: int) -> List[int]:
        """
        Depth-First Search subgraph selection
        
        Args:
            ppi_num: Total number of PPIs
            node_to_edge_index: Node to edge mapping
            sub_graph_size: Target subgraph size
        
        Returns:
            List of selected edge indices
        """
        stack = []
        selected_edge_indices = []
        selected_nodes = []
        
        # Start with a random node that has reasonable connectivity
        # Use actual protein nodes instead of PPI indices
        node_keys = list(node_to_edge_index.keys())
        if not node_keys:
            return []
        random_node = random.choice(node_keys)
        while len(node_to_edge_index.get(random_node, [])) > 20:
            random_node = random.choice(node_keys)
        
        stack.append(random_node)
        
        while len(selected_edge_indices) < sub_graph_size and stack:
            current_node = stack[-1]
            
            if current_node in selected_nodes:
                # Current node already processed, try to find unvisited neighbor
                found_unvisited = False
                
                for edge_index in node_to_edge_index.get(current_node, []):
                    edge = self.ppi_list[edge_index]
                    if edge[0] == current_node:
                        connected_node = edge[1]
                    else:
                        connected_node = edge[0]
                    
                    if connected_node not in selected_nodes:
                        stack.append(connected_node)
                        found_unvisited = True
                        break
                
                if not found_unvisited:
                    stack.pop()
                continue
            
            else:
                # Process current node
                selected_nodes.append(current_node)
                
                # Add all edges connected to this node
                for edge_index in node_to_edge_index.get(current_node, []):
                    if edge_index not in selected_edge_indices:
                        selected_edge_indices.append(edge_index)
        
        return selected_edge_indices
    
    def get_labels_tensor(self) -> torch.Tensor:
        """Get PPI labels as tensor"""
        return torch.FloatTensor(np.array(self.ppi_labels)).to(device)


def validate_data_consistency(protein_ids: List[str], 
                           pdb_dir: str, 
                           encoding_config: Dict,
                           protein_seq_file: Optional[str] = None,
                           verbose: bool = True) -> Tuple[List[str], Dict[str, Any]]:
    """
    Validate data consistency before processing
    
    Args:
        protein_ids: List of protein identifiers to validate
        pdb_dir: Directory containing PDB files
        encoding_config: Encoding configuration
        protein_seq_file: Optional protein sequence file
        verbose: Whether to print progress
    
    Returns:
        Tuple of (valid_protein_ids, validation_report)
    """
    if verbose:
        print("Starting data consistency validation...")
    
    validation_report = {
        'total_proteins': len(protein_ids),
        'valid_proteins': 0,
        'invalid_proteins': 0,
        'missing_pdb': [],
        'missing_embedding': [],
        'invalid_pdb': [],
        'dimension_mismatch': []
    }
    
    valid_proteins = []
    
    # Check PDB file availability
    if verbose:
        print("Checking PDB file availability...")
    
    available_pdbs = set()
    if os.path.exists(pdb_dir):
        available_pdbs = set(f[:-4] for f in os.listdir(pdb_dir) if f.endswith('.pdb'))
    
    # Check embedding file availability for precomputed encoding (parallel check)
    encoding_type = encoding_config.get('encoding_type', 'mape')
    embedding_files = set()
    
    if encoding_type in ['precomputed', 'alphafold']:
        embedding_dir = encoding_config.get('embedding_dir')
        
        if embedding_dir and os.path.exists(embedding_dir):
            embedding_files = set(f[:-4] for f in os.listdir(embedding_dir) if f.endswith('.npy'))
    
    # Parallel check: both PDB and embedding must exist
    for protein_id in protein_ids:
        has_pdb = protein_id in available_pdbs
        has_embedding = protein_id in embedding_files if encoding_type in ['precomputed', 'alphafold'] else True
        
        if not has_pdb:
            validation_report['missing_pdb'].append(protein_id)
            validation_report['invalid_proteins'] += 1
            if verbose and len(validation_report['missing_pdb']) <= 10:
                print(f"  Missing PDB for {protein_id}")
        elif not has_embedding:
            validation_report['missing_embedding'].append(protein_id)
            validation_report['invalid_proteins'] += 1
            if verbose and len(validation_report['missing_embedding']) <= 10:
                print(f"  Missing embedding for {protein_id}")
        else:
            validation_report['valid_proteins'] += 1
            valid_proteins.append(protein_id)
    
    # Validate embedding dimension consistency for ALL precomputed encoding files
    if encoding_type in ['precomputed', 'alphafold'] and embedding_dir and os.path.exists(embedding_dir):
        if verbose:
            print("Validating ALL embedding dimensions...")
        
        # Check dimensions for ALL valid proteins
        truly_valid = []
        for protein_id in valid_proteins:
            embedding_file = os.path.join(embedding_dir, f"{protein_id}.npy")
            try:
                embedding = np.load(embedding_file)
                if len(embedding.shape) != 2:
                    validation_report['dimension_mismatch'].append(protein_id)
                    validation_report['invalid_proteins'] += 1
                    validation_report['valid_proteins'] -= 1
                    if verbose:
                        print(f"  Invalid embedding shape for {protein_id}: {embedding.shape}")
                else:
                    truly_valid.append(protein_id)
                    if verbose and len(truly_valid) <= 5:  # Show first 5 as sample
                        print(f"  Valid embedding for {protein_id}: shape={embedding.shape}")
            except Exception as e:
                validation_report['dimension_mismatch'].append(protein_id)
                validation_report['invalid_proteins'] += 1
                validation_report['valid_proteins'] -= 1
                if verbose:
                    print(f"  Error reading embedding for {protein_id}: {str(e)}")
        
        valid_proteins = truly_valid
    
    # Validate PDB file integrity (sample check)
    if verbose and valid_proteins:
        print("Validating PDB file integrity...")
    
    sample_size = min(10, len(valid_proteins))
    sample_proteins = valid_proteins[:sample_size] if valid_proteins else []
    
    for protein_id in sample_proteins:
        pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
        try:
            atoms, residues = read_ca_atoms_from_pdb(pdb_file)
            if len(atoms) == 0:
                validation_report['invalid_pdb'].append(protein_id)
                validation_report['invalid_proteins'] += 1
                validation_report['valid_proteins'] -= 1
                valid_proteins.remove(protein_id)
                if verbose:
                    print(f"  Invalid PDB structure for {protein_id}: no CA atoms found")
        except Exception as e:
            validation_report['invalid_pdb'].append(protein_id)
            validation_report['invalid_proteins'] += 1
            validation_report['valid_proteins'] -= 1
            if protein_id in valid_proteins:
                valid_proteins.remove(protein_id)
            if verbose:
                print(f"  Error reading PDB for {protein_id}: {str(e)}")
    
    # Validate FASTA sequence consistency with PDB files
    if protein_seq_file and os.path.exists(protein_seq_file) and valid_proteins:
        if verbose:
            print("Validating FASTA-PDB consistency...")
        
        # Initialize FASTA-related validation items
        validation_report['missing_fasta'] = []
        validation_report['fasta_pdb_mismatch'] = []
        
        # Load FASTA sequences
        sequences = {}
        try:
            with open(protein_seq_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        protein_id = row[0].strip()
                        sequence = row[1].strip()
                        sequences[protein_id] = sequence
        except Exception as e:
            if verbose:
                print(f"  Error loading FASTA sequences: {str(e)}")
        
        # Check sequence consistency for ALL valid proteins
        truly_valid = []
        for protein_id in valid_proteins:
            if protein_id in sequences:
                fasta_length = len(sequences[protein_id])
                
                # Get PDB residue count
                pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
                try:
                    atoms, residues = read_ca_atoms_from_pdb(pdb_file)
                    pdb_residue_count = len(residues)
                    
                    # Check if FASTA sequence length matches PDB residue count
                    if fasta_length != pdb_residue_count:
                        validation_report['fasta_pdb_mismatch'].append(protein_id)
                        validation_report['invalid_proteins'] += 1
                        validation_report['valid_proteins'] -= 1
                        if verbose:
                            print(f"  FASTA-PDB mismatch for {protein_id}: FASTA={fasta_length}, PDB={pdb_residue_count}")
                    else:
                        truly_valid.append(protein_id)
                        if verbose and len(truly_valid) <= 5:
                            print(f"  Consistent FASTA-PDB for {protein_id}: length={fasta_length}")
                except Exception as e:
                    validation_report['fasta_pdb_mismatch'].append(protein_id)
                    validation_report['invalid_proteins'] += 1
                    validation_report['valid_proteins'] -= 1
                    if verbose:
                        print(f"  Error reading PDB for FASTA check {protein_id}: {str(e)}")
            else:
                # Protein not found in FASTA file
                validation_report['missing_fasta'].append(protein_id)
                validation_report['invalid_proteins'] += 1
                validation_report['valid_proteins'] -= 1
                if verbose:
                    print(f"  Missing FASTA sequence for {protein_id}")
        
        valid_proteins = truly_valid
    
    # Final validation summary
    if verbose:
        print(f"\nValidation Summary:")
        print(f"  Total proteins: {validation_report['total_proteins']}")
        print(f"  Valid proteins: {validation_report['valid_proteins']}")
        print(f"  Invalid proteins: {validation_report['invalid_proteins']}")
        
        if validation_report['missing_pdb']:
            print(f"  Missing PDB files: {len(validation_report['missing_pdb'])}")
            if len(validation_report['missing_pdb']) > 10:
                print(f"    (showing first 10) {validation_report['missing_pdb'][:10]}")
        
        if validation_report['missing_embedding']:
            print(f"  Missing embedding files: {len(validation_report['missing_embedding'])}")
            if len(validation_report['missing_embedding']) > 10:
                print(f"    (showing first 10) {validation_report['missing_embedding'][:10]}")
        
        if validation_report['invalid_pdb']:
            print(f"  Invalid PDB files: {len(validation_report['invalid_pdb'])}")
        
        if validation_report['dimension_mismatch']:
            print(f"  Dimension mismatch files: {len(validation_report['dimension_mismatch'])}")
            if len(validation_report['dimension_mismatch']) > 10:
                print(f"    (showing first 10) {validation_report['dimension_mismatch'][:10]}")
        
        if 'missing_fasta' in validation_report and validation_report['missing_fasta']:
            print(f"  Missing FASTA sequences: {len(validation_report['missing_fasta'])}")
            if len(validation_report['missing_fasta']) > 10:
                print(f"    (showing first 10) {validation_report['missing_fasta'][:10]}")
        
        if 'fasta_pdb_mismatch' in validation_report and validation_report['fasta_pdb_mismatch']:
            print(f"  FASTA-PDB mismatches: {len(validation_report['fasta_pdb_mismatch'])}")
            if len(validation_report['fasta_pdb_mismatch']) > 10:
                print(f"    (showing first 10) {validation_report['fasta_pdb_mismatch'][:10]}")
    
    return valid_proteins, validation_report


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024


def filter_ppi_data(ppi_list: List[List[int]], 
                   labels: List[List[int]], 
                   valid_proteins: List[str],
                   protein_id_to_idx: Dict[str, int],
                   verbose: bool = True) -> Tuple[List[List[int]], List[List[int]], Dict[str, int]]:
    """
    Filter PPI data to only include valid proteins
    
    Args:
        ppi_list: List of protein pairs (as indices)
        labels: List of corresponding labels
        valid_proteins: List of valid protein IDs
        protein_id_to_idx: Mapping from protein ID to original index
        verbose: Whether to print progress
    
    Returns:
        Tuple of (filtered_ppi_list, filtered_labels, new_protein_id_to_idx)
    """
    if verbose:
        print("Filtering PPI data for valid proteins...")
    
    # Create reverse mapping from index to protein ID
    idx_to_protein_id = {v: k for k, v in protein_id_to_idx.items()}
    valid_proteins_set = set(valid_proteins)
    
    # Create new mapping from valid protein IDs to new indices
    new_protein_id_to_idx = {}
    for protein_id in valid_proteins:
        new_protein_id_to_idx[protein_id] = len(new_protein_id_to_idx)
    
    # Filter PPIs to only include valid proteins
    filtered_ppis = []
    filtered_labels = []
    
    for i, (protein1_idx, protein2_idx) in enumerate(ppi_list):
        protein1_id = idx_to_protein_id[protein1_idx]
        protein2_id = idx_to_protein_id[protein2_idx]
        
        if protein1_id in valid_proteins_set and protein2_id in valid_proteins_set:
            # Map to new indices using the new mapping
            new_protein1_idx = new_protein_id_to_idx[protein1_id]
            new_protein2_idx = new_protein_id_to_idx[protein2_id]
            
            filtered_ppis.append([new_protein1_idx, new_protein2_idx])
            filtered_labels.append(labels[i])
    
    if verbose:
        original_count = len(ppi_list)
        filtered_count = len(filtered_ppis)
        print(f"  Original PPIs: {original_count}")
        print(f"  Filtered PPIs: {filtered_count}")
        print(f"  Removed PPIs: {original_count - filtered_count}")
        print(f"  Remaining proteins: {len(valid_proteins)}")
        
        # Debug info
        if filtered_count == 0:
            print("  DEBUG: No PPIs survived filtering!")
            print(f"  DEBUG: Sample valid proteins: {list(valid_proteins_set)[:5]}")
            print(f"  DEBUG: Sample PPI pairs: {ppi_list[:5] if ppi_list else 'None'}")
            if ppi_list:
                sample_idx = ppi_list[0][0]
                sample_protein = idx_to_protein_id.get(sample_idx, "NOT_FOUND")
                print(f"  DEBUG: First PPI protein exists in valid set: {sample_protein in valid_proteins_set}")
    
    return filtered_ppis, filtered_labels, new_protein_id_to_idx

