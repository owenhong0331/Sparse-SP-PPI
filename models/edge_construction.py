"""
Edge Construction Module for Protein Graphs
Implements three types of edges:
1. Sequence proximity edges (SEQ): Connect adjacent amino acids in sequence
2. Spatial proximity edges (STR_DIS, STR_KNN): Connect amino acids within spatial distance threshold
3. Surface proximity edges (SURF): Connect neighboring amino acids on protein surface
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from Bio.PDB import PDBParser, SASA
from Bio.PDB.SASA import ShrakeRupley


def euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between two 3D points
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def build_sequence_edges(num_residues: int) -> List[Tuple[int, int]]:
    """
    Build sequence proximity edges connecting adjacent amino acids
    
    Args:
        num_residues: Number of amino acid residues
    
    Returns:
        List of edge tuples (i, j) representing bidirectional edges
    """
    edges = []
    for i in range(num_residues - 1):
        # Bidirectional edges
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    
    return edges


def build_spatial_distance_edges(coords: np.ndarray, threshold: float = 10.0, 
                                  exclude_neighbors: bool = True) -> List[Tuple[int, int]]:
    """
    Build spatial proximity edges based on distance threshold
    
    Args:
        coords: Nx3 array of 3D coordinates (CA atoms)
        threshold: Distance threshold in Angstroms (default: 10.0)
        exclude_neighbors: Whether to exclude sequence neighbors (i-1, i, i+1)
    
    Returns:
        List of edge tuples (i, j)
    """
    edges = []
    num_residues = len(coords)
    
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            # Skip sequence neighbors if requested
            if exclude_neighbors and abs(i - j) <= 1:
                continue
            
            dist = euclidean_distance(coords[i], coords[j])
            if dist < threshold:
                # Bidirectional edges
                edges.append((i, j))
                edges.append((j, i))
    
    return edges


def build_spatial_knn_edges(coords: np.ndarray, k: int = 5, 
                            exclude_neighbors: bool = True) -> List[Tuple[int, int]]:
    """
    Build spatial k-nearest neighbor edges
    
    Args:
        coords: Nx3 array of 3D coordinates (CA atoms)
        k: Number of nearest neighbors
        exclude_neighbors: Whether to exclude sequence neighbors (i-1, i, i+1)
    
    Returns:
        List of edge tuples (i, j)
    """
    num_residues = len(coords)
    
    # Compute pairwise distance matrix
    dist_matrix = np.zeros((num_residues, num_residues))
    for i in range(num_residues):
        for j in range(num_residues):
            dist_matrix[i, j] = euclidean_distance(coords[i], coords[j])
    
    # Find k nearest neighbors for each residue
    edges = []
    for i in range(num_residues):
        # Sort by distance
        sorted_indices = np.argsort(dist_matrix[i])
        
        # Select k neighbors (excluding self and optionally sequence neighbors)
        count = 0
        for j in sorted_indices:
            if j == i:
                continue
            if exclude_neighbors and abs(i - j) <= 1:
                continue
            
            edges.append((i, j))
            count += 1
            
            if count >= k:
                break
    
    return edges


def compute_sasa(pdb_file: str, chain: str = 'A') -> np.ndarray:
    """
    Compute Solvent Accessible Surface Area (SASA) for each residue
    
    Args:
        pdb_file: Path to PDB file
        chain: Chain identifier
    
    Returns:
        Array of SASA values for each residue
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Calculate SASA
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    
    # Extract SASA values for specified chain
    sasa_values = []
    for model in structure:
        for chain_obj in model:
            if chain_obj.id == chain:
                for residue in chain_obj:
                    if residue.id[0] == ' ':  # Standard residue
                        sasa_values.append(residue.sasa)
    
    return np.array(sasa_values)


def identify_surface_residues(sasa_values: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Identify surface residues based on SASA threshold
    
    Args:
        sasa_values: Array of SASA values
        threshold: Relative SASA threshold (0-1)
    
    Returns:
        Boolean array indicating surface residues
    """
    # Normalize SASA values
    max_sasa = np.max(sasa_values)
    if max_sasa > 0:
        relative_sasa = sasa_values / max_sasa
    else:
        relative_sasa = sasa_values
    
    return relative_sasa > threshold


def build_surface_edges(coords: np.ndarray, sasa_values: Optional[np.ndarray] = None,
                        surface_threshold: float = 0.2, distance_threshold: float = 8.0) -> List[Tuple[int, int]]:
    """
    Build surface proximity edges connecting neighboring surface residues
    
    Args:
        coords: Nx3 array of 3D coordinates (CA atoms)
        sasa_values: Array of SASA values (if None, all residues considered surface)
        surface_threshold: Relative SASA threshold for surface identification
        distance_threshold: Distance threshold for surface neighbors (Angstroms)
    
    Returns:
        List of edge tuples (i, j)
    """
    num_residues = len(coords)
    
    # Identify surface residues
    if sasa_values is not None:
        is_surface = identify_surface_residues(sasa_values, surface_threshold)
    else:
        # If no SASA provided, consider all residues as potential surface
        is_surface = np.ones(num_residues, dtype=bool)
    
    # Build edges between surface residues within distance threshold
    edges = []
    surface_indices = np.where(is_surface)[0]
    
    for i in range(len(surface_indices)):
        for j in range(i + 1, len(surface_indices)):
            idx_i = surface_indices[i]
            idx_j = surface_indices[j]
            
            # Skip sequence neighbors
            if abs(idx_i - idx_j) <= 1:
                continue
            
            dist = euclidean_distance(coords[idx_i], coords[idx_j])
            if dist < distance_threshold:
                # Bidirectional edges
                edges.append((idx_i, idx_j))
                edges.append((idx_j, idx_i))
    
    return edges


def build_all_edges(coords: np.ndarray, sasa_values: Optional[np.ndarray] = None,
                   spatial_threshold: float = 10.0, knn_k: int = 5,
                   surface_threshold: float = 0.2, surface_distance: float = 8.0) -> dict:
    """
    Build all types of edges for a protein graph
    
    Args:
        coords: Nx3 array of 3D coordinates (CA atoms)
        sasa_values: Array of SASA values (optional)
        spatial_threshold: Distance threshold for spatial edges
        knn_k: Number of nearest neighbors for KNN edges
        surface_threshold: SASA threshold for surface identification
        surface_distance: Distance threshold for surface edges
    
    Returns:
        Dictionary with edge types as keys and edge lists as values
    """
    num_residues = len(coords)
    
    edges_dict = {
        'SEQ': build_sequence_edges(num_residues),
        'STR_DIS': build_spatial_distance_edges(coords, spatial_threshold),
        'STR_KNN': build_spatial_knn_edges(coords, knn_k),
        'SURF': build_surface_edges(coords, sasa_values, surface_threshold, surface_distance)
    }
    
    return edges_dict


def read_ca_atoms_from_pdb(pdb_file: str, chain: str = 'A') -> Tuple[np.ndarray, List[str]]:
    """
    Read CA atom coordinates and residue names from PDB file
    
    Args:
        pdb_file: Path to PDB file
        chain: Chain identifier
    
    Returns:
        Tuple of (coordinates array, residue names list)
    """
    coords = []
    residue_names = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_type = line[12:16].strip()
                chain_id = line[21:22]
                
                if atom_type == "CA" and chain_id == chain:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    res_name = line[17:20].strip()
                    
                    coords.append([x, y, z])
                    residue_names.append(res_name)
    
    return np.array(coords), residue_names

