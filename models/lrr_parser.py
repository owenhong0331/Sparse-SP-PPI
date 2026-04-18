"""
LRR (Leucine-Rich Repeat) Annotation Parser
Parses LRR region annotations from lrr_annotation_results.txt
"""

import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class LRRRegion:
    """Represents a single LRR region"""
    def __init__(self, region_num: int, start: int, end: int, sequence: str):
        self.region_num = region_num
        self.start = start  # 1-based, inclusive
        self.end = end      # 1-based, inclusive
        self.sequence = sequence
        self.length = end - start + 1
    
    def __repr__(self):
        return f"LRRRegion(num={self.region_num}, start={self.start}, end={self.end}, length={self.length})"
    
    def to_dict(self):
        return {
            'region_num': self.region_num,
            'start': self.start,
            'end': self.end,
            'length': self.length,
            'sequence': self.sequence
        }


class LRRDatabase:
    """Database of LRR annotations for proteins"""
    def __init__(self, annotation_file: str):
        """
        Initialize LRR database from annotation file
        
        Args:
            annotation_file: Path to lrr_annotation_results.txt
        """
        self.annotation_file = annotation_file
        self.protein_lrr_regions: Dict[str, List[LRRRegion]] = defaultdict(list)
        self.protein_full_lengths: Dict[str, int] = {}
        
        print(f"[DEBUG] LRRDatabase.__init__() called with annotation_file: {annotation_file}")
        print(f"[DEBUG] Absolute path: {os.path.abspath(annotation_file)}")
        print(f"[DEBUG] File exists: {os.path.exists(annotation_file)}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        
        if os.path.exists(annotation_file):
            print(f"[DEBUG] Starting to parse LRR annotation file...")
            self._parse_annotation_file()
            print(f"[DEBUG] LRRDatabase initialization completed successfully")
        else:
            print(f"[ERROR] LRR annotation file not found: {annotation_file}")
            print(f"[ERROR] Tried absolute path: {os.path.abspath(annotation_file)}")
            
            # Try to find the file in common locations
            possible_paths = [
                annotation_file,
                os.path.abspath(annotation_file),
                os.path.join(os.getcwd(), annotation_file),
                os.path.join(os.path.dirname(__file__), annotation_file),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), annotation_file)
            ]
            
            print(f"[DEBUG] Searching for file in possible locations:")
            for path in possible_paths:
                exists = os.path.exists(path)
                print(f"[DEBUG]   {path}: {'✓ FOUND' if exists else '✗ NOT FOUND'}")
    
    def _parse_annotation_file(self):
        """Parse the LRR annotation file"""
        print(f"[DEBUG] Parsing LRR annotation file: {self.annotation_file}")
        print(f"[DEBUG] File exists: {os.path.exists(self.annotation_file)}")
        print(f"[DEBUG] Absolute path: {os.path.abspath(self.annotation_file)}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        
        if not os.path.exists(self.annotation_file):
            print(f"[ERROR] LRR annotation file not found at: {self.annotation_file}")
            print(f"[ERROR] Tried absolute path: {os.path.abspath(self.annotation_file)}")
            return
        
        total_lines = 0
        parsed_lines = 0
        error_lines = 0
        
        with open(self.annotation_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab
                parts = line.split('\t')
                if len(parts) < 8:
                    print(f"[WARNING] Line {line_num} has insufficient columns: {len(parts)}")
                    error_lines += 1
                    continue
                
                try:
                    # Parse fields
                    pdb_filename = parts[0]
                    region_num = int(parts[1])
                    start_pos = int(parts[2])
                    end_pos = int(parts[3])
                    seq_length = int(parts[4])
                    full_length = int(parts[5])
                    total_regions = int(parts[6])
                    sequence = parts[7] if len(parts) > 7 else ""
                    
                    # Extract protein ID from PDB filename
                    # Format: "3702.AT1G01210.1.pdb" -> "3702.AT1G01210.1"
                    protein_id = pdb_filename.replace('.pdb', '')
                    
                    # Debug parsing
                    if line_num <= 5:  # Print first 5 lines for debugging
                        print(f"[DEBUG] Line {line_num}: protein_id={protein_id}, start={start_pos}, end={end_pos}")
                    
                    # Create LRR region
                    lrr_region = LRRRegion(region_num, start_pos, end_pos, sequence)
                    
                    # Store in database
                    self.protein_lrr_regions[protein_id].append(lrr_region)
                    self.protein_full_lengths[protein_id] = full_length
                    parsed_lines += 1
                    
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Error parsing line {line_num}: {e}")
                    print(f"[WARNING] Line content: {line}")
                    error_lines += 1
                    continue
        
        print(f"[DEBUG] Parsing completed: total_lines={total_lines}, parsed_lines={parsed_lines}, error_lines={error_lines}")
        print(f"[DEBUG] Loaded LRR annotations for {len(self.protein_lrr_regions)} proteins")
        print(f"[DEBUG] Total LRR regions: {sum(len(regions) for regions in self.protein_lrr_regions.values())}")
        
        # Print sample protein IDs for debugging
        if self.protein_lrr_regions:
            sample_ids = list(self.protein_lrr_regions.keys())[:10]
            print(f"[DEBUG] Sample protein IDs loaded: {sample_ids}")
        else:
            print(f"[ERROR] No protein IDs were loaded from LRR annotation file!")
            
        # Check if specific protein IDs are present
        test_proteins = ['9606.ENSP00000005340', '9606.ENSP00000000233']
        for test_protein in test_proteins:
            if test_protein in self.protein_lrr_regions:
                print(f"[DEBUG] ✓ Found test protein: {test_protein}")
                regions = self.protein_lrr_regions[test_protein]
                print(f"[DEBUG]   LRR regions: {[r.to_dict() for r in regions]}")
            else:
                print(f"[DEBUG] ✗ Missing test protein: {test_protein}")
    
    def has_lrr(self, protein_id: str) -> bool:
        """Check if protein has LRR annotations"""
        result = protein_id in self.protein_lrr_regions
        
        # Detailed debugging
        print(f"[DEBUG] LRRDatabase.has_lrr('{protein_id}'): {result}")
        
        if not result:
            print(f"[DEBUG] Protein '{protein_id}' not found in LRR database")
            print(f"[DEBUG] Total proteins in database: {len(self.protein_lrr_regions)}")
            
            # Check for similar IDs (potential matching issues)
            similar_ids = [pid for pid in self.protein_lrr_regions.keys() if protein_id in pid or pid in protein_id]
            if similar_ids:
                print(f"[DEBUG] Similar IDs found: {similar_ids[:5]}")
            else:
                print(f"[DEBUG] No similar IDs found")
                
            # Print first few IDs for reference
            if self.protein_lrr_regions:
                sample_ids = list(self.protein_lrr_regions.keys())[:5]
                print(f"[DEBUG] Sample IDs in database: {sample_ids}")
        else:
            regions = self.protein_lrr_regions[protein_id]
            print(f"[DEBUG] Protein '{protein_id}' has {len(regions)} LRR regions")
            for i, region in enumerate(regions):
                print(f"[DEBUG]   Region {i+1}: start={region.start}, end={region.end}, length={region.length}")
        
        return result
    
    def get_lrr_regions(self, protein_id: str) -> List[LRRRegion]:
        """Get LRR regions for a protein"""
        return self.protein_lrr_regions.get(protein_id, [])
    
    def get_full_length(self, protein_id: str) -> Optional[int]:
        """Get full sequence length for a protein"""
        return self.protein_full_lengths.get(protein_id)
    
    def get_lrr_residue_indices(self, protein_id: str, zero_based: bool = True) -> List[int]:
        """
        Get all residue indices that are part of LRR regions
        
        Args:
            protein_id: Protein identifier
            zero_based: If True, return 0-based indices; if False, return 1-based
        
        Returns:
            List of residue indices in LRR regions
        """
        regions = self.get_lrr_regions(protein_id)
        indices = []
        
        for region in regions:
            if zero_based:
                # Convert to 0-based
                indices.extend(range(region.start - 1, region.end))
            else:
                # Keep 1-based
                indices.extend(range(region.start, region.end + 1))
        
        return sorted(set(indices))
    
    def get_lrr_edges(self, protein_id: str, connect_all: bool = True, zero_based: bool = True) -> List[Tuple[int, int]]:
        """
        Generate edges connecting residues within LRR regions
        
        Args:
            protein_id: Protein identifier
            connect_all: If True, fully connect all residues in each region;
                        if False, only connect adjacent residues
            zero_based: If True, use 0-based indices; if False, use 1-based
        
        Returns:
            List of (source, target) edge tuples
        """
        regions = self.get_lrr_regions(protein_id)
        edges = []
        
        for region in regions:
            if zero_based:
                start = region.start - 1
                end = region.end - 1
            else:
                start = region.start
                end = region.end
            
            if connect_all:
                # Fully connect all residues in the region
                for i in range(start, end + 1):
                    for j in range(i + 1, end + 1):
                        edges.append((i, j))
                        edges.append((j, i))  # Bidirectional
            else:
                # Only connect adjacent residues
                for i in range(start, end):
                    edges.append((i, i + 1))
                    edges.append((i + 1, i))  # Bidirectional
        
        return edges
    
    def get_statistics(self) -> Dict:
        """Get statistics about LRR annotations"""
        total_proteins = len(self.protein_lrr_regions)
        total_regions = sum(len(regions) for regions in self.protein_lrr_regions.values())
        
        proteins_with_multiple_regions = sum(
            1 for regions in self.protein_lrr_regions.values() if len(regions) > 1
        )
        
        region_lengths = [
            region.length 
            for regions in self.protein_lrr_regions.values() 
            for region in regions
        ]
        
        return {
            'total_proteins': total_proteins,
            'total_regions': total_regions,
            'proteins_with_multiple_regions': proteins_with_multiple_regions,
            'avg_regions_per_protein': total_regions / total_proteins if total_proteins > 0 else 0,
            'avg_region_length': sum(region_lengths) / len(region_lengths) if region_lengths else 0,
            'min_region_length': min(region_lengths) if region_lengths else 0,
            'max_region_length': max(region_lengths) if region_lengths else 0
        }

