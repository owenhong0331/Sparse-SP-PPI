#!/bin/bash
# ============================================================================
# Sparse-SP-PPI Data Processing Pipeline
# ============================================================================
# This script handles the complete data processing pipeline:
#   1. Generate LRR annotations from PDB files
#   2. Convert protein sequences CSV to FASTA format
#   3. Generate ESM embeddings for protein sequences
#   4. Validate data consistency
#
# Usage:
#   bash process_data.sh [OPTIONS]
#
# Options:
#   --pdb_dir DIR         Directory containing PDB files (required for LRR annotation)
#   --seq_file FILE       Protein sequence CSV file (required for FASTA conversion)
#   --dataset NAME        Dataset name (e.g., SHS27k, SHS148k, Arabidopsis)
#   --embedding_model MODEL  ESM model to use (esmc_600m, esmc_300m, esm3_small, esm2_650m, esm2_3b)
#   --output_dir DIR      Output directory (default: ./data)
#   --skip_lrr            Skip LRR annotation generation
#   --skip_fasta          Skip FASTA conversion
#   --skip_embedding      Skip embedding generation
#   --help                Show this help message
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
PDB_DIR=""
SEQ_FILE=""
DATASET=""
EMBEDDING_MODEL="esmc_600m"
OUTPUT_DIR="$PROJECT_ROOT/data"
SKIP_LRR=false
SKIP_FASTA=false
SKIP_EMBEDDING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pdb_dir)
            PDB_DIR="$2"
            shift 2
            ;;
        --seq_file)
            SEQ_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --embedding_model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_lrr)
            SKIP_LRR=true
            shift
            ;;
        --skip_fasta)
            SKIP_FASTA=true
            shift
            ;;
        --skip_embedding)
            SKIP_EMBEDDING=true
            shift
            ;;
        --help)
            echo "Usage: bash process_data.sh [OPTIONS]"
            echo ""
            echo "Sparse-SP-PPI Data Processing Pipeline"
            echo ""
            echo "Options:"
            echo "  --pdb_dir DIR            Directory containing PDB files"
            echo "  --seq_file FILE          Protein sequence CSV file"
            echo "  --dataset NAME           Dataset name (SHS27k, SHS148k, etc.)"
            echo "  --embedding_model MODEL  ESM model (esmc_600m, esmc_300m, esm3_small, esm2_650m, esm2_3b)"
            echo "  --output_dir DIR         Output directory (default: ./data)"
            echo "  --skip_lrr               Skip LRR annotation generation"
            echo "  --skip_fasta             Skip FASTA conversion"
            echo "  --skip_embedding         Skip embedding generation"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  Sparse-SP-PPI Data Processing Pipeline"
echo "============================================"
echo "Project Root: $PROJECT_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/lrr"
mkdir -p "$OUTPUT_DIR/embedding"
mkdir -p "$OUTPUT_DIR/fasta"

# ============================================================================
# Step 1: Generate LRR Annotations from PDB files
# ============================================================================
if [ "$SKIP_LRR" = false ]; then
    echo ""
    echo "=== Step 1: Generate LRR Annotations ==="

    if [ -z "$PDB_DIR" ]; then
        echo "Warning: --pdb_dir not specified, skipping LRR annotation generation"
        echo "  To generate LRR annotations, provide PDB directory:"
        echo "  bash process_data.sh --pdb_dir /path/to/pdbs"
    else
        if [ ! -d "$PDB_DIR" ]; then
            echo "Error: PDB directory not found: $PDB_DIR"
            exit 1
        fi

        LRR_OUTPUT="$PROJECT_ROOT/lrr_annotation/lrr_annotation_results.txt"

        echo "PDB Directory: $PDB_DIR"
        echo "LRR Output: $LRR_OUTPUT"

        cd "$PROJECT_ROOT/lrr_annotation"
        python generate_lrr_annotations.py "$PDB_DIR" -o "$LRR_OUTPUT"
        cd "$PROJECT_ROOT"

        # Copy to lrr/ directory for config compatibility
        if [ -f "$LRR_OUTPUT" ]; then
            cp "$LRR_OUTPUT" "$PROJECT_ROOT/lrr/lrr_annotation_results.txt"
            echo "LRR annotations generated successfully"
        else
            echo "Warning: LRR annotation file not generated"
        fi
    fi
else
    echo "Skipping LRR annotation generation (--skip_lrr)"
fi

# ============================================================================
# Step 2: Convert CSV to FASTA format
# ============================================================================
if [ "$SKIP_FASTA" = false ]; then
    echo ""
    echo "=== Step 2: Convert Sequences to FASTA ==="

    if [ -z "$SEQ_FILE" ]; then
        echo "Warning: --seq_file not specified, skipping FASTA conversion"
        echo "  To convert sequences, provide CSV file:"
        echo "  bash process_data.sh --seq_file /path/to/sequences.csv"
    else
        if [ ! -f "$SEQ_FILE" ]; then
            echo "Error: Sequence file not found: $SEQ_FILE"
            exit 1
        fi

        FASTA_OUTPUT="$OUTPUT_DIR/fasta"

        echo "Sequence File: $SEQ_FILE"
        echo "FASTA Output: $FASTA_OUTPUT"

        # Check if csv_to_fasta_converter.py exists
        if [ -f "$PROJECT_ROOT/scripts/csv_to_fasta_converter.py" ]; then
            cd "$PROJECT_ROOT/scripts"
            python csv_to_fasta_converter.py --input "$SEQ_FILE" --output "$FASTA_OUTPUT"
            cd "$PROJECT_ROOT"
        else
            echo "Note: csv_to_fasta_converter.py not found in scripts/"
            echo "  Please install or provide the converter manually"
            echo "  You can also use batch_csv_to_fasta.sh for batch conversion"
        fi
    fi
else
    echo "Skipping FASTA conversion (--skip_fasta)"
fi

# ============================================================================
# Step 3: Generate ESM Embeddings
# ============================================================================
if [ "$SKIP_EMBEDDING" = false ]; then
    echo ""
    echo "=== Step 3: Generate ESM Embeddings ==="

    FASTA_DIR="$OUTPUT_DIR/fasta"
    EMBEDDING_OUTPUT="$OUTPUT_DIR/embedding/$EMBEDDING_MODEL"

    if [ ! -d "$FASTA_DIR" ] || [ -z "$(ls -A $FASTA_DIR 2>/dev/null)" ]; then
        echo "Warning: No FASTA files found in $FASTA_DIR"
        echo "  Please run FASTA conversion first or provide FASTA files"
    else
        echo "FASTA Directory: $FASTA_DIR"
        echo "Embedding Model: $EMBEDDING_MODEL"
        echo "Embedding Output: $EMBEDDING_OUTPUT"

        # Check if generate_esm_embeddings_v2.py exists
        if [ -f "$PROJECT_ROOT/scripts/generate_esm_embeddings_v2.py" ]; then
            cd "$PROJECT_ROOT/scripts"
            python generate_esm_embeddings_v2.py \
                --model "$EMBEDDING_MODEL" \
                --input_dir "$FASTA_DIR" \
                --output_dir "$EMBEDDING_OUTPUT"
            cd "$PROJECT_ROOT"
        else
            echo "Note: generate_esm_embeddings_v2.py not found in scripts/"
            echo "  Please install ESM embedding tools manually"
            echo "  You can also use batch_generate_esm_embeddings.sh for batch processing"
            echo ""
            echo "  To install ESM models:"
            echo "    pip install fair-esm        # For ESM2"
            echo "    pip install esm             # For ESM3"
            echo "    pip install esm-c           # For ESMC"
        fi
    fi
else
    echo "Skipping embedding generation (--skip_embedding)"
fi

# ============================================================================
# Step 4: Validate data consistency
# ============================================================================
echo ""
echo "=== Step 4: Validate Data Consistency ==="

# Check LRR annotation file
LRR_FILE="$PROJECT_ROOT/lrr/lrr_annotation_results.txt"
if [ -f "$LRR_FILE" ]; then
    LRR_COUNT=$(tail -n +2 "$LRR_FILE" | wc -l | tr -d ' ')
    echo "LRR annotations: $LRR_COUNT regions found"
else
    echo "Warning: No LRR annotation file found at $LRR_FILE"
fi

# Check embedding directory
if [ -d "$OUTPUT_DIR/embedding" ]; then
    EMBED_COUNT=$(find "$OUTPUT_DIR/embedding" -name "*.npy" | wc -l | tr -d ' ')
    echo "Embeddings: $EMBED_COUNT .npy files found"
else
    echo "Warning: No embedding directory found"
fi

echo ""
echo "============================================"
echo "  Data Processing Complete!"
echo "============================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify the generated data files"
echo "  2. Update config files in configs/ if needed"
echo "  3. Run training: bash scripts/train_sparse_sp_ppi_experiments.sh"
