#!/bin/bash
# Example usage scripts for Sparse-SP-PPI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============================================================================
# Example 1: Train with ESMC-600m encoding
# ============================================================================
echo "Example 1: Training with ESMC-600m encoding..."

python train.py \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/pdbs \
    --cache_dir $PROJECT_ROOT/data/cache \
    --config $PROJECT_ROOT/configs/precomputed_esmc_600m_lrr_shs27k.json \
    --encoding_type precomputed \
    --embedding_dir $PROJECT_ROOT/embedding/esmc-600m-2024-12/SHS27k \
    --experiment_name esmc_600m_experiment \
    --max_epochs 100 \
    --seed 42

# ============================================================================
# Example 2: Inference - Predict all interactions
# ============================================================================
echo "Example 2: Inference - Predict all interactions..."

python inference.py \
    --checkpoint $PROJECT_ROOT/logs/esmc_600m_experiment_*/checkpoints/best_model.pth \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/pdbs \
    --cache_dir $PROJECT_ROOT/data/cache \
    --mode all \
    --output $PROJECT_ROOT/results/all_predictions.csv \
    --threshold 0.5
