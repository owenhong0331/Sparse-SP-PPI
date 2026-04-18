#!/bin/bash
# Example usage scripts for Customer PPI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============================================================================
# Example 1: Train with MAPE encoding (default)
# ============================================================================
echo "Example 1: Training with MAPE encoding..."

python customer_train.py \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/raw_data/STRING_AF2DB \
    --cache_dir $PROJECT_ROOT/data/cache \
    --config $PROJECT_ROOT/configs/default_config.json \
    --experiment_name mape_experiment \
    --max_epochs 100 \
    --batch_size 5000 \
    --seed 42

# ============================================================================
# Example 2: Train with ESM2 encoding
# ============================================================================
echo "Example 2: Training with ESM2 encoding..."

python customer_train.py \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/raw_data/STRING_AF2DB \
    --cache_dir $PROJECT_ROOT/data/cache \
    --config $PROJECT_ROOT/configs/esm2_config.json \
    --encoding_type esm2 \
    --experiment_name esm2_experiment \
    --max_epochs 100 \
    --seed 42

# ============================================================================
# Example 3: Train with precomputed embeddings (AlphaFold)
# ============================================================================
echo "Example 3: Training with precomputed embeddings..."

python customer_train.py \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/raw_data/STRING_AF2DB \
    --cache_dir $PROJECT_ROOT/data/cache \
    --config $PROJECT_ROOT/configs/precomputed_config.json \
    --encoding_type precomputed \
    --embedding_dir $PROJECT_ROOT/data/embeddings/alphafold \
    --experiment_name alphafold_experiment \
    --max_epochs 100 \
    --seed 42

# ============================================================================
# Example 4: Train with attention mechanism
# ============================================================================
echo "Example 4: Training with attention mechanism..."

python customer_train.py \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/raw_data/STRING_AF2DB \
    --cache_dir $PROJECT_ROOT/data/cache \
    --use_attention \
    --num_heads 8 \
    --experiment_name attention_experiment \
    --max_epochs 100 \
    --seed 42

# ============================================================================
# Example 5: Inference - Predict all interactions
# ============================================================================
echo "Example 5: Inference - Predict all interactions..."

python customer_inference.py \
    --checkpoint $PROJECT_ROOT/logs/mape_experiment_*/checkpoints/best_model.pth \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/raw_data/STRING_AF2DB \
    --cache_dir $PROJECT_ROOT/data/cache \
    --mode all \
    --output $PROJECT_ROOT/results/all_predictions.csv \
    --threshold 0.5

# ============================================================================
# Example 6: Inference - Predict specific pairs
# ============================================================================
echo "Example 6: Inference - Predict specific pairs..."

# Create pairs file
cat > pairs.txt << EOF
0 1
0 5
2 3
10 20
EOF

python customer_inference.py \
    --checkpoint $PROJECT_ROOT/logs/mape_experiment_*/checkpoints/best_model.pth \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/raw_data/STRING_AF2DB \
    --cache_dir $PROJECT_ROOT/data/cache \
    --mode pairs \
    --pairs_file pairs.txt \
    --output $PROJECT_ROOT/results/pair_predictions.csv

# ============================================================================
# Example 7: Inference - Predict single pair
# ============================================================================
echo "Example 7: Inference - Predict single pair..."

python customer_inference.py \
    --checkpoint $PROJECT_ROOT/logs/mape_experiment_*/checkpoints/best_model.pth \
    --ppi_file $PROJECT_ROOT/data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file $PROJECT_ROOT/data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir $PROJECT_ROOT/data/raw_data/STRING_AF2DB \
    --cache_dir $PROJECT_ROOT/data/cache \
    --mode single \
    --protein1 0 \
    --protein2 5 \
    --output $PROJECT_ROOT/results/single_prediction.json

# ============================================================================
# Example 8: View training logs with TensorBoard
# ============================================================================
echo "Example 8: Launch TensorBoard..."

tensorboard --logdir=$PROJECT_ROOT/logs --port=6006

echo "Open http://localhost:6006 in your browser to view training logs"
