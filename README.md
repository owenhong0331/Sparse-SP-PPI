# Sparse-SP-PPI

Sparse-SP-PPI: Sparse Structure-aware Protein-Protein Interaction Prediction via Hierarchical Graph Neural Network with Learnable LRR Attention.

## Overview

Sparse-SP-PPI is a hierarchical heterogeneous graph neural network for protein-protein interaction (PPI) prediction. The system implements a two-level architecture:

- **Protein-level (Bottom View)**: Each protein is modeled as a heterogeneous graph with amino acids as nodes. Multiple edge types capture structural and sequence relationships, including sequence adjacency (SEQ), spatial distance (STR_DIS), spatial K-nearest neighbors (STR_KNN), surface accessibility (SURF), and LRR region connectivity (LRR_REGION).
- **PPI-level (Top View)**: A protein-protein interaction network where proteins are nodes and interactions are edges. Graph convolution layers propagate information across the interaction network.

The key innovation is the **SparseEdgeAttentionEncoder**, which learns attention-weighted aggregation across different edge types, with special emphasis on Leucine-Rich Repeat (LRR) domain regions that are critical for plant immune receptor interactions.

## Architecture

### Model: ProteinGINModelSimple

```
Protein Graph → SparseEdgeAttentionEncoder (attention-weighted edge aggregation)
             → Mean Pooling → Protein Embedding
                                              ↓
PPI Network  → GraphConv Layers → Interaction Prediction Head
```

**SparseEdgeAttentionEncoder**:
- Processes 5 edge types: SEQ, STR_KNN, STR_DIS, SURF, LRR_REGION
- Learns relation-specific attention scores via MLP
- Applies temperature-scaled softmax with attention bias
- Supports periodic gradient updates for LRR attention weights

### Encoding

| Encoding | Description | Dimension |
|----------|-------------|-----------|
| ESMC-600M | ESMC-600M pre-trained embeddings (recommended) | 1152 |
| Precomputed | Custom .npy embedding files | Variable |

Other encodings (MAPE, ESM2, ESM3, OneHot) are deprecated but still available in code.

## Project Structure

```
Sparse-SP-PPI/
├── models/                           # Core model package
│   ├── sparse_sp_ppi.py              # Main model (ProteinGINModelSimple, SparseEdgeAttentionEncoder)
│   ├── dataloader.py                 # Data loading and graph construction
│   ├── edge_construction.py          # Edge type builders (SEQ, STR_KNN, STR_DIS, SURF)
│   ├── node_encoding.py              # Node feature encoders (PrecomputedEncoder primary)
│   ├── protein_graph_builder.py      # Graph construction strategies
│   ├── lrr_parser.py                 # LRR annotation parser
│   ├── lrr_extractor.py              # LRR node embedding extraction
│   ├── metrics.py                    # Evaluation metrics (Acc, Prec, Rec, F1, AUC-ROC, AUPR)
│   ├── logger.py                     # TensorBoard + text logging
│   └── checkpoint.py                 # Model checkpoint management
├── scripts/                          # Training and inference scripts
│   ├── train.py                      # Main training script
│   ├── inference.py                  # Inference script
│   ├── train_sparse_sp_ppi_experiments.sh  # Batch training experiments
│   ├── infer_sparse_sp_ppi_experiments.sh  # Batch inference experiments
│   ├── run_cross_dataset_inference.sh      # Cross-dataset inference
│   ├── batch_csv_to_fasta.sh               # CSV to FASTA conversion
│   ├── batch_generate_esm_embeddings.sh    # ESM embedding generation
│   └── example_usage.sh                    # Usage examples
├── configs/                          # JSON configuration files
│   └── precomputed_esmc_600m_lrr_*.json   # ESMC-600M LRR configs per dataset
├── lrr_annotation/                   # LRR annotation generation tools
│   ├── geom_lrr/                     # Core geometry analysis module
│   │   ├── loader.py                 # PDB loading and CA atom extraction
│   │   ├── analyzer.py               # Winding number computation and LRR detection
│   │   └── plotter.py                # Regression visualization
│   ├── extract_lrr_sequences.py      # LRR sequence extraction
│   ├── generate_lrr_annotations.py   # Main annotation generation script
│   └── parse_lrr_annotation.py       # Convert annotations to FASTA
├── lrr/                              # LRR annotation data
│   └── lrr_annotation_results.txt    # Pre-generated LRR annotations
├── process_data.sh                   # Consolidated data processing pipeline
└── requirements.txt                  # Python dependencies
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# ESMC support (for embedding generation)
pip install esm-c
```

## Data Preparation

### Data Requirements

```
data/
├── processed_data/
│   ├── protein.actions.<DATASET>.txt       # PPI interactions
│   └── protein.<DATASET>.sequences.dictionary.csv  # Protein sequences
├── pdbs/                                    # PDB structure files (AlphaFold2)
└── embedding/                               # Precomputed embeddings
    └── esmc-600m-2024-12/<DATASET>/        # Per-dataset embedding files
```

### PPI Interaction File Format

Tab-separated file with columns: `protein1 \t protein2 \t interaction_type`

### Running the Data Processing Pipeline

```bash
# Full pipeline: LRR annotation + FASTA conversion + embedding generation
bash process_data.sh \
    --pdb_dir /path/to/pdb_files \
    --seq_file /path/to/sequences.csv \
    --dataset SHS27k \
    --embedding_model esmc_600m

# Skip specific steps
bash process_data.sh --pdb_dir /path/to/pdbs --skip_fasta --skip_embedding
```

### Generating LRR Annotations

LRR annotations are generated from PDB structure files using the geometric analysis pipeline in `lrr_annotation/`:

```bash
cd lrr_annotation
python generate_lrr_annotations.py /path/to/pdb_dir -o lrr_annotation_results.txt
```

This computes winding numbers via parallel transport along the protein backbone, then detects LRR regions through piecewise linear regression.

## Training

```bash
# Training with ESMC-600M encoding
python scripts/train.py \
    --ppi_file data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir data/pdbs \
    --config configs/precomputed_esmc_600m_lrr_shs27k.json \
    --encoding_type precomputed \
    --experiment_name my_experiment

# Batch training
bash scripts/train_sparse_sp_ppi_experiments.sh \
    --dataset SHS27k \
    --encoder esmc_600m \
    --encoder-type lrr \
    --split random
```

## Inference

```bash
# Predict all interactions
python scripts/inference.py \
    --checkpoint logs/my_experiment/checkpoints/best_model.pth \
    --ppi_file data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir data/pdbs \
    --mode all \
    --output predictions.csv

# Predict specific protein pair
python scripts/inference.py \
    --checkpoint logs/my_experiment/checkpoints/best_model.pth \
    --ppi_file data/processed_data/protein.actions.SHS27k.txt \
    --protein_seq_file data/processed_data/protein.SHS27k.sequences.dictionary.csv \
    --pdb_dir data/pdbs \
    --mode single \
    --protein1 0 --protein2 5 \
    --output prediction.json

# Cross-dataset inference
bash scripts/run_cross_dataset_inference.sh
```

## Configuration

Configuration files use JSON with the following sections:

| Section | Description |
|---------|-------------|
| `model` | Architecture parameters (dimensions, layers, dropout, attention) |
| `encoding` | Node encoding settings (type, embedding_dir, LRR config) |
| `edge_construction` | Graph edge parameters (thresholds, KNN, LRR edges) |
| `training` | Hyperparameters (epochs, batch_size, learning_rate, LRR-specific LR) |
| `data_split` | Split strategy (random/bfs/dfs, ratios) |
| `logging` | Experiment logging (directory, checkpoint frequency, selection metric) |

## Evaluation Metrics

The system computes comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1 (micro/macro/weighted)
- AUC-ROC, AUPR (micro/macro)
- Per-class metrics
- Confusion matrix

## Data Availability

The datasets used in this project are available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

### Downloaded Archives

| Archive | Description | Size |
|---------|-------------|------|
| `Sparse-SP-PPI_esmc_embeddings.zip` | ESMC-600M protein embeddings (.npy) for SHS27k, SHS148k, STRING, SYS30k, SYS60k | ~66.5 GB |
| `Sparse-SP-PPI_pdb_structures.zip` | AlphaFold2 PDB structures (Human + Yeast) | ~175 MB |
| `Sparse-SP-PPI_processed_data.zip` | Preprocessed graph data for SHS27k, SHS148k, STRING | ~5.3 GB |
| `Sparse-SP-PPI_raw_data.zip` | PPI interaction files and sequence dictionaries | ~313 MB |

### Setup Instructions

1. Download all archives from Zenodo
2. Extract to your project directory:

```bash
# ESMC embeddings → embedding directory
unzip Sparse-SP-PPI_esmc_embeddings.zip -d embedding/esmc-600m-2024-12/

# PDB structures → data directory
unzip Sparse-SP-PPI_pdb_structures.zip -d data/all_pdbs/

# Processed data → data directory
unzip Sparse-SP-PPI_processed_data.zip -d data/

# Raw data → data/raw_data/
unzip Sparse-SP-PPI_raw_data.zip -d data/raw_data/
```

3. Expected directory structure after extraction:

```
Sparse-SP-PPI/
├── embedding/
│   └── esmc-600m-2024-12/
│       ├── SHS27k/     (*.npy)
│       ├── SHS148k/    (*.npy)
│       ├── STRING/     (*.npy)
│       ├── SYS30k/     (*.npy)
│       └── SYS60k/     (*.npy)
├── data/
│   ├── all_pdbs/       (*.pdb, Human + Yeast)
│   ├── processed_data_SHS27k/
│   ├── processed_data_SHS148k/
│   ├── processed_data_STRING/
│   └── raw_data/       (*.txt, *.tsv)
```

## Citation

```bibtex
@article{sparse_sp_ppi,
  title={Sparse-SP-PPI: Sparse Structure-aware Protein-Protein Interaction Prediction via Hierarchical Graph Neural Network with Learnable LRR Attention},
  author={},
  journal={},
  year={2026}
}
```
