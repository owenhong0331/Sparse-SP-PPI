# SPIN-PPI

**SPIN-PPI** (**S**tructural **P**rior **In**tegration Network for protein–protein interaction prediction) is a memory-efficient framework for structure-guided protein–protein interaction (PPI) prediction through adaptive structural prior integration.

SPIN-PPI integrates complementary residue-level structural priors—including sequence adjacency, local structural neighborhoods, distance-based contacts, exposed-surface accessibility, and LRR-region regularity—and adaptively calibrates their contributions before compressing residue features into protein-level representations for downstream GraphConv-based multi-type PPI prediction.

This repository provides source code, training and inference scripts, configuration files, preprocessing utilities, and instructions for reproducing the experiments reported in the manuscript:

> **SPIN-PPI: Adaptive structural prior integration for memory-efficient protein–protein interaction prediction**

---

## Highlights

- **Adaptive structural prior integration**  
  SPIN-PPI learns relation-specific weights over multiple structural-prior channels instead of treating all structural cues as equally informative.

- **Reduced dependence on structure-only contacts**  
  The model combines sequence continuity, local geometry, distance-based contacts, exposed-surface accessibility, and LRR-region regularity to calibrate potentially noisy structure-derived contacts.

- **Residue-to-protein compression under memory constraints**  
  Residue-level prior information is integrated before being compressed into protein-level representations, enabling efficient PPI graph reasoning under constrained GPU memory.

- **Benchmark coverage**  
  The repository supports experiments on SHS27k, SHS148k, STRING, SYS30k, and SYS60k with Random, DFS, and BFS splits.

---

## Overview

SPIN-PPI uses a two-level architecture.

### 1. Residue-level structural-prior encoding

Each protein is represented as a heterogeneous residue graph. Residues are nodes, and edges correspond to complementary relation channels:

| Prior channel | Description |
|--------------|-------------|
| `SEQ` | Sequence adjacency between neighboring residues |
| `STR-KNN` | Local structural neighborhoods based on Cα K-nearest neighbors |
| `STR-DIS` | Distance-thresholded structural contacts |
| `SURF` | Exposed-surface proximity based on solvent-accessible surface area |
| `LRR-REGION` | LRR-region connectivity derived from structure-aware LRR annotation |

A lightweight relation scorer learns adaptive weights over these priors, allowing the model to calibrate structural evidence according to the input protein and dataset context.

### 2. Protein-level PPI graph reasoning

After adaptive residue-level prior integration, calibrated residue features are pooled into compact protein-level representations. These representations are then used as node features in a GraphConv-based PPI graph encoder for multi-type interaction prediction.

```text
Protein sequence + structure
        ↓
Residue embeddings + structural-prior graph construction
        ↓
Adaptive relation weighting over SEQ / STR-KNN / STR-DIS / SURF / LRR-REGION
        ↓
Residue-to-protein compression
        ↓
Protein-level GraphConv reasoning
        ↓
Multi-type PPI prediction
```

---

## Repository Structure

```text
SPIN-PPI/
├── models/                                  # Core model package
│   ├── sparse_sp_ppi.py                     # Main model implementation
│   ├── dataloader.py                        # Data loading and graph construction
│   ├── edge_construction.py                 # Edge builders: SEQ, STR-KNN, STR-DIS, SURF
│   ├── node_encoding.py                     # Node feature encoders
│   ├── protein_graph_builder.py             # Protein graph construction utilities
│   ├── lrr_parser.py                        # LRR annotation parser
│   ├── lrr_extractor.py                     # LRR node embedding extraction
│   ├── metrics.py                           # Evaluation metrics
│   ├── logger.py                            # TensorBoard and text logging
│   └── checkpoint.py                        # Checkpoint management
├── scripts/                                 # Training and inference scripts
│   ├── train.py                             # Main training script
│   ├── inference.py                         # Inference script
│   ├── train_sparse_sp_ppi_experiments.sh   # Batch training experiments
│   ├── infer_sparse_sp_ppi_experiments.sh   # Batch inference experiments
│   ├── run_cross_dataset_inference.sh       # Cross-dataset inference
│   ├── batch_csv_to_fasta.sh                # CSV-to-FASTA conversion
│   ├── batch_generate_esm_embeddings.sh     # ESMC embedding generation
│   └── example_usage.sh                     # Usage examples
├── configs/                                 # JSON configuration files
│   └── precomputed_esmc_600m_lrr_*.json
├── lrr_annotation/                          # LRR annotation tools
│   ├── geom_lrr/                            # Geometry-based LRR analysis module
│   │   ├── loader.py
│   │   ├── analyzer.py
│   │   └── plotter.py
│   ├── extract_lrr_sequences.py
│   ├── generate_lrr_annotations.py
│   └── parse_lrr_annotation.py
├── lrr/                                     # LRR annotation files
│   └── lrr_annotation_results.txt
├── process_data.sh                          # Data processing pipeline
├── requirements.txt                         # Python dependencies
└── README.md
```

> Note: Some script or file names may still contain the legacy name `sparse_sp_ppi` for backward compatibility. These files implement the SPIN-PPI framework.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/owenhong0331/SPIN-PPI.git
cd SPIN-PPI
```

### 2. Create an environment

Using Conda is recommended.

```bash
conda create -n spin_ppi python=3.10 -y
conda activate spin_ppi
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you need to generate ESMC embeddings from scratch, install ESMC support:

```bash
pip install esm-c
```

### 4. Verify installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import dgl; print(dgl.__version__)"
```

---

## Data Availability

The datasets and processed resources are available from Hugging Face:

- Dataset repository: `https://huggingface.co/datasets/owenhong0331/SPIN-PPI`

The dataset repository contains raw interaction files, sequence dictionaries, PDB structures, processed graph data, and precomputed ESMC-600M embeddings.

### Available archives

| Archive | Description | Approximate size |
|--------|-------------|------------------|
| `SPIN-PPI_raw_data.zip` | PPI interaction files and sequence dictionaries | ~31 MB |
| `SPIN-PPI_pdb_structures.zip` | Protein structure files used for graph construction | ~1.7 GB |
| `SPIN-PPI_processed_data.zip` | Preprocessed graph data | ~1.3 GB |
| `SPIN-PPI_esmc_embeddings.zip` | ESMC-600M residue embeddings | ~60 GB |

> If the archived files still use the legacy prefix `Sparse-SP-PPI`, they correspond to the same SPIN-PPI dataset release. Please keep file paths consistent with your local setup.

---

## Data Download

### Option A: download with Hugging Face CLI

```bash
pip install huggingface_hub

huggingface-cli download owenhong0331/SPIN-PPI \
  --repo-type dataset \
  --local-dir ./data_download
```

### Option B: clone with Git LFS

```bash
git lfs install
git clone https://huggingface.co/datasets/owenhong0331/SPIN-PPI ./data_download
```

### Option C: download a specific archive

```bash
huggingface-cli download owenhong0331/SPIN-PPI \
  SPIN-PPI_raw_data.zip \
  --repo-type dataset \
  --local-dir ./data_download
```

---

## Expected Directory Layout

After downloading and extracting the data, the project directory should look like this:

```text
SPIN-PPI/
├── data/
│   ├── raw_data/
│   │   ├── protein.actions.SHS27k.txt
│   │   ├── protein.SHS27k.sequences.dictionary.csv
│   │   └── ...
│   ├── all_pdbs/
│   │   └── *.pdb
│   ├── processed_data_SHS27k/
│   ├── processed_data_SHS148k/
│   ├── processed_data_STRING/
│   ├── processed_data_SYS30k/
│   └── processed_data_SYS60k/
├── embedding/
│   └── esmc-600m-2024-12/
│       ├── SHS27k/
│       ├── SHS148k/
│       ├── STRING/
│       ├── SYS30k/
│       └── SYS60k/
└── lrr/
    └── lrr_annotation_results.txt
```

Example extraction commands:

```bash
unzip SPIN-PPI_raw_data.zip -d data/raw_data/
unzip SPIN-PPI_pdb_structures.zip -d data/all_pdbs/
unzip SPIN-PPI_processed_data.zip -d data/
unzip SPIN-PPI_esmc_embeddings.zip -d embedding/esmc-600m-2024-12/
```

If your downloaded archives use the legacy prefix, use the corresponding file names:

```bash
unzip Sparse-SP-PPI_raw_data.zip -d data/raw_data/
unzip Sparse-SP-PPI_pdb_structures.zip -d data/all_pdbs/
unzip Sparse-SP-PPI_processed_data.zip -d data/
unzip Sparse-SP-PPI_esmc_embeddings.zip -d embedding/esmc-600m-2024-12/
```

---

## Data Format

### PPI interaction file

Tab-separated file:

```text
protein1    protein2    interaction_type
```

### Sequence dictionary

CSV file containing protein identifiers and amino-acid sequences.

### Structure files

PDB files are used to construct residue-level structural-prior graphs.

### Embeddings

Precomputed ESMC-600M residue embeddings are stored as `.npy` files.

---

## Preprocessing

### Full preprocessing pipeline

```bash
bash process_data.sh \
  --pdb_dir /path/to/pdb_files \
  --seq_file /path/to/sequences.csv \
  --dataset SHS27k \
  --embedding_model esmc_600m
```

### Skip selected steps

```bash
bash process_data.sh \
  --pdb_dir /path/to/pdb_files \
  --seq_file /path/to/sequences.csv \
  --dataset SHS27k \
  --skip_fasta \
  --skip_embedding
```

### Generate LRR annotations

```bash
cd lrr_annotation

python generate_lrr_annotations.py \
  /path/to/pdb_dir \
  -o lrr_annotation_results.txt
```

The LRR annotation pipeline computes geometry-derived winding patterns along the protein backbone and detects LRR-consistent regions through piecewise linear regression.

---

## Training

### Single training run

```bash
python scripts/train.py \
  --ppi_file data/raw_data/protein.actions.SHS27k.txt \
  --protein_seq_file data/raw_data/protein.SHS27k.sequences.dictionary.csv \
  --pdb_dir data/all_pdbs \
  --config configs/precomputed_esmc_600m_lrr_shs27k.json \
  --encoding_type precomputed \
  --experiment_name spin_ppi_shs27k_random
```

### Batch training

```bash
bash scripts/train_sparse_sp_ppi_experiments.sh \
  --dataset SHS27k \
  --encoder esmc_600m \
  --encoder-type lrr \
  --split random
```

Supported datasets include:

```text
SHS27k, SHS148k, STRING, SYS30k, SYS60k
```

Supported split strategies include:

```text
random, bfs, dfs
```

---

## Inference

### Predict all interactions in a file

```bash
python scripts/inference.py \
  --checkpoint logs/spin_ppi_shs27k_random/checkpoints/best_model.pth \
  --ppi_file data/raw_data/protein.actions.SHS27k.txt \
  --protein_seq_file data/raw_data/protein.SHS27k.sequences.dictionary.csv \
  --pdb_dir data/all_pdbs \
  --mode all \
  --output predictions.csv
```

### Predict a specific protein pair

```bash
python scripts/inference.py \
  --checkpoint logs/spin_ppi_shs27k_random/checkpoints/best_model.pth \
  --ppi_file data/raw_data/protein.actions.SHS27k.txt \
  --protein_seq_file data/raw_data/protein.SHS27k.sequences.dictionary.csv \
  --pdb_dir data/all_pdbs \
  --mode single \
  --protein1 0 \
  --protein2 5 \
  --output prediction.json
```

### Cross-dataset inference

```bash
bash scripts/run_cross_dataset_inference.sh
```

---

## Configuration

Configuration files are stored in `configs/` and use JSON format.

| Section | Description |
|--------|-------------|
| `model` | Architecture parameters, hidden dimensions, dropout, relation weighting |
| `encoding` | Node encoding settings and embedding directories |
| `edge_construction` | Structural-prior graph parameters, including KNN and distance thresholds |
| `training` | Epochs, batch size, learning rate, and optimization settings |
| `data_split` | Split strategy and train/validation/test ratios |
| `logging` | Output directories, checkpoint settings, and selected metrics |

---

## Evaluation Metrics

The training and inference scripts report:

- Accuracy
- Precision
- Recall
- F1 score: micro, macro, and weighted
- AUC-ROC
- AUPR
- Per-class metrics
- Confusion matrix

The primary metric reported in the manuscript is `F1_micro`.

---

## Reproducing Manuscript Results

A typical workflow for reproducing the main benchmark results is:

```bash
# 1. Download data
huggingface-cli download owenhong0331/SPIN-PPI \
  --repo-type dataset \
  --local-dir ./data_download

# 2. Extract data
unzip data_download/SPIN-PPI_raw_data.zip -d data/raw_data/
unzip data_download/SPIN-PPI_pdb_structures.zip -d data/all_pdbs/
unzip data_download/SPIN-PPI_processed_data.zip -d data/
unzip data_download/SPIN-PPI_esmc_embeddings.zip -d embedding/esmc-600m-2024-12/

# 3. Run benchmark training
bash scripts/train_sparse_sp_ppi_experiments.sh \
  --dataset SHS27k \
  --encoder esmc_600m \
  --encoder-type lrr \
  --split random

# 4. Run inference
bash scripts/infer_sparse_sp_ppi_experiments.sh \
  --dataset SHS27k \
  --encoder esmc_600m \
  --encoder-type lrr \
  --split random
```

For full reproduction, repeat the workflow for all datasets and split strategies:

```text
Datasets: SHS27k, SHS148k, STRING, SYS30k, SYS60k
Splits: random, bfs, dfs
```

---

## Hardware Notes

The constrained setting reported in the manuscript was designed for a single 22 GB GPU, with peak training GPU memory around 20 GB and host memory around 30 GB.

If you encounter out-of-memory errors, consider reducing:

- `batch_size`
- number of data-loader workers
- classifier batch size
- VAE or embedding batch size, if applicable

---

## Citation

If you use this code or dataset, please cite the SPIN-PPI manuscript:

```bibtex
@article{spin_ppi_2026,
  title   = {SPIN-PPI: Adaptive structural prior integration for memory-efficient protein-protein interaction prediction},
  author  = {Chen, Haowen and Hong, Weihao and Yang, Xinyu and Fu, Xiangzheng},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {Manuscript submitted}
}
```

Please also cite the dataset repository if you use the processed data:

```bibtex
@misc{spin_ppi_dataset_2026,
  title        = {SPIN-PPI Dataset},
  author       = {Chen, Haowen and Hong, Weihao and Yang, Xinyu and Fu, Xiangzheng},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/owenhong0331/SPIN-PPI}}
}
```

If a Zenodo DOI is available, please cite the archived software/data release as well.

---

## License

Please specify the license for this repository before public release, for example:

- MIT License
- Apache License 2.0
- GPL-3.0 License

---

## Contact

For questions about the code, data, or manuscript, please contact:

- Xinyu Yang: `yangxinyu621@foxmail.com`
- Xiangzheng Fu: `fuxiangzheng@suat-sz.edu.cn`
