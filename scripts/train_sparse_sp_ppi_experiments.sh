#!/bin/bash

# Sparse-SP-PPI Experiments Training Script
# Comprehensive training pipeline for different datasets and split methods

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_ROOT="$PROJECT_ROOT/graphcache"

# Default configurations
PYTHON_SCRIPT="train.py"
DATA_DIR="$PROJECT_ROOT/data"

# Available datasets (all supported datasets)
DATASETS=("SHS27k" "SHS148k" "Arabidopsis" "rice" "STRING")

# Available encoder types
ENCODERS=("esmc_600m")

# Available encoder types (standard, lrr, pep)
ENCODER_TYPES=("standard" "lrr" "pep")

# Available split methods (all supported split methods)
SPLIT_METHODS=("random" "bfs" "dfs")

# Backup and setup data directory
setup_data_directory() {
    local dataset="$1"
    local split_method="$2"

    echo "=== Setting up data directory for $dataset with $split_method split ==="

    # Backup existing processed_data directory
    if [ -d "$DATA_DIR/processed_data" ]; then
        echo "Backing up existing processed_data directory..."
        # rm -rf  "$DATA_DIR/processed_data"
    fi

    # Copy the dataset-specific processed data
    local source_dir="$DATA_DIR/processed_data_${dataset}"
    if [ ! -d "$source_dir" ]; then
        echo "Error: Source directory not found: $source_dir"
        return 1
    fi

    echo "Copying data from $source_dir to processed_data..."
    # cp -r "$source_dir" "$DATA_DIR/processed_data"

    # Update the split method in the protein data file if needed
    update_split_method "$dataset" "$split_method"

    return 0
}

# Update split method in protein data file
update_split_method() {
    local dataset="$1"
    local split_method="$2"

    local protein_file="$DATA_DIR/processed_data/protein.actions.${dataset}.txt"
    if [ ! -f "$protein_file" ]; then
        echo "Warning: Protein actions file not found: $protein_file"
        return 1
    fi

    echo "Updating split method to $split_method in $protein_file..."
    # This assumes the protein file has a column for split method
    # You may need to adjust this based on your actual file format
    sed -i "s/split=[^[:space:]]*/split=$split_method/g" "$protein_file"

    return 0
}

# Generate experiment name
generate_experiment_name() {
    local dataset="$1"
    local encoder="$2"
    local encoder_type="$3"
    local split_method="$4"

    # Use standard as default for backward compatibility
    if [ -z "$encoder_type" ] || [ "$encoder_type" = "standard" ]; then
        echo "${dataset}_${encoder}_${split_method}"
    else
        echo "${dataset}_${encoder}_${encoder_type}_${split_method}"
    fi
}

# Get config file path
get_config_path() {
    local encoder="$1"
    local dataset="$2"
    local encoder_type="$3"

    if [ -z "$encoder_type" ]; then
        encoder_type="lrr"
    fi

    if [ "$encoder_type" = "lrr" ]; then
        local lrr_config="precomputed_esmc_600m_lrr_${dataset,,}.json"
        if [ -f "$PROJECT_ROOT/configs/$lrr_config" ]; then
            echo "$PROJECT_ROOT/configs/$lrr_config"
        else
            echo "$PROJECT_ROOT/configs/precomputed_esmc_600m_lrr_shs27k.json"
        fi
    else
        echo "$PROJECT_ROOT/configs/precomputed_esmc_600m_lrr_${dataset,,}.json"
    fi
}

# Get PPI file path
get_ppi_file_path() {
    local dataset="$1"
    echo "$DATA_DIR/processed_data/protein.actions.${dataset}.txt"
}

# Get protein dictionary file path
get_protein_dict_path() {
    local dataset="$1"
    echo "$DATA_DIR/processed_data/protein.${dataset}.sequences.dictionary.csv"
}

# Get embedding directory for precomputed embeddings
get_embedding_dir() {
    local encoder="$1"
    local dataset="$2"
    echo "$PROJECT_ROOT/embedding/esmc-600m-2024-12/${dataset}"
}

# Run training for a single configuration
train_experiment() {
    local dataset="$1"
    local encoder="$2"
    local encoder_type="$3"
    local split_method="$4"

    echo ""
    echo "=== Starting Experiment: $dataset - $encoder - $encoder_type - $split_method ==="

    # Generate experiment name
    local experiment_name=$(generate_experiment_name "$dataset" "$encoder" "$encoder_type" "$split_method")
    echo "Experiment Name: $experiment_name"

    # Setup data directory
    if ! setup_data_directory "$dataset" "$split_method"; then
        echo "Error: Failed to setup data directory for $dataset"
        return 1
    fi

    # Get file paths
    local ppi_file=$(get_ppi_file_path "$dataset")
    local protein_seq_file=$(get_protein_dict_path "$dataset")
    local config_file=$(get_config_path "$encoder" "$dataset" "$encoder_type")
    local embedding_dir=$(get_embedding_dir "$encoder" "$dataset")

    # Check if required files exist
    if [ ! -f "$ppi_file" ]; then
        echo "Error: PPI file not found: $ppi_file"
        return 1
    fi

    if [ ! -f "$protein_seq_file" ]; then
        echo "Error: Protein sequence file not found: $protein_seq_file"
        return 1
    fi

    if [ ! -f "$config_file" ]; then
        echo "Error: Config file not found: $config_file"
        return 1
    fi

    # Create output directory
    local output_dir="$PROJECT_ROOT/results/${experiment_name}"
    mkdir -p "$output_dir"

    # Create cache directory
    local cache_dir="$CACHE_ROOT/data/cache/${experiment_name}"
    mkdir -p "$cache_dir"

    echo "PPI File: $ppi_file"
    echo "Protein Seq File: $protein_seq_file"
    echo "Config File: $config_file"
    echo "Output Dir: $output_dir"
    echo "Cache Dir: $cache_dir"
    if [ -n "$embedding_dir" ]; then
        echo "Embedding Dir: $embedding_dir"
    fi

    # Run training
    echo "Starting training..."
    cd "$SCRIPT_DIR"

    # Build the command with optional embedding_dir parameter
    local encoding_type="precomputed"

    local cmd="python -B \"$PYTHON_SCRIPT\" \
        --ppi_file \"$ppi_file\" \
        --protein_seq_file \"$protein_seq_file\" \
        --pdb_dir \"$PROJECT_ROOT/data/pdbs\" \
        --cache_dir \"$cache_dir\" \
        --config \"$config_file\" \
        --encoding_type \"$encoding_type\" \
        --split_mode \"$split_method\" \
        --experiment_name \"$experiment_name\""

    # Add embedding_dir if specified
    if [ -n "$embedding_dir" ]; then
        cmd="$cmd --embedding_dir \"$embedding_dir\""
    fi

    # Add balance_dataset parameter for Arabidopsis and rice datasets
    if [ "$dataset" = "Arabidopsis" ] || [ "$dataset" = "rice" ]; then
        cmd="$cmd --balance_dataset"
        # Enable protein positive set split for balanced datasets
        cmd="$cmd --enable_protein_positive_split"
    fi
    cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 $cmd"
    # Execute the command
    echo "Full command: $cmd  > "$output_dir/training.log""
    eval "$cmd" > "$output_dir/training.log" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Training completed successfully for $experiment_name"
        # Copy log file to results directory
        cp "$output_dir/training.log" "$output_dir/"
    else
        echo "Training failed for $experiment_name"
        echo "Check log file: $output_dir/training.log"
    fi

    # Restore original data directory
    restore_data_directory

    return $exit_code
}

# Restore original data directory
restore_data_directory() {
    if [ -d "$DATA_DIR/processed_data_bk" ]; then
        echo "Restoring original processed_data directory..."
        rm -rf "$DATA_DIR/processed_data"
        mv "$DATA_DIR/processed_data_bk" "$DATA_DIR/processed_data"
    fi
}

# Main execution function
main() {
    echo "=== Sparse-SP-PPI Experiments Training ==="
    echo "Project Root: $PROJECT_ROOT"
    echo "Script Directory: $SCRIPT_DIR"
    echo ""

    # Parse command line arguments
    local specific_dataset=""
    local specific_encoder=""
    local specific_encoder_type=""
    local specific_split=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset)
                specific_dataset="$2"
                shift 2
                ;;
            --encoder)
                specific_encoder="$2"
                shift 2
                ;;
            --encoder-type)
                specific_encoder_type="$2"
                shift 2
                ;;
            --split)
                specific_split="$2"
                shift 2
                ;;
            --help)
                show_help
                return 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                return 1
                ;;
        esac
    done

    # Use specific values or all available
    local datasets_to_run=("${specific_dataset:-${DATASETS[@]}}")
    local encoders_to_run=("${specific_encoder:-${ENCODERS[@]}}")
    local encoder_types_to_run=("${specific_encoder_type:-${ENCODER_TYPES[@]}}")
    local splits_to_run=("${specific_split:-${SPLIT_METHODS[@]}}")

    echo "Datasets to run: ${datasets_to_run[*]}"
    echo "Encoders to run: ${encoders_to_run[*]}"
    echo "Encoder types to run: ${encoder_types_to_run[*]}"
    echo "Split methods to run: ${splits_to_run[*]}"
    echo ""

    # Create results directory
    mkdir -p "$PROJECT_ROOT/results"

    # Run all combinations
    local success_count=0
    local total_count=0

    for dataset in "${datasets_to_run[@]}"; do
        for encoder in "${encoders_to_run[@]}"; do
            for encoder_type in "${encoder_types_to_run[@]}"; do
                for split_method in "${splits_to_run[@]}"; do
                    ((total_count++))
                    train_experiment "$dataset" "$encoder" "$encoder_type" "$split_method"
                    if [ $? -eq 0 ]; then
                        ((success_count++))
                    fi
                    echo ""
                done
            done
        done
    done

    # Summary
    echo "=== Training Summary ==="
    echo "Total experiments: $total_count"
    echo "Successful: $success_count"
    echo "Failed: $((total_count - success_count))"
    echo ""
    echo "Results saved in: $PROJECT_ROOT/results/"

    if [ $success_count -eq $total_count ]; then
        echo "All experiments completed successfully!"
        return 0
    else
        echo "Some experiments failed. Check individual log files for details."
        return 1
    fi
}

# Show help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run Sparse-SP-PPI experiments with different configurations"
    echo ""
    echo "Options:"
    echo "  --dataset DATASET    Run specific dataset (default: all)"
    echo "                       Available: ${DATASETS[*]}"
    echo "  --encoder ENCODER    Run specific encoder (default: all)"
    echo "                       Available: ${ENCODERS[*]}"
    echo "  --encoder-type TYPE  Run specific encoder type (default: all)"
    echo "                       Available: ${ENCODER_TYPES[*]}"
    echo "  --split SPLIT       Run specific split method (default: all)"
    echo "                       Available: ${SPLIT_METHODS[*]}"
    echo "  --help              Show this help message"
    echo ""
    echo "Encoder Details:"
    echo "  esmc_600m  - ESMC-600m encoder (1152 dim, 2024-12)"
    echo ""
    echo "Encoder Type Details:"
    echo "  standard   - Standard protein encoder (default)"
    echo "  lrr        - LRR (Leucine-Rich Repeat) domain encoder"
    echo "  pep        - Peptide encoder (for small peptide interactions)"
    echo ""
    echo "Examples:"
    echo "  $0                            # Run all experiments"
    echo "  $0 --dataset SHS27k           # Run SHS27k with all splits"
    echo "  $0 --split bfs                # Run BFS split"
    echo "  $0 --encoder-type lrr         # Run all LRR experiments"
    echo "  $0 --dataset SHS27k --encoder-type lrr --split dfs  # Specific config"
    echo ""
}

# Execute main function
main "$@"
