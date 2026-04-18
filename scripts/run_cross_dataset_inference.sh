#!/bin/bash

# Cross-Dataset Inference Script for Sparse-SP-PPI
# Performs inference on a test dataset using a model trained on a different dataset
# Supports: SYS30k->SYS60k, SHS27k->SHS148k, SHS148k->STRING, etc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_ROOT="$PROJECT_ROOT/graphcache"

# Default configurations
PYTHON_SCRIPT="inference.py"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"
OUTPUT_DIR="$PROJECT_ROOT/cross_dataset_results"

# Available datasets
ALL_DATASETS=("SHS27k" "SHS148k" "Arabidopsis" "rice" "STRING" "SYS30k" "SYS60k")

# Available encoders
ENCODERS=("esmc_600m")

# Available encoder types (standard, lrr, pep)
ENCODER_TYPES=("standard" "lrr" "pep")

# Available split methods
SPLIT_METHODS=("random" "bfs" "dfs")

# Default cross-dataset pairs (can be overridden)
CROSS_DATASET_PAIRS=(
    "SYS30k:SYS60k"
    "SHS27k:SHS148k"
    "SHS148k:STRING"
)

# Show help
show_help() {
    echo "Cross-Dataset Inference Script for Sparse-SP-PPI"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --train_dataset DATASET     Training dataset name (e.g., SYS30k, SHS27k)"
    echo "  --test_dataset DATASET      Testing dataset name (e.g., SYS60k, SHS148k)"
    echo "  --encoder ENCODER           Encoder type (default: esmc_600m)"
    echo "  --encoder-type TYPE         Encoder type: standard, lrr, pep (default: lrr)"
    echo "  --split SPLIT               Split method (default: random)"
    echo "  --run_all                   Run all predefined cross-dataset pairs"
    echo "  --list_pairs                List available cross-dataset pairs"
    echo "  --results_dir DIR           Results directory (default: $RESULTS_DIR)"
    echo "  --output_dir DIR            Output directory (default: $OUTPUT_DIR)"
    echo "  --help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  # SYS30k -> SYS60k"
    echo "  $0 --train_dataset SYS30k --test_dataset SYS60k"
    echo ""
    echo "  # SHS27k -> SHS148k with LRR encoder"
    echo "  $0 --train_dataset SHS27k --test_dataset SHS148k --encoder-type lrr --split dfs"
    echo ""
    echo "  # Run all predefined pairs"
    echo "  $0 --run_all"
    echo ""
    echo "Available datasets: ${ALL_DATASETS[*]}"
    echo "Available encoders: ${ENCODERS[*]}"
    echo "Available encoder types: ${ENCODER_TYPES[*]}"
    echo "Available splits: ${SPLIT_METHODS[*]}"
    echo ""
    echo "Predefined cross-dataset pairs:"
    for pair in "${CROSS_DATASET_PAIRS[@]}"; do
        IFS=':' read -r train test <<< "$pair"
        echo "  $train -> $test"
    done
}

# Generates experiment name (consistent with train_sparse_sp_ppi_experiments.sh)
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

# Generate cross-dataset experiment name
generate_cross_experiment_name() {
    local train_dataset="$1"
    local test_dataset="$2"
    local encoder="$3"
    local encoder_type="$4"
    local split_method="$5"

    if [ -z "$encoder_type" ] || [ "$encoder_type" = "standard" ]; then
        echo "${train_dataset}_to_${test_dataset}_${encoder}_${split_method}"
    else
        echo "${train_dataset}_to_${test_dataset}_${encoder}_${encoder_type}_${split_method}"
    fi
}

# Get checkpoint path for a training experiment
# Searches multiple possible locations for model checkpoints
get_checkpoint_path() {
    local train_dataset="$1"
    local encoder="$2"
    local encoder_type="$3"
    local split_method="$4"

    local experiment_name=$(generate_experiment_name "$train_dataset" "$encoder" "$encoder_type" "$split_method")
    local exp_results_dir="$RESULTS_DIR/$experiment_name"
    local logs_dir="$PROJECT_ROOT/logs"

    echo "Searching for checkpoint: $experiment_name" >&2

    # Search locations in order of preference
    local search_paths=()

    # 1. Standard location: results/{experiment_name}/checkpoints/
    search_paths+=("$exp_results_dir/checkpoints")

    # 2. Logs location: logs/{experiment_name}_*/checkpoints/
    # Sort by modification time (newest first) to get the latest training run
    local log_dirs=()
    for log_dir in "$logs_dir"/${experiment_name}_*/checkpoints; do
        if [ -d "$log_dir" ]; then
            log_dirs+=("$log_dir")
        fi
    done

    # Sort directories by modification time (newest first)
    if [ ${#log_dirs[@]} -gt 0 ]; then
        IFS=$'\n' sorted_dirs=($(ls -td "${log_dirs[@]}" 2>/dev/null))
        unset IFS
        for log_dir in "${sorted_dirs[@]}"; do
            search_paths+=("$log_dir")
        done
    fi

    # 3. Dataset-based location: results/{dataset}/*/
    for dataset_dir in "$RESULTS_DIR"/$train_dataset/*; do
        if [ -d "$dataset_dir" ]; then
            search_paths+=("$dataset_dir/checkpoints")
            search_paths+=("$dataset_dir/SEES_0")
        fi
    done

    # Search for checkpoint files in all locations
    for search_dir in "${search_paths[@]}"; do
        if [ ! -d "$search_dir" ]; then
            continue
        fi

        echo "  Checking: $search_dir" >&2

        # Check for best checkpoint (.pt or .pth) - various naming conventions
        if [ -f "$search_dir/best_model.pt" ]; then
            echo "  Found: best_model.pt" >&2
            echo "$search_dir/best_model.pt"
            return 0
        fi

        if [ -f "$search_dir/best_model.pth" ]; then
            echo "  Found: best_model.pth" >&2
            echo "$search_dir/best_model.pth"
            return 0
        fi

        if [ -f "$search_dir/checkpoint_best.pt" ]; then
            echo "  Found: checkpoint_best.pt" >&2
            echo "$search_dir/checkpoint_best.pt"
            return 0
        fi

        if [ -f "$search_dir/checkpoint_best.pth" ]; then
            echo "  Found: checkpoint_best.pth" >&2
            echo "$search_dir/checkpoint_best.pth"
            return 0
        fi

        # Check for latest checkpoint
        if [ -f "$search_dir/checkpoint_latest.pt" ]; then
            echo "  Found: checkpoint_latest.pt" >&2
            echo "$search_dir/checkpoint_latest.pt"
            return 0
        fi

        if [ -f "$search_dir/checkpoint_latest.pth" ]; then
            echo "  Found: checkpoint_latest.pth" >&2
            echo "$search_dir/checkpoint_latest.pth"
            return 0
        fi

        # Check for model_state.pth (old format)
        if [ -f "$search_dir/model_state.pth" ]; then
            echo "  Found: model_state.pth" >&2
            echo "$search_dir/model_state.pth"
            return 0
        fi

        # Check for any .pt file
        local pt_file=$(ls -t "$search_dir"/*.pt 2>/dev/null | head -1)
        if [ -n "$pt_file" ]; then
            echo "  Found: $(basename "$pt_file")" >&2
            echo "$pt_file"
            return 0
        fi

        # Check for any .pth file
        local pth_file=$(ls -t "$search_dir"/*.pth 2>/dev/null | head -1)
        if [ -n "$pth_file" ]; then
            echo "  Found: $(basename "$pth_file")" >&2
            echo "$pth_file"
            return 0
        fi
    done

    echo "  No checkpoint found!" >&2
    echo ""
    return 1
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

# Get PPI file path for a dataset
get_ppi_file_path() {
    local dataset="$1"
    echo "$DATA_DIR/processed_data_${dataset}/protein.actions.${dataset}.txt"
}

# Get protein sequence file path for a dataset
get_protein_seq_file_path() {
    local dataset="$1"
    echo "$DATA_DIR/processed_data_${dataset}/protein.${dataset}.sequences.dictionary.csv"
}

# Get embedding directory for a dataset and encoder
get_embedding_dir() {
    local encoder="$1"
    local dataset="$2"
    echo "$PROJECT_ROOT/embedding/esmc-600m-2024-12/${dataset}"
}

# Setup data directory for inference
setup_data_directory() {
    local test_dataset="$1"

    echo "=== Setting up data directory for $test_dataset ==="

    # Backup existing processed_data directory
    if [ -d "$DATA_DIR/processed_data" ]; then
        echo "Backing up existing processed_data directory..."
        rm -rf "$DATA_DIR/processed_data"
    fi

    # Copy test dataset processed data
    local source_dir="$DATA_DIR/processed_data_${test_dataset}"
    if [ ! -d "$source_dir" ]; then
        echo "Error: Source directory not found: $source_dir"
        return 1
    fi

    echo "Copying data from $source_dir to processed_data..."
    cp -r "$source_dir" "$DATA_DIR/processed_data"

    return 0
}

# Run single cross-dataset inference
run_cross_dataset_inference() {
    local train_dataset="$1"
    local test_dataset="$2"
    local encoder="$3"
    local split_method="$4"
    local encoder_type="${5:-standard}"

    echo ""
    echo "========================================"
    echo "Cross-Dataset Inference"
    echo "Training Dataset: $train_dataset"
    echo "Testing Dataset: $test_dataset"
    echo "Encoder: $encoder"
    echo "Encoder Type: $encoder_type"
    echo "Split Method: $split_method"
    echo "========================================"

    # Generate experiment name (same as training)
    local train_experiment_name=$(generate_experiment_name "$train_dataset" "$encoder" "$encoder_type" "$split_method")
    echo "Training Experiment: $train_experiment_name"

    # Get checkpoint path
    local checkpoint_path=$(get_checkpoint_path "$train_dataset" "$encoder" "$encoder_type" "$split_method")
    if [ -z "$checkpoint_path" ]; then
        echo "Error: No checkpoint found for training dataset: $train_dataset ($encoder, $encoder_type, $split_method)"
        echo "   Please train the model first using train_mape_experiments.sh"
        return 1
    fi

    echo "Checkpoint: $checkpoint_path"

    # Get test dataset file paths
    local ppi_file=$(get_ppi_file_path "$test_dataset")
    local protein_seq_file=$(get_protein_seq_file_path "$test_dataset")

    # Validate test dataset files
    if [ ! -f "$ppi_file" ]; then
        echo "Error: PPI file not found: $ppi_file"
        return 1
    fi

    if [ ! -f "$protein_seq_file" ]; then
        echo "Error: Protein sequence file not found: $protein_seq_file"
        return 1
    fi

    echo "PPI File: $ppi_file"
    echo "Protein Seq File: $protein_seq_file"

    # Setup data directory
    if ! setup_data_directory "$test_dataset"; then
        echo "Error: Failed to setup data directory"
        return 1
    fi

    # Create output directory
    local exp_output_dir="$OUTPUT_DIR/${train_dataset}_to_${test_dataset}/${encoder}_${encoder_type}_${split_method}"
    mkdir -p "$exp_output_dir"

    # Generate test experiment name for cache directory
    local test_experiment_name=$(generate_experiment_name "$test_dataset" "$encoder" "$encoder_type" "$split_method")

    # Use the test dataset's cache directory (for cross-dataset inference)
    local cache_dir="$CACHE_ROOT/data/cache/${test_experiment_name}"
    echo "Using test dataset cache directory: $cache_dir"

    echo "Output Dir: $exp_output_dir"
    echo "Cache Dir: $cache_dir"

    # Get config file path (consistent with train_mape_experiments.sh)
    # Use training dataset for config selection with encoder_type
    local config_file=$(get_config_path "$encoder" "$train_dataset" "$encoder_type")
    echo "Config File: $config_file"

    # Validate config file exists
    if [ ! -f "$config_file" ]; then
        echo "Warning: Config file not found: $config_file"
        echo "   Will try to use default config from checkpoint or minimal config"
        config_file=""
    fi

    # Build inference command
    local output_file="$exp_output_dir/predictions.csv"
    local log_file="$exp_output_dir/inference.log"

    local cmd="python -B \"$PYTHON_SCRIPT\" \
        --checkpoint \"$checkpoint_path\" \
        --ppi_file \"$ppi_file\" \
        --protein_seq_file \"$protein_seq_file\" \
        --pdb_dir \"$PROJECT_ROOT/data/pdbs\" \
        --cache_dir \"$cache_dir\" \
        --train_dataset \"$train_dataset\" \
        --test_dataset \"$test_dataset\" \
        --cross_dataset \
        --evaluate \
        --output \"$output_file\" \
        --log_file \"$log_file\""
    cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 $cmd"
    # Add config file if available
    if [ -n "$config_file" ] && [ -f "$config_file" ]; then
        cmd="$cmd --config \"$config_file\""
    fi

    # Add embedding directory if needed
    local embedding_dir=$(get_embedding_dir "$encoder" "$test_dataset")
    if [ -n "$embedding_dir" ]; then
        cmd="$cmd --embedding_dir \"$embedding_dir\""
    fi

    # Execute inference (consistent with train_mape_experiments.sh)
    echo ""
    echo "Running inference..."
    echo "Full command: $cmd > \"$exp_output_dir/inference.log\""
    echo ""

    cd "$SCRIPT_DIR"
    eval "$cmd" > "$exp_output_dir/inference.log" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "Cross-dataset inference completed successfully!"
        echo "Results saved to: $exp_output_dir"

        # Generate summary
        generate_summary "$train_dataset" "$test_dataset" "$encoder" "$encoder_type" "$split_method" "$exp_output_dir"
    else
        echo ""
        echo "Cross-dataset inference failed!"
        echo "Check log file: $exp_output_dir/inference.log"
    fi

    # Cleanup
    if [ -d "$DATA_DIR/processed_data" ]; then
        rm -rf "$DATA_DIR/processed_data"
    fi

    return $exit_code
}

# Generate summary for a cross-dataset inference run
generate_summary() {
    local train_dataset="$1"
    local test_dataset="$2"
    local encoder="$3"
    local encoder_type="$4"
    local split_method="$5"
    local exp_output_dir="$6"

    local summary_file="$exp_output_dir/summary.txt"
    local metrics_file="$exp_output_dir/predictions_metrics.json"

    echo "Generating summary..."

    {
        echo "=============================================="
        echo "Cross-Dataset Inference Summary"
        echo "=============================================="
        echo "Training Dataset: $train_dataset"
        echo "Testing Dataset: $test_dataset"
        echo "Encoder: $encoder"
        echo "Encoder Type: $encoder_type"
        echo "Split Method: $split_method"
        echo "Timestamp: $(date)"
        echo ""

        # Include metrics if available
        if [ -f "$metrics_file" ]; then
            echo "Metrics:"
            cat "$metrics_file"
        else
            echo "Metrics file not found: $metrics_file"
        fi

        echo ""
        echo "Output Directory: $exp_output_dir"
        echo "=============================================="
    } > "$summary_file"

    echo "Summary saved to: $summary_file"
}

# Run all predefined cross-dataset pairs
run_all_pairs() {
    echo "Running all predefined cross-dataset pairs..."
    echo ""

    local success_count=0
    local total_count=0

    for pair in "${CROSS_DATASET_PAIRS[@]}"; do
        IFS=':' read -r train_dataset test_dataset <<< "$pair"

        # Use default encoder and split
        local encoder="esmc_600m"
        local split_method="random"

        ((total_count++))

        echo ""
        echo "[$total_count/${#CROSS_DATASET_PAIRS[@]}] Running: $train_dataset -> $test_dataset"

        run_cross_dataset_inference "$train_dataset" "$test_dataset" "$encoder" "$split_method"

        if [ $? -eq 0 ]; then
            ((success_count++))
        fi

        echo ""
        echo "----------------------------------------"
    done

    # Generate overall summary
    generate_overall_summary "$success_count" "$total_count"
}

# Generate overall summary
generate_overall_summary() {
    local success_count="$1"
    local total_count="$2"

    local overall_summary="$OUTPUT_DIR/overall_summary.txt"

    {
        echo "=============================================="
        echo "Cross-Dataset Inference Overall Summary"
        echo "=============================================="
        echo "Timestamp: $(date)"
        echo ""
        echo "Total experiments: $total_count"
        echo "Successful: $success_count"
        echo "Failed: $((total_count - success_count))"
        echo "Success rate: $(echo "scale=2; $success_count * 100 / $total_count" | bc)%"
        echo ""
        echo "Results saved in: $OUTPUT_DIR"
        echo "=============================================="
    } > "$overall_summary"

    echo ""
    echo "=============================================="
    echo "Overall Summary"
    echo "=============================================="
    cat "$overall_summary"
}

# List available pairs
list_pairs() {
    echo "Available Cross-Dataset Pairs:"
    echo ""
    echo "Predefined pairs:"
    for pair in "${CROSS_DATASET_PAIRS[@]}"; do
        IFS=':' read -r train test <<< "$pair"
        echo "  $train -> $test"
    done
    echo ""
    echo "You can also specify custom pairs using --train_dataset and --test_dataset"
    echo ""
    echo "Available datasets: ${ALL_DATASETS[*]}"
}

# Main function
main() {
    # Default values
    local train_dataset=""
    local test_dataset=""
    local encoder="esmc_600m"
    local encoder_type="lrr"
    local split_method="random"
    local run_all=false
    local list_pairs_flag=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --train_dataset)
                train_dataset="$2"
                shift 2
                ;;
            --test_dataset)
                test_dataset="$2"
                shift 2
                ;;
            --encoder)
                encoder="$2"
                shift 2
                ;;
            --encoder-type)
                encoder_type="$2"
                shift 2
                ;;
            --split)
                split_method="$2"
                shift 2
                ;;
            --run_all)
                run_all=true
                shift
                ;;
            --list_pairs)
                list_pairs_flag=true
                shift
                ;;
            --results_dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --output_dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # List pairs if requested
    if [ "$list_pairs_flag" = true ]; then
        list_pairs
        exit 0
    fi

    # Run all pairs if requested
    if [ "$run_all" = true ]; then
        run_all_pairs
        exit $?
    fi

    # Validate required arguments for single run
    if [ -z "$train_dataset" ] || [ -z "$test_dataset" ]; then
        echo "Error: --train_dataset and --test_dataset are required (or use --run_all)"
        echo ""
        show_help
        exit 1
    fi

    # Validate datasets
    local train_valid=false
    local test_valid=false

    for ds in "${ALL_DATASETS[@]}"; do
        if [ "$ds" = "$train_dataset" ]; then
            train_valid=true
        fi
        if [ "$ds" = "$test_dataset" ]; then
            test_valid=true
        fi
    done

    if [ "$train_valid" = false ]; then
        echo "Error: Invalid training dataset: $train_dataset"
        echo "Available datasets: ${ALL_DATASETS[*]}"
        exit 1
    fi

    if [ "$test_valid" = false ]; then
        echo "Error: Invalid testing dataset: $test_dataset"
        echo "Available datasets: ${ALL_DATASETS[*]}"
        exit 1
    fi

    # Validate encoder type
    local encoder_type_valid=false
    for et in "${ENCODER_TYPES[@]}"; do
        if [ "$et" = "$encoder_type" ]; then
            encoder_type_valid=true
        fi
    done

    if [ "$encoder_type_valid" = false ]; then
        echo "Error: Invalid encoder type: $encoder_type"
        echo "Available encoder types: ${ENCODER_TYPES[*]}"
        exit 1
    fi

    # Run single cross-dataset inference
    run_cross_dataset_inference "$train_dataset" "$test_dataset" "$encoder" "$split_method" "$encoder_type"
    exit $?
}

# Execute main function
main "$@"
