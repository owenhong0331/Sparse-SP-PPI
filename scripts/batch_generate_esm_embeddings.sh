#!/bin/bash

# Batch ESM Embeddings Generator
# Automatically generates embeddings for all models and datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EMBEDDING_DIR="$PROJECT_ROOT/embedding"
DATA_DIR="$PROJECT_ROOT/data"
PYTHON_SCRIPT="$SCRIPT_DIR/generate_esm_embeddings_v2.py"

# Models to generate embeddings for
# MODELS=("esm2_t33_650M_UR50D" "esm2_t36_3B_UR50D" "esm2_t48_15B_UR50D" "esm3-small-2024-03" "esmc-300m-2024-12" "esmc-600m-2024-12")
# MODELS=("esm2_t33_650M_UR50D" "esm2_t36_3B_UR50D" "esm3-small-2024-03" "esmc-300m-2024-12" "esmc-600m-2024-12")
MODELS=("esm3-small-2024-03" "esmc-300m-2024-12" "esmc-600m-2024-12")
# Note: ESM3 medium model is not available

# Datasets to process
# DATASETS=("Arabidopsis" "rice" "SHS27k" "SHS148k" "STRING")
DATASETS=("Arabidopsis" "rice" "SHS27k")

# Create embedding directory if it doesn't exist
mkdir -p "$EMBEDDING_DIR"

echo "=== Batch ESM Embeddings Generation ==="
echo "Project Root: $PROJECT_ROOT"
echo "Embedding Dir: $EMBEDDING_DIR"
echo "Data Dir: $DATA_DIR"
echo ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Function to process a single dataset for a given model
process_dataset() {
    local model="$1"
    local dataset="$2"
    local data_path="$DATA_DIR/processed_data_$dataset"
    local output_dir="$EMBEDDING_DIR/$model/$dataset"
    local csv_file=""

    echo "Processing: $dataset with $model"

    # Find the appropriate CSV file - first check if dataset-specific directory exists
    if [ -d "$data_path" ]; then
        if [ -f "$data_path/protein.$dataset.sequences.dictionary.csv" ]; then
            csv_file="$data_path/protein.$dataset.sequences.dictionary.csv"
        elif [ -f "$data_path/protein.STRING.sequences.dictionary.csv" ] && [ "$dataset" = "STRING" ]; then
            csv_file="$data_path/protein.STRING.sequences.dictionary.csv"
        elif [ -f "$data_path/protein.SHS27k.sequences.dictionary.csv" ] && [ "$dataset" = "SHS27k" ]; then
            csv_file="$data_path/protein.SHS27k.sequences.dictionary.csv"
        elif [ -f "$data_path/protein.SHS148k.sequences.dictionary.csv" ] && [ "$dataset" = "SHS148k" ]; then
            csv_file="$data_path/protein.SHS148k.sequences.dictionary.csv"
        else
            # Try to find any CSV file in the directory
            csv_file=$(find "$data_path" -name "*.sequences.dictionary.csv" | head -1)
            if [ -z "$csv_file" ]; then
                echo "Warning: No CSV file found for dataset $dataset in $data_path"
                echo "Trying main processed_data directory..."
                # Fallback to main processed_data directory
                main_data_path="$DATA_DIR/processed_data"
                if [ -f "$main_data_path/protein.$dataset.sequences.dictionary.csv" ]; then
                    csv_file="$main_data_path/protein.$dataset.sequences.dictionary.csv"
                elif [ -f "$main_data_path/protein.SHS27k.sequences.dictionary.csv" ] && [ "$dataset" = "SHS27k" ]; then
                    csv_file="$main_data_path/protein.SHS27k.sequences.dictionary.csv"
                else
                    csv_file=$(find "$main_data_path" -name "*.sequences.dictionary.csv" | head -1)
                    if [ -z "$csv_file" ]; then
                        echo "Error: No CSV file found for dataset $dataset"
                        return 1
                    fi
                fi
            fi
            echo "Using CSV file: $(basename "$csv_file")"
        fi
    else
        # Dataset directory doesn't exist, try main processed_data directory
        echo "Dataset directory $data_path not found, trying main processed_data directory..."
        main_data_path="$DATA_DIR/processed_data"
        if [ -f "$main_data_path/protein.$dataset.sequences.dictionary.csv" ]; then
            csv_file="$main_data_path/protein.$dataset.sequences.dictionary.csv"
        elif [ -f "$main_data_path/protein.SHS27k.sequences.dictionary.csv" ] && [ "$dataset" = "SHS27k" ]; then
            csv_file="$main_data_path/protein.SHS27k.sequences.dictionary.csv"
        else
            csv_file=$(find "$main_data_path" -name "*.sequences.dictionary.csv" | head -1)
            if [ -z "$csv_file" ]; then
                echo "Error: No CSV file found for dataset $dataset"
                return 1
            fi
        fi
        echo "Using CSV file: $(basename "$csv_file")"
    fi

    if [ ! -f "$csv_file" ]; then
        echo "Error: CSV file not found: $csv_file"
        return 1
    fi

    echo "CSV File: $csv_file"
    echo "Output Dir: $output_dir"

    # Create output directory
    mkdir -p "$output_dir"

    # Generate embeddings
    echo "Running: python "$PYTHON_SCRIPT" --model "$model" --csv_dict "$csv_file" --output_dir "$output_dir""

    # Check if HF token is available in environment
    if [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
        echo "Using Hugging Face token from environment"
        python "$PYTHON_SCRIPT" --model "$model" --csv_dict "$csv_file" --output_dir "$output_dir" --hf_token "$HUGGINGFACE_HUB_TOKEN"
    else
        python "$PYTHON_SCRIPT" --model "$model" --csv_dict "$csv_file" --output_dir "$output_dir"
    fi

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Successfully processed $dataset with $model"
        # Count the number of embedding files generated
        local num_files=$(find "$output_dir" -name "*.npy" | wc -l)
        echo "  Generated $num_files embedding files"
    else
        echo "Failed to process $dataset with $model"
    fi

    echo ""
    return $exit_code
}

# Process all combinations
success_count=0
total_count=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        ((total_count++))
        process_dataset "$model" "$dataset"
        if [ $? -eq 0 ]; then
            ((success_count++))
        fi
    done
done

echo "=== Summary ==="
echo "Total tasks: $total_count"
echo "Successful: $success_count"
echo "Failed: $((total_count - success_count))"
echo ""

# Generate final directory structure summary
echo "=== Generated Directory Structure ==="
find "$EMBEDDING_DIR" -type d | sort | sed 's|'"$EMBEDDING_DIR"'|  |'

echo ""
echo "Embedding files are saved in: $EMBEDDING_DIR"
echo "Each model has its own subdirectory with dataset-specific embeddings"

if [ $success_count -eq $total_count ]; then
    echo "All tasks completed successfully!"
    exit 0
else
    echo "Some tasks failed. Check the logs above for details."
    exit 1
fi