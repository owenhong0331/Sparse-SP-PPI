#!/bin/bash

# Batch CSV to FASTA Converter
# Automatically converts all dataset CSV files to individual FASTA files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/csv_to_fasta_converter.py"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_BASE_DIR="$PROJECT_ROOT/fasta_files"

# Datasets to process
DATASETS=("Arabidopsis" "rice" "SHS27k" "SHS148k" "STRING")

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

echo "=== Batch CSV to FASTA Conversion ==="
echo "Project Root: $PROJECT_ROOT"
echo "Data Dir: $DATA_DIR"
echo "Output Base Dir: $OUTPUT_BASE_DIR"
echo ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Function to process a single dataset
process_dataset() {
    local dataset="$1"
    local data_path="$DATA_DIR/processed_data_$dataset"
    local output_dir="$OUTPUT_BASE_DIR/$dataset"
    local csv_file=""

    echo "Processing dataset: $dataset"
    echo "Data path: $data_path"

    # Find the appropriate CSV file
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
            echo "Error: No CSV file found for dataset $dataset in $data_path"
            return 1
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

    # Convert CSV to FASTA
    echo "Running: python "$PYTHON_SCRIPT" --csv "$csv_file" --output-dir "$output_dir" --dataset-name "$dataset""
    python "$PYTHON_SCRIPT" --csv "$csv_file" --output-dir "$output_dir" --dataset-name "$dataset"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Successfully processed $dataset"
        # Count the number of FASTA files generated
        local num_files=$(find "$output_dir" -name "*.fasta" | wc -l)
        echo "  Generated $num_files FASTA files"
    else
        echo "Failed to process $dataset"
    fi

    echo ""
    return $exit_code
}

# Process all datasets
success_count=0
total_count=0

for dataset in "${DATASETS[@]}"; do
    ((total_count++))
    process_dataset "$dataset"
    if [ $? -eq 0 ]; then
        ((success_count++))
    fi
done

echo "=== Summary ==="
echo "Total datasets: $total_count"
echo "Successful: $success_count"
echo "Failed: $((total_count - success_count))"
echo ""

# Generate final directory structure summary
echo "=== Generated Directory Structure ==="
find "$OUTPUT_BASE_DIR" -type d | sort | sed 's|'"$OUTPUT_BASE_DIR"'|  |'

echo ""
echo "FASTA files are saved in: $OUTPUT_BASE_DIR"
echo "Each dataset has its own subdirectory with individual FASTA files"

# Count total FASTA files
total_fasta_files=$(find "$OUTPUT_BASE_DIR" -name "*.fasta" | wc -l)
echo "Total FASTA files generated: $total_fasta_files"

if [ $success_count -eq $total_count ]; then
    echo "All datasets converted successfully!"
    exit 0
else
    echo "Some datasets failed. Check the logs above for details."
    exit 1
fi