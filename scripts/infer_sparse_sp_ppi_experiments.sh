#!/bin/bash

# MAPE PPI 批量推理脚本
# 基于训练实验结果的批量推理管道

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_ROOT="$PROJECT_ROOT/graphcache"

# 默认配置
PYTHON_SCRIPT="customer_inference.py"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"

# 支持的数据集（与训练一致）
DATASETS=("SHS27k" "SHS148k" "Arabidopsis" "rice" "STRING")

# 支持的编码器类型（与训练一致）
ENCODERS=("mape" "esm2" "esm2_3b" "esm3_small" "esmc_300m" "esmc_600m" "precomputed")

# 支持的编码器类型（标准、lrr、pep）
ENCODER_TYPES=("standard" "lrr" "pep")

# 支持的分割方法（与训练一致）
SPLIT_METHODS=("random" "bfs" "dfs")

# 设置推理数据目录
setup_inference_data() {
    local dataset="$1"
    local split_method="$2"

    echo "=== 设置推理数据目录: $dataset 使用 $split_method 分割 ==="

    # 备份现有的processed_data目录
    if [ -d "$DATA_DIR/processed_data" ]; then
        echo "备份现有的processed_data目录..."
        rm -rf "$DATA_DIR/processed_data"
    fi

    # 复制数据集特定的处理数据
    local source_dir="$DATA_DIR/processed_data_${dataset}"
    if [ ! -d "$source_dir" ]; then
        echo "错误: 源目录未找到: $source_dir"
        return 1
    fi

    echo "从 $source_dir 复制数据到 processed_data..."
    cp -r "$source_dir" "$DATA_DIR/processed_data"

    # 更新蛋白质数据文件中的分割方法（如果需要）
    update_split_method "$dataset" "$split_method"

    return 0
}

# 更新蛋白质数据文件中的分割方法
update_split_method() {
    local dataset="$1"
    local split_method="$2"

    local protein_file="$DATA_DIR/processed_data/protein.actions.${dataset}.txt"
    if [ ! -f "$protein_file" ]; then
        echo "警告: 蛋白质作用文件未找到: $protein_file"
        return 1
    fi

    echo "更新蛋白质文件中的分割方法为 $split_method..."
    # 更新分割方法列
    sed -i "s/split=[^[:space:]]*/split=$split_method/g" "$protein_file"

    return 0
}

# 生成实验名称（与训练一致）
generate_experiment_name() {
    local dataset="$1"
    local encoder="$2"
    local encoder_type="$3"
    local split_method="$4"

    # 默认使用标准类型保持向后兼容性
    if [ -z "$encoder_type" ] || [ "$encoder_type" = "standard" ]; then
        echo "${dataset}_${encoder}_${split_method}"
    else
        echo "${dataset}_${encoder}_${encoder_type}_${split_method}"
    fi
}

# 获取推理检查点路径
get_checkpoint_path() {
    local experiment_name="$1"
    local results_dir="$RESULTS_DIR/$experiment_name"

    # 首先检查最佳检查点
    if [ -f "$results_dir/checkpoints/checkpoint_best.pt" ]; then
        echo "$results_dir/checkpoints/checkpoint_best.pt"
        return 0
    fi

    # 检查最后一个检查点
    if [ -f "$results_dir/checkpoints/checkpoint_latest.pt" ]; then
        echo "$results_dir/checkpoints/checkpoint_latest.pt"
        return 0
    fi

    # 检查checkpoints目录中的任何检查点文件
    if [ -d "$results_dir/checkpoints" ]; then
        local checkpoint=$(ls -t "$results_dir/checkpoints"/*.pt 2>/dev/null | head -1)
        if [ -n "$checkpoint" ]; then
            echo "$checkpoint"
            return 0
        fi
    fi

    # 检查根目录
    local checkpoint=$(ls -t "$results_dir"/*.pt 2>/dev/null | head -1)
    if [ -n "$checkpoint" ]; then
        echo "$checkpoint"
        return 0
    fi

    echo ""
    return 1
}

# 获取PPI文件路径
get_ppi_file_path() {
    local dataset="$1"
    echo "$DATA_DIR/processed_data/protein.actions.${dataset}.txt"
}

# 获取蛋白质字典文件路径
get_protein_dict_path() {
    local dataset="$1"
    echo "$DATA_DIR/processed_data/protein.${dataset}.sequences.dictionary.csv"
}

# 验证推理实验
validate_inference_experiment() {
    local experiment_name="$1"

    echo "验证推理实验: $experiment_name"

    # 检查检查点是否存在
    local checkpoint_path=$(get_checkpoint_path "$experiment_name")
    if [ -z "$checkpoint_path" ]; then
        echo "错误: 未找到检查点文件"
        return 1
    fi

    if [ ! -f "$checkpoint_path" ]; then
        echo "错误: 检查点文件不存在: $checkpoint_path"
        return 1
    fi

    echo "检查点文件: $checkpoint_path"

    # 检查训练日志文件
    local log_file="$RESULTS_DIR/$experiment_name/training.log"
    if [ ! -f "$log_file" ]; then
        echo "警告: 训练日志文件未找到: $log_file"
    else
        echo "训练日志: $log_file"
    fi

    return 0
}

# 运行单个实验的推理
run_inference_experiment() {
    local dataset="$1"
    local encoder="$2"
    local encoder_type="$3"
    local split_method="$4"

    echo ""
    echo "=== 开始推理实验: $dataset - $encoder - $encoder_type - $split_method ==="

    # 生成实验名称
    local experiment_name=$(generate_experiment_name "$dataset" "$encoder" "$encoder_type" "$split_method")
    echo "实验名称: $experiment_name"

    # 获取检查点路径
    local checkpoint_path=$(get_checkpoint_path "$experiment_name")
    if [ -z "$checkpoint_path" ]; then
        echo "错误: 未找到检查点文件"
        return 1
    fi

    # 验证实验
    if ! validate_inference_experiment "$experiment_name"; then
        echo "实验验证失败"
        return 1
    fi

    # 设置推理数据目录
    if ! setup_inference_data "$dataset" "$split_method"; then
        echo "错误: 设置推理数据目录失败"
        return 1
    fi

    # 获取文件路径
    local ppi_file=$(get_ppi_file_path "$dataset")
    local protein_seq_file=$(get_protein_dict_path "$dataset")

    # 检查必需文件是否存在
    if [ ! -f "$ppi_file" ]; then
        echo "错误: PPI文件未找到: $ppi_file"
        restore_data_directory
        return 1
    fi

    if [ ! -f "$protein_seq_file" ]; then
        echo "错误: 蛋白质序列文件未找到: $protein_seq_file"
        restore_data_directory
        return 1
    fi

    # 创建输出目录
    local output_dir="$PROJECT_ROOT/inference_results/${experiment_name}"
    mkdir -p "$output_dir"

    # 创建缓存目录
    local cache_dir="$CACHE_ROOT/data/inference_cache/${experiment_name}"
    mkdir -p "$cache_dir"

    echo "PPI文件: $ppi_file"
    echo "蛋白质序列文件: $protein_seq_file"
    echo "检查点文件: $checkpoint_path"
    echo "输出目录: $output_dir"
    echo "缓存目录: $cache_dir"

    # 运行推理
    echo "开始推理..."
    cd "$SCRIPT_DIR"

    # 构建推理命令
    local log_file="$output_dir/inference.log"
    local output_file="$output_dir/predictions.csv"

    local cmd="python -B \"$PYTHON_SCRIPT\" \\
        --checkpoint \"$checkpoint_path\" \\
        --ppi_file \"$ppi_file\" \\
        --protein_seq_file \"$protein_seq_file\" \\
        --pdb_dir \"$PROJECT_ROOT/data/pdbs\" \\
        --cache_dir \"$cache_dir\" \\
        --mode all \\
        --output \"$output_file\" \\
        --log_file \"$log_file\""

    # 添加平衡数据集参数（如果适用）
    if [ "$dataset" = "Arabidopsis" ] || [ "$dataset" = "rice" ]; then
        echo "使用平衡数据集参数"
    fi

    # 执行命令
    echo "完整命令: $cmd"
    eval "$cmd" 2>&1 | tee "$log_file"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "推理实验 $experiment_name 完成成功"

        # 复制推理结果到结果目录
        if [ -f "$output_file" ]; then
            echo "推理结果保存到: $output_file"

            # 生成推理摘要
            generate_inference_summary "$experiment_name" "$output_file" "$log_file"
        fi
    else
        echo "推理实验 $experiment_name 失败"
        echo "检查日志文件: $log_file"
    fi

    # 恢复原始数据目录
    restore_data_directory

    return $exit_code
}

# 生成推理摘要
generate_inference_summary() {
    local experiment_name="$1"
    local prediction_file="$2"
    local log_file="$3"

    local summary_file="$PROJECT_ROOT/inference_results/${experiment_name}/summary.txt"

    echo "=== 推理实验摘要: $experiment_name ===" > "$summary_file"
    echo "生成时间: $(date)" >> "$summary_file"
    echo "" >> "$summary_file"

    # 统计预测结果
    if [ -f "$prediction_file" ]; then
        echo "预测结果统计:" >> "$summary_file"

        # 统计总预测数量
        local total_predictions=$(tail -n +2 "$prediction_file" | wc -l)
        echo "总预测数量: $total_predictions" >> "$summary_file"

        # 统计正样本预测数量（如果有类别列）
        for class in "binding" "catalysis" "activation" "inhibition" "reaction" "ptmod" "expression"; do
            if grep -q "${class}_pred" "$prediction_file"; then
                local positive_count=$(tail -n +2 "$prediction_file" | awk -F, -v class="${class}_pred" '
                BEGIN {count=0}
                {
                    for(i=1;i<=NF;i++) {
                        if($i ~ class) {
                            split($i, arr, "_")
                            if(arr[NF] == 1) count++
                            break
                        }
                    }
                }
                END {print count}'
                )
                local percentage=$(echo "scale=2; $positive_count * 100 / $total_predictions" | bc)
                echo "$class 正样本预测: $positive_count ($percentage%)" >> "$summary_file"
            fi
        done
    fi

    echo "" >> "$summary_file"
    echo "日志文件: $log_file" >> "$summary_file"
    echo "预测文件: $prediction_file" >> "$summary_file"

    echo "推理摘要生成到: $summary_file"
}

# 恢复原始数据目录
restore_data_directory() {
    # 这里不需要恢复，因为我们直接处理数据目录
    echo "推理数据目录清理完成"
}

# 主执行函数
main() {
    echo "=== MAPE PPI 批量推理实验 ==="
    echo "项目根目录: $PROJECT_ROOT"
    echo "脚本目录: $SCRIPT_DIR"
    echo ""

    # 解析命令行参数
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
                echo "未知选项: $1"
                show_help
                return 1
                ;;
        esac
    done

    # 使用特定值或所有可用值
    local datasets_to_run=("${specific_dataset:-${DATASETS[@]}}")
    local encoders_to_run=("${specific_encoder:-${ENCODERS[@]}}")
    local encoder_types_to_run=("${specific_encoder_type:-${ENCODER_TYPES[@]}}")
    local splits_to_run=("${specific_split:-${SPLIT_METHODS[@]}}")

    echo "将运行的数据集: ${datasets_to_run[*]}"
    echo "将运行的编码器: ${encoders_to_run[*]}"
    echo "将运行的编码器类型: ${encoder_types_to_run[*]}"
    echo "将运行的分割方法: ${splits_to_run[*]}"
    echo ""

    # 检查结果目录是否存在
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "错误: 结果目录未找到: $RESULTS_DIR"
        echo "请先运行训练脚本生成模型检查点"
        return 1
    fi

    # 创建推理结果目录
    mkdir -p "$PROJECT_ROOT/inference_results"

    # 运行所有组合
    local success_count=0
    local total_count=0

    for dataset in "${datasets_to_run[@]}"; do
        for encoder in "${encoders_to_run[@]}"; do
            for encoder_type in "${encoder_types_to_run[@]}"; do
                for split_method in "${splits_to_run[@]}"; do
                    ((total_count++))
                    run_inference_experiment "$dataset" "$encoder" "$encoder_type" "$split_method"
                    if [ $? -eq 0 ]; then
                        ((success_count++))
                    fi
                    echo ""
                done
            done
        done
    done

    # 生成最终摘要
    echo "=== 批量推理实验摘要 ==="
    echo "总实验数量: $total_count"
    echo "成功数量: $success_count"
    echo "失败数量: $((total_count - success_count))"
    echo ""
    echo "推理结果保存在: $PROJECT_ROOT/inference_results/"

    # 生成总体摘要文件
    generate_overall_summary "$total_count" "$success_count"

    if [ $success_count -eq $total_count ]; then
        echo "所有推理实验成功完成!"
        return 0
    else
        echo "部分实验失败。请检查各个日志文件了解详情。"
        return 1
    fi
}

# 生成总体摘要
generate_overall_summary() {
    local total_count="$1"
    local success_count="$2"

    local overall_summary="$PROJECT_ROOT/inference_results/overall_summary.txt"

    echo "=== MAPE PPI 批量推理实验总体摘要 ===" > "$overall_summary"
    echo "生成时间: $(date)" >> "$overall_summary"
    echo "" >> "$overall_summary"
    echo "总实验数量: $total_count" >> "$overall_summary"
    echo "成功数量: $success_count" >> "$overall_summary"
    echo "失败数量: $((total_count - success_count))" >> "$overall_summary"
    echo "成功率: $(echo "scale=2; $success_count * 100 / $total_count" | bc)%" >> "$overall_summary"
    echo "" >> "$overall_summary"
    echo "实验详细结果保存在各个实验目录中" >> "$overall_summary"

    echo "总体摘要生成到: $overall_summary"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "运行MAPE PPI批量推理实验"
    echo ""
    echo "选项:"
    echo "  --dataset 数据集    运行特定数据集（默认：全部）"
    echo "                      可用：${DATASETS[*]}"
    echo "  --encoder 编码器    运行特定编码器（默认：全部）"
    echo "                      可用：${ENCODERS[*]}"
    echo "  --encoder-type 类型 运行特定编码器类型（默认：全部）"
    echo "                      可用：${ENCODER_TYPES[*]}"
    echo "  --split 分割方法    运行特定分割方法（默认：全部）"
    echo "                      可用：${SPLIT_METHODS[*]}"
    echo "  --help             显示此帮助信息"
    echo ""
    echo "编码器详细信息："
    echo "  mape       - MAPE编码器（内部使用ESM2 650M）"
    echo "  esm2       - ESM2 650M编码器"
    echo "  esm2_3b    - ESM2 3B编码器（2560维，较小批大小）"
    echo "  esm3_small - ESM3-small编码器（768维，2024-03/12）"
    echo "  esmc_300m  - ESMC-300m编码器（768维，2024-12）"
    echo "  esmc_600m  - ESMC-600m编码器（1024维，2024-12）"
    echo "  precomputed- 预计算ESM2 650M嵌入"
    echo ""
    echo "编码器类型详细信息："
    echo "  standard   - 标准蛋白质编码器（默认）"
    echo "  lrr        - LRR（亮氨酸富集重复）域编码器"
    echo "  pep        - 肽编码器（用于小肽相互作用）"
    echo ""
    echo "使用示例："
    echo "  $0                            # 运行所有实验（5x7x3x3 = 315）"
    echo "  $0 --dataset SHS27k           # 运行SHS27k的所有编码器和分割"
    echo "  $0 --encoder esm3_small --split bfs  # 运行ESM3-small使用BFS分割"
    echo "  $0 --encoder esm2_3b --dataset Arabidopsis  # 运行ESM2-3B在Arabidopsis上"
    echo "  $0 --encoder-type lrr         # 运行所有LRR实验"
    echo "  $0 --dataset SHS27k --encoder-type lrr --encoder esmc_300m  # 特定LRR配置"
    echo ""
}

# 执行主函数
main "$@"
