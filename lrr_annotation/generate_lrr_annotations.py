#!/usr/bin/env python3
"""
PDB to LRR Annotation Generator (Robust Version with Progress Bars)
====================================================================

从PDB文件生成LRR注释的脚本 - 健壮版本，包含所有步骤的进度条
单个结构失败不会停止整个流程，仅记录并跳过

作者: AI助手
基于Danielle M. Stevens的mamp-ml LRR_Annotation代码
"""

import sys
import os
from pathlib import Path
import argparse
import traceback
from tqdm import tqdm
import numpy as np
import json
import pickle

# 本脚本位于 lrr_annotation/ 目录下，geom_lrr 和 extract_lrr_sequences 是同级模块
# 直接添加当前目录到 sys.path 以确保导入正常
sys.path.insert(0, str(Path(__file__).parent))

try:
    # 导入LRR_Annotation模块中的核心类
    from geom_lrr import Loader, Analyzer, Plotter
    from extract_lrr_sequences import LRRSequenceExtractor
except ImportError as e:
    print(f"导入LRR_Annotation模块时出错: {e}")
    print("请确保在 Sparse-SP-PPI 项目根目录下运行此脚本")
    sys.exit(1)


class RobustLoaderWithProgress:
    """
    健壮的PDB加载器，包含进度条和错误处理
    单个文件加载失败不会停止整个流程
    """
    def __init__(self, failed_files_output="failed_loading.txt"):
        self.structures = {}
        self.bfactors = {}
        self.failed_files = []
        self.failed_files_output = failed_files_output  # 加载失败记录文件
        
    def load_single_pdb_safe(self, pdb_file_path):
        """
        安全地加载单个PDB文件
        遇到错误时记录并返回False，但不抛出异常
        """
        from Bio.PDB import PDBParser
        
        try:
            parser = PDBParser(QUIET=True)
            pdb_path = Path(pdb_file_path)
            key = pdb_path.stem
            
            # 解析PDB文件
            structure = parser.get_structure(key, str(pdb_path))
            
            # 获取第一个链
            chain = next(structure.get_chains())
            
            # 安全地提取CA原子坐标
            ca_coords = []
            
            for residue in chain.get_residues():
                try:
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        ca_coords.append(np.array(list(ca_atom.get_vector())))
                except Exception:
                    # 跳过有问题的残基
                    continue
            
            if len(ca_coords) == 0:
                raise ValueError("没有找到有效的CA原子")
            
            # 检查是否有足够的残基
            if len(ca_coords) < 10:
                raise ValueError("残基数太少，无法进行几何分析")
            
            self.structures[key] = np.array(ca_coords)
            return True
            
        except Exception as e:
            error_info = {
                'file': str(pdb_file_path),
                'error': str(e),
                'key': Path(pdb_file_path).stem
            }
            self.failed_files.append(error_info)
            
            # 立即记录到文件
            with open(self.failed_files_output, 'a') as f:
                f.write(f"{Path(pdb_file_path).name}\t{str(e)}\n")
            
            return False
    
    def load_batch_with_progress(self, directory, show_progress=True):
        """
        批量加载PDB文件，包含进度条和错误处理
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        # 获取所有PDB文件
        pdb_files = list(directory.glob("*.pdb"))
        
        if not pdb_files:
            raise ValueError(f"在目录 {directory} 中没有找到PDB文件")
        
        # 初始化失败记录文件
        with open(self.failed_files_output, 'w') as f:
            f.write("# PDB文件加载失败记录\n")
            f.write("# 格式: 文件名\t错误信息\n")
            f.write("Filename\tError\n")
        
        print(f"找到 {len(pdb_files)} 个PDB文件，开始加载...")
        print(f"   加载失败记录: {self.failed_files_output}")
        
        # 使用进度条处理文件
        successful = 0
        
        if show_progress:
            pbar = tqdm(pdb_files, desc="🔄 加载PDB文件")
        else:
            pbar = pdb_files
        
        for pdb_file in pbar:
            if self.load_single_pdb_safe(pdb_file):
                successful += 1
                if show_progress:
                    pbar.set_postfix({
                        '成功': successful, 
                        '失败': len(self.failed_files),
                        '当前': pdb_file.name[:15] + '...' if len(pdb_file.name) > 15 else pdb_file.name
                    })
        
        if show_progress:
            pbar.close()
        
        print(f"✅ 加载完成: {successful} 个文件成功, {len(self.failed_files)} 个文件失败")
        
        if successful > 0:
            print(f"📊 有效结构: {len(self.structures)} 个")
        
        return successful > 0
    
    def print_failed_files_summary(self, max_display=20):
        """
        打印失败文件的摘要信息
        """
        if not self.failed_files:
            return
        
        print(f"\n❌ 失败文件摘要 ({len(self.failed_files)} 个文件):")
        print("-" * 80)
        
        for i, error_info in enumerate(self.failed_files[:max_display]):
            print(f"  {i+1}. {Path(error_info['file']).name}")
            print(f"     错误: {error_info['error']}")
            print()
        
        if len(self.failed_files) > max_display:
            print(f"  ... 还有 {len(self.failed_files) - max_display} 个失败文件")
        
        print("-" * 80)


def run_lrr_annotation_with_progress(pdb_directory, output_file="lrr_annotation_results.txt", 
                                   show_progress=True):
    """
    完整的LRR注释流程，包含所有步骤的进度条
    单个结构失败不会停止整个流程，每个步骤单独记录失败
    """
    pdb_directory = Path(pdb_directory)
    
    # 检查输入目录
    if not pdb_directory.exists():
        print(f"❌ 错误: PDB目录 {pdb_directory} 不存在")
        return False
    
    # 定义各步骤的失败记录文件
    failed_loading_file = "failed_loading.txt"
    failed_regression_file = "failed_regression.txt"
    failed_analysis_file = "failed_analysis.txt"
    
    # 初始化健壮的加载器
    L = RobustLoaderWithProgress(failed_loading_file)
    
    print(f"🚀 开始从 {pdb_directory} 加载PDB结构...")
    print(f"📝 失败记录文件:")
    print(f"   加载失败: {failed_loading_file}")
    print(f"   回归失败: {failed_regression_file}")
    print(f"   分析失败: {failed_analysis_file}")
    
    # 步骤1: 加载PDB文件
    try:
        success = L.load_batch_with_progress(pdb_directory, show_progress=show_progress)
        
        if not success:
            print("❌ 错误: 没有成功加载任何PDB文件")
            L.print_failed_files_summary()
            return False
        
        # 显示失败文件摘要
        if L.failed_files:
            L.print_failed_files_summary()
        
        print(f"\n✅ 成功加载 {len(L.structures)} 个结构，开始几何分析...")
        
    except Exception as e:
        print(f"❌ 加载PDB文件时发生严重错误: {e}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return False
    
    # 步骤2: 几何分析
    try:
        # 使用原始的Loader和Analyzer
        original_loader = Loader()
        original_loader.structures = L.structures
        
        A = Analyzer()
        sequence_extractor = LRRSequenceExtractor()
        
        if show_progress:
            print("\n🔬 分析几何形状并计算缠绕数...")
        
        # 计算缠绕数
        A.load_structures(L.structures)
        A.compute_windings(progress=show_progress)
        
        # 步骤3: 回归分析（使用标准Analyzer方法，包含自适应断点检测）
        if show_progress:
            print("\n📈 计算回归分析（使用标准参数配置）...")
        
        # 初始化回归失败记录文件
        with open(failed_regression_file, 'w') as f:
            f.write("# 回归分析失败记录\n")
            f.write("# 格式: 结构ID\t错误信息\n")
            f.write("Structure_ID\tError\n")
        
        print(f"   回归失败记录: {failed_regression_file}")
        
        successful_regressions = 0
        failed_regressions = []
        
        # 使用Analyzer的标准compute_regressions方法，确保参数与标准脚本一致
        # 参数: penalties=[1, 1.5], learning_rate=0.01, iterations=10000, std_cutoff=1
        try:
            # 使用与02_alphafold_to_lrr_annotation.py相同的参数配置
            A.compute_regressions(penalties=[1, 1.5], learning_rate=0.01, 
                                iterations=10000, std_cutoff=1, progress=show_progress)
            
            # 成功处理所有结构
            successful_regressions = len(A.breakpoints)
            
        except Exception as e:
            # 如果批量计算失败，尝试逐个结构处理
            print(f"⚠️ 批量回归分析失败，尝试逐个结构处理: {e}")
            
            # 为每个结构单独计算回归
            if show_progress:
                pbar = tqdm(A.windings.items(), desc="计算回归分析")
            else:
                pbar = A.windings.items()
            
            for key, winding in pbar:
                try:
                    # 使用与标准脚本相同的参数配置
                    from geom_lrr.analyzer import compute_regression, compute_lrr_std
                    
                    # 第一步: 使用2个断点计算回归
                    res = compute_regression(winding, n_breakpoints=2, 
                                           penalties=[1, 1.5], learning_rate=0.01, 
                                           iterations=10000)
                    
                    # 检查标准偏差，如果std > 1则使用4个断点（自适应断点检测）
                    std = compute_lrr_std(winding, res["breakpoints"], res["slope"])
                    
                    if std > 1:  # std_cutoff=1
                        # 使用4个断点重新计算
                        [a, b] = res["breakpoints"]
                        breakpoints = [a, a + (b-a)/2, a + (b-a)/2 + 1, b]
                        res = compute_regression(winding, n_breakpoints=4, 
                                               initial_guess=breakpoints, 
                                               penalties=[1, 1.5], learning_rate=0.01, 
                                               iterations=10000)
                    
                    # 存储结果
                    A.slopes[key] = res['slope']
                    A.breakpoints[key] = res['breakpoints']
                    A.losses[key] = res['loss']
                    A.stds[key] = std
                    successful_regressions += 1
                    
                    if show_progress:
                        pbar.set_postfix({
                            '成功': successful_regressions,
                            '失败': len(failed_regressions),
                            '当前': key[:15] + '...' if len(key) > 15 else key
                        })
                    
                except Exception as e:
                    # 记录失败的结构，但继续处理其他结构
                    failed_regressions.append((key, str(e)))
                    
                    # 立即记录到回归失败文件
                    with open(failed_regression_file, 'a') as f:
                        f.write(f"{key}\t{str(e)}\n")
                    
                    if show_progress:
                        pbar.set_postfix({
                            '成功': successful_regressions,
                            '失败': len(failed_regressions),
                            '当前': f"{key[:10]}...失败" if len(key) > 10 else key + "失败"
                        })
                    
                    # 从分析结果中移除这个结构
                    if key in A.breakpoints:
                        del A.breakpoints[key]
                    if key in A.slopes:
                        del A.slopes[key]
                    if key in A.losses:
                        del A.losses[key]
            
            if show_progress:
                pbar.close()
        
        print(f"✅ 回归分析完成: {successful_regressions} 个结构成功, {len(failed_regressions)} 个结构失败")
        
        # 打印失败的结构信息
        if failed_regressions:
            print(f"\n❌ 失败的结构列表 ({len(failed_regressions)} 个):")
            for i, (key, error) in enumerate(failed_regressions[:10]):
                print(f"  {i+1}. {key}: {error}")
            if len(failed_regressions) > 10:
                print(f"  ... 还有 {len(failed_regressions) - 10} 个失败结构")
        
        if successful_regressions == 0:
            print("❌ 错误: 没有成功的回归分析")
            return False
            
    except Exception as e:
        print(f"❌ 几何分析过程中出错: {e}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return False
    
    # 步骤4: 提取LRR序列
    # 创建输出文件
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化分析失败记录文件
    with open(failed_analysis_file, 'w') as f:
        f.write("# LRR序列分析失败记录\n")
        f.write("# 格式: 结构ID\t错误信息\n")
        f.write("Structure_ID\tError\n")
    
    print(f"\n💾 提取LRR序列并写入到 {output_file}...")
    print(f"   分析失败记录: {failed_analysis_file}")
    
    with open(output_file, 'w') as f:
        # 写入表头
        f.write("PDB_Filename\tRegion_Number\tStart_Position\tEnd_Position\t"
               "Sequence_Length\tFull_Sequence_Length\tTotal_LRR_Regions\tSequence\n")
    
    successful_analyses = 0
    failed_analyses = []
    
    # 使用进度条分析每个结构
    if show_progress:
        pbar = tqdm(A.breakpoints.items(), desc="🔍 分析LRR区域")
    else:
        pbar = A.breakpoints.items()
    
    for pdb_id, breakpoints in pbar:
        # 查找对应的PDB文件
        pdb_files = list(pdb_directory.glob(f"{pdb_id}.pdb"))
        
        if not pdb_files:
            # 尝试其他命名模式
            pdb_files = list(pdb_directory.glob(f"*{pdb_id}*.pdb"))
        
        if not pdb_files:
            error_msg = "未找到对应的PDB文件"
            failed_analyses.append((pdb_id, error_msg))
            
            # 记录到分析失败文件
            with open(failed_analysis_file, 'a') as f:
                f.write(f"{pdb_id}\t{error_msg}\n")
            
            if show_progress:
                pbar.set_postfix({
                    '成功': successful_analyses,
                    '失败': len(failed_analyses),
                    '当前': f"{pdb_id[:10]}...未找到" if len(pdb_id) > 10 else pdb_id + "未找到"
                })
            continue
        
        pdb_file = pdb_files[0]
        pdb_filename = pdb_file.name
        
        try:
            # 分析LRR区域
            results = sequence_extractor.analyze_lrr_regions(str(pdb_file), breakpoints)
            
            # 写入结果
            with open(output_file, 'a') as f:
                for i, (seq, (start, end)) in enumerate(zip(results['lrr_sequences'], 
                                                           results['lrr_positions'])):
                    f.write(f"{pdb_filename}\t{i+1}\t{start}\t{end}\t"
                           f"{len(seq)}\t{results['sequence_length']}\t"
                           f"{results['num_lrr_regions']}\t{seq}\n")
            
            successful_analyses += 1
            
            if show_progress:
                pbar.set_postfix({
                    '成功': successful_analyses,
                    '失败': len(failed_analyses),
                    '当前': f"{pdb_id[:10]}...{results['num_lrr_regions']}区域" if len(pdb_id) > 10 
                           else f"{pdb_id}:{results['num_lrr_regions']}区域"
                })
            
        except Exception as e:
            failed_analyses.append((pdb_id, str(e)))
            
            # 记录到分析失败文件
            with open(failed_analysis_file, 'a') as f:
                f.write(f"{pdb_id}\t{str(e)}\n")
            
            if show_progress:
                pbar.set_postfix({
                    '成功': successful_analyses,
                    '失败': len(failed_analyses),
                    '当前': f"{pdb_id[:10]}...分析失败" if len(pdb_id) > 10 else pdb_id + "分析失败"
                })
    
    if show_progress:
        pbar.close()
    
    # 步骤5: 缓存数据和生成绘图（与标准脚本一致）
    try:
        cache_dir = Path('./LRR_Annotation/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("💾 缓存几何数据和回归结果...")
        
        # 缓存几何数据（与标准脚本一致）
        L.cache(str(cache_dir))
        A.cache_geometry(str(cache_dir))
        A.cache_regressions(str(cache_dir))
        
        # 生成回归分析绘图（与标准脚本一致）
        print("📊 生成回归分析绘图...")
        P = Plotter()
        P.load(A.windings, A.breakpoints, A.slopes)
        plot_dir = Path('./intermediate_files/lrr_annotation_plots')
        plot_dir.mkdir(parents=True, exist_ok=True)
        P.plot_regressions(save=True, directory=str(plot_dir))
        
    except Exception as e:
        print(f"⚠️ 警告: 缓存或绘图失败: {e}")
    
    # 输出最终统计信息
    print(f"\n🎉 LRR注释完成!")
    print(f"📊 最终统计:")
    print(f"✅ 成功分析了 {successful_analyses} 个文件")
    
    # 检查并显示所有失败记录
    failed_files_count = {
        'loading': len(L.failed_files),
        'regression': len(failed_regressions),
        'analysis': len(failed_analyses)
    }
    
    if failed_files_count['loading'] > 0:
        print(f"❌ 加载失败: {failed_files_count['loading']} 个文件 (记录在 {failed_loading_file})")
    
    if failed_files_count['regression'] > 0:
        print(f"❌ 回归失败: {failed_files_count['regression']} 个文件 (记录在 {failed_regression_file})")
    
    if failed_files_count['analysis'] > 0:
        print(f"❌ 分析失败: {failed_files_count['analysis']} 个文件 (记录在 {failed_analysis_file})")
    
    # 显示文件信息
    print(f"\n💾 输出文件:")
    print(f"   LRR结果: {output_file}")
    
    # 显示失败记录文件（如果存在）
    failed_files_exist = []
    for file_name in [failed_loading_file, failed_regression_file, failed_analysis_file]:
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            failed_files_exist.append(file_name)
    
    if failed_files_exist:
        print(f"📝 失败记录:")
        for file_name in failed_files_exist:
            print(f"   {file_name}")
    
    if successful_analyses > 0:
        print(f"\n✨ 分析完成! 请检查上述文件获取详细结果。")
    
    return successful_analyses > 0


class CheckpointManager:
    """
    检查点管理器，用于保存和恢复任务状态
    允许任务在中断后从断点继续运行
    """
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, checkpoint_name, data):
        """保存检查点数据"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        return checkpoint_file
        
    def load_checkpoint(self, checkpoint_name):
        """加载检查点数据"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
        
    def checkpoint_exists(self, checkpoint_name):
        """检查检查点是否存在"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        return checkpoint_file.exists()
        
    def clear_checkpoint(self, checkpoint_name):
        """清除检查点"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False


def run_lrr_annotation_with_checkpoint(pdb_directory, output_file="lrr_annotation_results.txt", 
                                      show_progress=True, checkpoint_dir="./checkpoints",
                                      resume=False):
    """
    支持断点恢复的LRR注释流程
    """
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # 导入tqdm用于进度条
    from tqdm import tqdm
    
    # 定义检查点名称
    checkpoint_names = {
        'loading': 'lrr_annotation_loading',
        'windings': 'lrr_annotation_windings', 
        'regressions': 'lrr_annotation_regressions',
        'analysis': 'lrr_annotation_analysis'
    }
    
    # 步骤1: 加载PDB文件（可恢复）
    if resume and checkpoint_manager.checkpoint_exists(checkpoint_names['loading']):
        print("🔄 从检查点恢复: 加载结构")
        L = checkpoint_manager.load_checkpoint(checkpoint_names['loading'])
        print(f"✅ 已加载 {len(L.structures)} 个结构")
    else:
        print("🚀 开始加载PDB结构...")
        L = RobustLoaderWithProgress("failed_loading.txt")
        success = L.load_batch_with_progress(pdb_directory, show_progress=show_progress)
        
        if not success:
            print("❌ 错误: 没有成功加载任何PDB文件")
            L.print_failed_files_summary()
            return False
        
        # 保存加载检查点
        checkpoint_manager.save_checkpoint(checkpoint_names['loading'], L)
        print(f"✅ 加载完成，保存检查点: {checkpoint_names['loading']}")
    
    # 步骤2: 几何分析（可恢复）
    if resume and checkpoint_manager.checkpoint_exists(checkpoint_names['windings']):
        print("🔄 从检查点恢复: 几何分析")
        A = checkpoint_manager.load_checkpoint(checkpoint_names['windings'])
        print(f"✅ 已计算 {len(A.windings)} 个结构的缠绕数")
    else:
        print("🔬 开始几何分析...")
        
        # 使用原始的Loader和Analyzer
        original_loader = Loader()
        original_loader.structures = L.structures
        
        A = Analyzer()
        
        # 计算缠绕数
        A.load_structures(L.structures)
        A.compute_windings(progress=show_progress)
        
        # 保存几何分析检查点
        checkpoint_manager.save_checkpoint(checkpoint_names['windings'], A)
        print(f"✅ 几何分析完成，保存检查点: {checkpoint_names['windings']}")
    
    # 步骤3: 回归分析（可恢复）
    if resume and checkpoint_manager.checkpoint_exists(checkpoint_names['regressions']):
        print("🔄 从检查点恢复: 回归分析")
        A = checkpoint_manager.load_checkpoint(checkpoint_names['regressions'])
        print(f"✅ 已计算 {len(A.breakpoints)} 个结构的回归分析")
    else:
        print("📈 开始回归分析...")
        
        # 初始化回归失败记录文件
        failed_regression_file = "failed_regression.txt"
        with open(failed_regression_file, 'w') as f:
            f.write("# 回归分析失败记录\n")
            f.write("# 格式: 结构ID\t错误信息\n")
            f.write("Structure_ID\tError\n")
        
        print(f"   回归失败记录: {failed_regression_file}")
        
        successful_regressions = 0
        failed_regressions = []
        
        # 使用与标准脚本相同的参数配置，但使用逐结构处理以确保健壮性
        print("⚠️  使用逐结构处理模式确保健壮性...")
        
        # 为每个结构单独计算回归，确保单个失败不影响整体
        if show_progress:
            pbar = tqdm(A.windings.items(), desc="计算回归分析")
        else:
            pbar = A.windings.items()
        
        for key, winding in pbar:
            try:
                # 使用与标准脚本相同的参数配置
                from geom_lrr.analyzer import compute_regression, compute_lrr_std
                
                # 第一步: 使用2个断点计算回归
                res = compute_regression(winding, n_breakpoints=2, 
                                       penalties=[1, 1.5], learning_rate=0.01, 
                                       iterations=10000)
                
                # 检查标准偏差，如果std > 1则使用4个断点（自适应断点检测）
                std = compute_lrr_std(winding, res["breakpoints"], res["slope"])
                
                if std > 1:  # std_cutoff=1
                    # 使用4个断点重新计算
                    [a, b] = res["breakpoints"]
                    breakpoints = [a, a + (b-a)/2, a + (b-a)/2 + 1, b]
                    res = compute_regression(winding, n_breakpoints=4, 
                                           initial_guess=breakpoints, 
                                           penalties=[1, 1.5], learning_rate=0.01, 
                                           iterations=10000)
                
                # 存储结果
                A.slopes[key] = res['slope']
                A.breakpoints[key] = res['breakpoints']
                A.losses[key] = res['loss']
                A.stds[key] = std
                successful_regressions += 1
                
                if show_progress:
                    pbar.set_postfix({
                        '成功': successful_regressions,
                        '失败': len(failed_regressions),
                        '当前': key[:15] + '...' if len(key) > 15 else key
                    })
                
            except Exception as e:
                # 记录失败的结构，但继续处理其他结构
                error_msg = str(e)
                failed_regressions.append((key, error_msg))
                
                # 立即记录到回归失败文件
                with open(failed_regression_file, 'a') as f:
                    f.write(f"{key}\t{error_msg}\n")
                
                if show_progress:
                    pbar.set_postfix({
                        '成功': successful_regressions,
                        '失败': len(failed_regressions),
                        '当前': f"{key[:10]}...失败" if len(key) > 10 else key + "失败"
                    })
                
                # 从分析结果中移除这个结构
                if key in A.breakpoints:
                    del A.breakpoints[key]
                if key in A.slopes:
                    del A.slopes[key]
                if key in A.losses:
                    del A.losses[key]
                if key in A.stds:
                    del A.stds[key]
        
        if show_progress:
            pbar.close()
        
        print(f"✅ 回归分析完成: {successful_regressions} 个结构成功, {len(failed_regressions)} 个结构失败")
        
        # 打印失败的结构信息
        if failed_regressions:
            print(f"\n❌ 失败的结构列表 ({len(failed_regressions)} 个):")
            for i, (key, error) in enumerate(failed_regressions[:10]):
                print(f"  {i+1}. {key}: {error}")
            if len(failed_regressions) > 10:
                print(f"  ... 还有 {len(failed_regressions) - 10} 个失败结构")
        
        if successful_regressions == 0:
            print("❌ 错误: 没有成功的回归分析")
            return False
        
        # 保存回归分析检查点
        checkpoint_manager.save_checkpoint(checkpoint_names['regressions'], A)
        print(f"💾 回归分析检查点已保存: {checkpoint_names['regressions']}")
    
    # 步骤4: 提取LRR序列（可恢复）
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化序列提取器
    sequence_extractor = LRRSequenceExtractor()
    
    # 检查是否已经有部分结果
    processed_structures = set()
    if resume and output_file.exists():
        # 读取已处理的结枃
        with open(output_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # 跳过表头
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 8:
                        pdb_filename = parts[0]
                        pdb_id = Path(pdb_filename).stem
                        processed_structures.add(pdb_id)
        
        print(f"🔄 已发现 {len(processed_structures)} 个已处理的结构")
    
    # 继续处理未完成的结构
    remaining_structures = {k: v for k, v in A.breakpoints.items() 
                          if k not in processed_structures}
    
    if not remaining_structures:
        print("✅ 所有结构已完成处理")
        successful_analyses = len(A.breakpoints)
    else:
        print(f"📝 继续处理 {len(remaining_structures)} 个未完成的结构")
        
        # 分析失败记录文件
        failed_analysis_file = "failed_analysis.txt"
        with open(failed_analysis_file, 'w') as f:
            f.write("# LRR序列分析失败记录\n")
            f.write("# 格式: 结构ID\t错误信息\n")
            f.write("Structure_ID\tError\n")
        
        successful_analyses = len(processed_structures)
        failed_analyses = []
        
        # 使用进度条分析每个结构
        if show_progress:
            pbar = tqdm(remaining_structures.items(), desc="🔍 分析LRR区域")
        else:
            pbar = remaining_structures.items()
        
        for pdb_id, breakpoints in pbar:
            # 查找对应的PDB文件
            pdb_files = list(pdb_directory.glob(f"{pdb_id}.pdb"))
            
            if not pdb_files:
                # 尝试其他命名模式
                pdb_files = list(pdb_directory.glob(f"*{pdb_id}*.pdb"))
            
            if not pdb_files:
                error_msg = "未找到对应的PDB文件"
                failed_analyses.append((pdb_id, error_msg))
                
                with open(failed_analysis_file, 'a') as f:
                    f.write(f"{pdb_id}\t{error_msg}\n")
                
                if show_progress:
                    pbar.set_postfix({
                        '成功': successful_analyses,
                        '失败': len(failed_analyses),
                        '当前': f"{pdb_id[:10]}...未找到" if len(pdb_id) > 10 else pdb_id + "未找到"
                    })
                continue
            
            pdb_file = pdb_files[0]
            pdb_filename = pdb_file.name
            
            try:
                # 分析LRR区域
                results = sequence_extractor.analyze_lrr_regions(str(pdb_file), breakpoints)
                
                # 写入结果
                with open(output_file, 'a') as f:
                    for i, (seq, (start, end)) in enumerate(zip(results['lrr_sequences'], 
                                                               results['lrr_positions'])):
                        f.write(f"{pdb_filename}\t{i+1}\t{start}\t{end}\t"
                               f"{len(seq)}\t{results['sequence_length']}\t"
                               f"{results['num_lrr_regions']}\t{seq}\n")
                
                successful_analyses += 1
                
                if show_progress:
                    pbar.set_postfix({
                        '成功': successful_analyses,
                        '失败': len(failed_analyses),
                        '当前': f"{pdb_id[:10]}...{results['num_lrr_regions']}区域" if len(pdb_id) > 10 
                               else f"{pdb_id}:{results['num_lrr_regions']}区域"
                    })
                
            except Exception as e:
                failed_analyses.append((pdb_id, str(e)))
                
                with open(failed_analysis_file, 'a') as f:
                    f.write(f"{pdb_id}\t{str(e)}\n")
                
                if show_progress:
                    pbar.set_postfix({
                        '成功': successful_analyses,
                        '失败': len(failed_analyses),
                        '当前': f"{pdb_id[:10]}...分析失败" if len(pdb_id) > 10 else pdb_id + "分析失败"
                    })
        
        if show_progress:
            pbar.close()
    
    # 步骤5: 缓存数据和生成绘图（仅在完整运行时执行）
    if not resume:  # 仅在全新运行时执行缓存和绘图
        try:
            cache_dir = Path('./LRR_Annotation/cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            print("💾 缓存几何数据和回归结果...")
            
            # 缓存几何数据
            L.cache(str(cache_dir))
            A.cache_geometry(str(cache_dir))
            A.cache_regressions(str(cache_dir))
            
            # 生成回归分析绘图
            print("📊 生成回归分析绘图...")
            P = Plotter()
            P.load(A.windings, A.breakpoints, A.slopes)
            plot_dir = Path('./intermediate_files/lrr_annotation_plots')
            plot_dir.mkdir(parents=True, exist_ok=True)
            P.plot_regressions(save=True, directory=str(plot_dir))
            
        except Exception as e:
            print(f"⚠️ 警告: 缓存或绘图失败: {e}")
    
    # 输出最终统计信息
    print(f"\n🎉 LRR注释完成!")
    print(f"📊 最终统计:")
    print(f"✅ 成功分析了 {successful_analyses} 个文件")
    
    # 清理检查点（任务完成）
    if not resume:
        for checkpoint_name in checkpoint_names.values():
            checkpoint_manager.clear_checkpoint(checkpoint_name)
        print("🧹 任务完成，清理检查点文件")
    
    return successful_analyses > 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="健壮的PDB到LRR注释生成器（包含断点恢复功能）"
    )
    
    parser.add_argument("input", 
                       help="输入PDB文件或包含PDB文件的目录")
    
    parser.add_argument("-o", "--output", 
                       default="lrr_annotation_results.txt",
                       help="输出文件路径（默认: lrr_annotation_results.txt）")
    
    parser.add_argument("--no-progress", 
                       action="store_false", dest="show_progress",
                       help="不显示进度条")
    
    parser.add_argument("--max-errors", 
                       type=int, default=20,
                       help="最多显示的错误数量（默认: 20）")
    
    parser.add_argument("--resume", 
                       action="store_true",
                       help="从断点继续运行（如果存在检查点）")
    
    parser.add_argument("--checkpoint-dir", 
                       default="./checkpoints",
                       help="检查点文件目录（默认: ./checkpoints）")
    
    parser.add_argument("--clear-checkpoints", 
                       action="store_true",
                       help="清除所有检查点文件并重新开始")
    
    args = parser.parse_args()
    
    # 显示断点恢复功能帮助信息
    if args.resume:
        print("\n🔄 断点恢复模式已启用")
        print("   检查点目录:", args.checkpoint_dir)
        print("   如果存在检查点，将从上次中断的地方继续运行")
        print("   使用 --clear-checkpoints 清除所有检查点并重新开始\n")
    
    # 清理检查点（如果需要）
    if args.clear_checkpoints:
        checkpoint_manager = CheckpointManager(args.checkpoint_dir)
        checkpoint_files = list(Path(args.checkpoint_dir).glob("*.pkl"))
        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()
        print(f"🧹 已清理 {len(checkpoint_files)} 个检查点文件")
    
    input_path = Path(args.input)
    
    # 处理单个文件或目录
    if input_path.is_file() and input_path.suffix.lower() == '.pdb':
        # 创建临时目录
        import shutil
        temp_dir = Path("./temp_pdb_dir_progress")
        temp_dir.mkdir(exist_ok=True)
        
        temp_pdb = temp_dir / input_path.name
        shutil.copy2(input_path, temp_pdb)
        
        print(f"📁 处理单个PDB文件: {input_path.name}")
        
        success = run_lrr_annotation_with_checkpoint(
            temp_dir, 
            args.output, 
            show_progress=args.show_progress,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume
        )
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
    elif input_path.is_dir():
        print(f"📁 处理目录中的PDB文件: {input_path}")
        
        success = run_lrr_annotation_with_checkpoint(
            input_path, 
            args.output, 
            show_progress=args.show_progress,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume
        )
        
    else:
        print(f"❌ 错误: 输入路径 {args.input} 必须是PDB文件或包含PDB文件的目录")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())