"""
Module: dse.py
Part: D
Purpose: Design Space Exploration (DSE) for accelerator architectures.

Learning Goals:
- Define a multi-dimensional design space for architecture evaluation
- Systematically sweep configurations and collect results
- Identify performance bottlenecks (memory-bound vs compute-bound)
- Use the roofline model to interpret results

Prerequisites:
- Step 4.2: scalesim_runner.py implemented
- Understanding of roofline model (from paper reading)

什么是 DSE（Design Space Exploration）？
- 给定一组可调的架构参数（PE array 大小、SRAM 容量、数据流）
- 对每种配置运行模拟，收集性能指标
- 分析趋势，找到最优或"甜点"配置
- 解释为什么某个配置更好（瓶颈分析）

这是面试中最常被问到的问题之一：
"你如何选择架构配置？基于什么证据？"

Roofline Model 基础：
- 把计算性能表示为"算术强度"（ops/byte）的函数
- 算术强度 = 计算量 / 数据搬运量
- 如果算术强度低（< 转折点）：memory-bound（瓶颈在带宽）
- 如果算术强度高（> 转折点）：compute-bound（瓶颈在计算）
- 转折点 = peak_compute / peak_bandwidth

Usage:
    python -m src.arch.dse
"""

import os
import sys
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def define_design_space():
    """
    Define the architecture design space to explore.

    Returns:
        design_space: dict with parameter names and their candidate values
    """
    # =========================================================================
    # TODO [Step 4.3]: Define the design space
    #
    # Suggested parameters and ranges:
    #
    # design_space = {
    #     'array_size': [(8, 8), (16, 16), (32, 32)],
    #     'ifmap_sram_kb': [32, 64, 128, 256],
    #     'filter_sram_kb': [32, 64, 128, 256],
    #     'ofmap_sram_kb': [16, 32, 64],
    #     'dataflow': ['os', 'ws', 'is'],
    # }
    #
    # 参数选择说明：
    # - array_size: PE 阵列大小。更大 = 更多并行 = 更快，但面积也更大
    #   8×8 是小型边缘设备，16×16 是中等，32×32 是较大的加速器
    #
    # - SRAM sizes: 片上缓存大小。影响数据复用策略和 DRAM 访问次数
    #   总 SRAM 通常在 64KB ~ 1MB 范围
    #
    # - dataflow: 数据流模式。不同 workload 适合不同数据流
    #   WS 通常对 depthwise conv 不友好
    #   OS 通常对小 output feature map 不友好
    #
    # 注意：完整扫描 3×4×4×3×3 = 432 个配置，可能比较慢。
    # 可以先用简化的空间（如固定 SRAM，只扫 array_size × dataflow）
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.3]: Implement define_design_space")


def generate_all_configs(design_space):
    """
    Generate all configuration combinations from the design space.

    Args:
        design_space: dict from define_design_space()

    Returns:
        configs: list of dicts, each representing one configuration
    """
    # =========================================================================
    # TODO [Step 4.3]: Generate configuration combinations
    #
    # Use itertools.product to generate all combinations:
    #
    # keys = list(design_space.keys())
    # values = list(design_space.values())
    # configs = []
    # for combo in itertools.product(*values):
    #     config = dict(zip(keys, combo))
    #     configs.append(config)
    #
    # print(f"Total configurations: {len(configs)}")
    # return configs
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.3]: Implement generate_all_configs")


def run_dse_sweep(model, design_space, output_dir):
    """
    Run the full DSE sweep: generate configs, run simulations, collect results.

    Args:
        model: PyTorch model (for topology extraction)
        design_space: Design space dict
        output_dir: Output directory for results

    Returns:
        results: list of dicts with config + simulation results
    """
    # =========================================================================
    # TODO [Step 4.3]: Implement DSE sweep
    #
    # Steps:
    #   1. Generate topology CSV from model
    #   2. Generate all configs
    #   3. For each config:
    #      a. Generate SCALE-Sim config file
    #      b. Run simulation
    #      c. Parse results
    #      d. Append to results list
    #   4. Save all results to CSV
    #
    # from src.arch.scalesim_runner import (
    #     generate_topology_csv,
    #     generate_scalesim_config,
    #     run_scalesim,
    #     parse_compute_report,
    # )
    #
    # topology_path = os.path.join(output_dir, 'topology.csv')
    # generate_topology_csv(model, topology_path)
    #
    # results = []
    # configs = generate_all_configs(design_space)
    # for i, config in enumerate(configs):
    #     print(f"Running config {i+1}/{len(configs)}: {config}")
    #     run_name = f"dse_{i:03d}"
    #     config_path = generate_scalesim_config(
    #         run_name, config['array_size'][0], config['array_size'][1],
    #         config['ifmap_sram_kb'], config['filter_sram_kb'],
    #         config['ofmap_sram_kb'], config['dataflow'], output_dir)
    #     run_scalesim(config_path, topology_path, output_dir)
    #     report = parse_compute_report(os.path.join(output_dir, run_name, ...))
    #     results.append({**config, **report})
    #
    # return results
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.3]: Implement run_dse_sweep")


def analyze_dse_results(results):
    """
    Analyze DSE results: identify trends, bottlenecks, and optimal configs.

    Args:
        results: List of result dicts from run_dse_sweep

    Returns:
        analysis: dict with summary statistics and recommendations
    """
    # =========================================================================
    # TODO [Step 4.3]: Implement DSE result analysis
    #
    # Analysis to perform:
    #
    # 1. Find best/worst configurations (by total cycles)
    # 2. Compute speedup relative to smallest config
    # 3. Classify each config as memory-bound or compute-bound:
    #    - If stall_cycles > compute_cycles: memory-bound
    #    - If compute_cycles > stall_cycles: compute-bound
    # 4. Identify "diminishing returns" points:
    #    - At what array size does doubling stop helping?
    #    - At what SRAM size does the config switch from memory to compute bound?
    # 5. Generate summary table
    #
    # 关键输出：
    # - "最佳配置是 16×16 array + 128KB SRAM + WS dataflow"
    # - "原因：在这个配置下，PE 利用率 > 80%，且 SRAM 足够避免大量 DRAM 访问"
    # - "如果 array 增大到 32×32，利用率下降到 40%，因为 workload 太小了"
    #
    # 这就是面试时要讲的 "故事"：
    # 不只是展示数字，而是解释为什么。
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.3]: Implement analyze_dse_results")


def identify_bottleneck(total_cycles, compute_cycles, stall_cycles):
    """
    Classify a configuration as memory-bound or compute-bound.

    Args:
        total_cycles: Total simulation cycles
        compute_cycles: Cycles spent doing useful computation
        stall_cycles: Cycles spent stalling (waiting for data)

    Returns:
        'memory-bound' or 'compute-bound'
    """
    # =========================================================================
    # TODO [Step 4.3]: Implement bottleneck identification
    #
    # if stall_cycles > compute_cycles:
    #     return 'memory-bound'
    # else:
    #     return 'compute-bound'
    #
    # Memory-bound 的含义：
    #   PE 在等数据，计算能力没有被充分利用
    #   解决方法：增加 SRAM、优化 tiling、改善数据复用
    #
    # Compute-bound 的含义：
    #   数据供应足够，PE 在全力计算
    #   解决方法：增加 PE 数量、提高时钟频率
    #
    # 理想状态：两者平衡（roofline 的转折点）
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.3]: Implement identify_bottleneck")


if __name__ == "__main__":
    print("=" * 60)
    print("  Design Space Exploration (DSE)")
    print("=" * 60)
    print("\nThis module requires:")
    print("  1. SCALE-Sim installed (pip install scalesim)")
    print("  2. A trained model (Part A complete)")
    print("  3. scalesim_runner.py implemented (Step 4.2)")
    print("\nRun via: python scripts/run_arch_eval.py")
