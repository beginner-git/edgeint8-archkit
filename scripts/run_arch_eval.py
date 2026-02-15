"""
Module: run_arch_eval.py
Part: D
Purpose: Top-level script for architecture evaluation using SCALE-Sim.

Learning Goals:
- Run Design Space Exploration (DSE) with SCALE-Sim
- Extract Conv layer parameters from a PyTorch model
- Understand dataflow, tiling, and memory hierarchy trade-offs

Prerequisites:
- pip install scalesim
- Trained and exported models (Part A)

什么是 DSE（Design Space Exploration）？
- 给定一个 workload（你的模型的 Conv 层参数）
- 扫描不同的硬件配置（PE array 大小、SRAM 容量、数据流）
- 对比每种配置的周期数、利用率、带宽需求
- 找到最优或"足够好"的配置，并解释为什么

这回答了面试中的核心问题：
"你为什么选这个架构配置？瓶颈在哪里？怎么改？"

Usage:
    python scripts/run_arch_eval.py --workload 2d
    python scripts/run_arch_eval.py --workload all
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import set_seed, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Architecture Evaluation")
    parser.add_argument("--workload", choices=["1d", "2d", "all"], default="2d")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for DSE results")
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.output_dir)

    # =========================================================================
    # TODO [Step 4.4]: Implement architecture evaluation pipeline
    #
    # Steps:
    #   1. Load model architecture (just need layer params, not weights)
    #      from models.tiny_cnn_2d import get_tiny_cnn_2d
    #      model = get_tiny_cnn_2d()
    #
    #   2. Generate SCALE-Sim topology CSV:
    #      from src.arch.scalesim_runner import generate_topology_csv
    #      generate_topology_csv(model, "results/topology_2d.csv")
    #
    #   3. Run DSE sweep:
    #      from src.arch.dse import define_design_space, run_dse_sweep
    #      design_space = define_design_space()
    #      results = run_dse_sweep(model, design_space, args.output_dir)
    #
    #   4. Analyze results:
    #      from src.arch.dse import analyze_dse_results
    #      summary = analyze_dse_results(results)
    #
    #   5. Save results and print summary table
    #
    # 建议先用 Workload-2 (2D CNN) 做 DSE，因为 Conv2d 层更典型，
    # SCALE-Sim 对 2D 卷积的支持也更成熟。
    # =========================================================================
    print("TODO: Implement architecture evaluation pipeline")
    print("Make sure SCALE-Sim is installed: pip install scalesim")


if __name__ == "__main__":
    main()
