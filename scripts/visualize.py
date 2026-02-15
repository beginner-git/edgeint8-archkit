"""
Module: visualize.py
Part: B + D
Purpose: Generate comparison charts and figures for the project report.

Learning Goals:
- Create publication-quality figures with matplotlib
- Visualize quantization accuracy/latency trade-offs
- Plot architecture DSE trends

Prerequisites:
- Part B: Benchmark results from scripts/bench.py
- Part D: DSE results from scripts/run_arch_eval.py

Usage:
    python scripts/visualize.py --type quantization --data results/tables/benchmark_comparison.csv
    python scripts/visualize.py --type dse --data results/tables/dse_results.csv
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import ensure_dir


# =============================================================================
# Part B: Quantization comparison charts
# =============================================================================

def plot_accuracy_comparison(results_csv, output_path="results/figures/accuracy_comparison.png"):
    """
    Bar chart comparing accuracy across quantization strategies.

    Args:
        results_csv: Path to benchmark comparison CSV
        output_path: Path to save the figure
    """
    # =========================================================================
    # TODO [Step 2.6]: Implement accuracy comparison bar chart
    #
    # Steps:
    #   import pandas as pd
    #   import matplotlib.pyplot as plt
    #
    #   df = pd.read_csv(results_csv)
    #   fig, ax = plt.subplots(figsize=(10, 6))
    #   ax.bar(df['model'], df['accuracy'])
    #   ax.set_ylabel('Accuracy')
    #   ax.set_title('Quantization Strategy Comparison: Accuracy')
    #   ax.axhline(y=fp32_accuracy, color='r', linestyle='--', label='FP32 baseline')
    #   plt.xticks(rotation=45, ha='right')
    #   plt.tight_layout()
    #   plt.savefig(output_path, dpi=150)
    #
    # 图表设计建议：
    # - 用红色虚线标出 FP32 baseline
    # - 按精度从高到低排序
    # - 标注精度下降最大的配置
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.6]: Implement accuracy comparison chart")


def plot_latency_comparison(results_csv, output_path="results/figures/latency_comparison.png"):
    """Bar chart comparing latency across models."""
    # =========================================================================
    # TODO [Step 2.6]: Implement latency comparison bar chart
    #
    # Similar to accuracy chart but with latency on y-axis.
    # Include error bars using std.
    # Color-code: FP32 in blue, INT8 variants in green shades.
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.6]: Implement latency comparison chart")


def plot_accuracy_vs_latency(results_csv, output_path="results/figures/acc_vs_latency.png"):
    """Scatter plot: accuracy vs latency for all models."""
    # =========================================================================
    # TODO [Step 2.6]: Implement accuracy-latency scatter plot
    #
    # This is the most informative chart for quantization analysis:
    # x-axis: latency (ms), y-axis: accuracy
    # Each point is labeled with the model/strategy name
    # Ideal: top-left corner (high accuracy, low latency)
    #
    # 这张图直观展示了精度-延迟的 trade-off，面试时非常有说服力。
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.6]: Implement accuracy-latency scatter")


def plot_error_distribution(error_data, output_path="results/figures/error_distribution.png"):
    """Histogram of quantization errors per layer."""
    # =========================================================================
    # TODO [Step 2.6]: Implement per-layer error distribution histogram
    #
    # Input: error_data is a dict of {layer_name: error_array}
    # Create subplots, one per layer, showing error distribution
    # Mark: mean error, max error, std
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.6]: Implement error distribution plot")


# =============================================================================
# Part D: Architecture evaluation charts
# =============================================================================

def plot_dse_cycles(dse_csv, output_path="results/figures/dse_cycles.png"):
    """Plot cycle count vs PE array size for different dataflows."""
    # =========================================================================
    # TODO [Step 4.5]: Implement DSE cycle trend plot
    #
    # x-axis: PE array size (e.g., "8x8", "16x16", "32x32")
    # y-axis: Total cycles
    # Lines: one per dataflow (WS, OS, IS)
    #
    # 这张图回答："PE array 越大越好吗？什么时候收益递减？"
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.5]: Implement DSE cycle plot")


def plot_dse_utilization(dse_csv, output_path="results/figures/dse_utilization.png"):
    """Plot PE utilization vs SRAM size."""
    # =========================================================================
    # TODO [Step 4.5]: Implement DSE utilization plot
    #
    # x-axis: SRAM size (KB)
    # y-axis: PE utilization (%)
    # Annotate memory-bound vs compute-bound regions
    #
    # 这张图回答："SRAM 要多大才够？更大的 SRAM 收益是什么？"
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.5]: Implement DSE utilization plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualization charts")
    parser.add_argument("--type", choices=["quantization", "dse", "all"],
                        default="all", help="Type of charts to generate")
    parser.add_argument("--data", type=str, help="Path to data CSV")
    args = parser.parse_args()

    ensure_dir("results/figures")

    # =========================================================================
    # TODO: Call appropriate plotting functions based on --type argument
    # =========================================================================
    print("TODO: Implement visualization main script")
