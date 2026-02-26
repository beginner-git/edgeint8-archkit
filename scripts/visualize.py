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
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df['accuracy'] = df['accuracy'].astype(float)
    df = df.sort_values('accuracy', ascending=False)

    # Identify FP32 baseline
    fp32_row = df[~df['model'].str.contains('int8', case=False)]
    fp32_accuracy = float(fp32_row['accuracy'].iloc[0]) if len(fp32_row) > 0 else None

    # Color: blue for FP32, green for INT8
    colors = ['#2196F3' if 'int8' not in m.lower() else '#4CAF50' for m in df['model']]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(df)), df['accuracy'], color=colors)

    # FP32 baseline line
    if fp32_accuracy is not None:
        ax.axhline(y=fp32_accuracy, color='r', linestyle='--', linewidth=1.5, label=f'FP32 baseline ({fp32_accuracy:.4f})')
        ax.legend(fontsize=10)

    # Label the bar with lowest accuracy
    min_idx = df['accuracy'].idxmin()
    min_pos = list(df.index).index(min_idx)
    bars[min_pos].set_color('#F44336')

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Quantization Strategy Comparison: Accuracy', fontsize=14)

    ensure_dir(os.path.dirname(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_comparison(results_csv, output_path="results/figures/latency_comparison.png"):
    """Bar chart comparing latency across models."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df['latency_mean_ms'] = df['latency_mean_ms'].astype(float)
    df['latency_std_ms'] = df['latency_std_ms'].astype(float)
    df = df.sort_values('latency_mean_ms', ascending=True)

    # Color: blue for FP32, green for INT8
    colors = ['#2196F3' if 'int8' not in m.lower() else '#4CAF50' for m in df['model']]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(df)), df['latency_mean_ms'], yerr=df['latency_std_ms'],
           color=colors, capsize=4, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Quantization Strategy Comparison: Latency', fontsize=14)

    ensure_dir(os.path.dirname(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_latency(results_csv, output_path="results/figures/acc_vs_latency.png"):
    """Scatter plot: accuracy vs latency for all models."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df['latency_mean_ms'] = df['latency_mean_ms'].astype(float)
    df['accuracy'] = df['accuracy'].astype(float)

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        is_fp32 = 'int8' not in row['model'].lower()
        color = '#2196F3' if is_fp32 else '#4CAF50'
        marker = 's' if is_fp32 else 'o'
        size = 120 if is_fp32 else 80
        ax.scatter(row['latency_mean_ms'], row['accuracy'],
                   c=color, marker=marker, s=size, edgecolors='black', linewidth=0.5, zorder=3)
        # Label each point
        label = row['model'].replace('.onnx', '').replace('int8_', '')
        ax.annotate(label, (row['latency_mean_ms'], row['accuracy']),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

    # Mark ideal region (top-left)
    ax.annotate('← Ideal', xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
                fontsize=10, color='gray', style='italic',
                xytext=(10, -15), textcoords='offset points')

    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Latency Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)

    ensure_dir(os.path.dirname(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_distribution(error_data, output_path="results/figures/error_distribution.png"):
    """Histogram of quantization errors per layer."""
    import matplotlib.pyplot as plt
    import numpy as np

    num_layers = len(error_data)
    if num_layers == 0:
        print("No error data to plot.")
        return

    cols = min(3, num_layers)
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (layer_name, errors) in enumerate(error_data.items()):
        ax = axes[idx]
        errors = np.array(errors)
        ax.hist(errors, bins=50, color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.8)

        # Mark statistics
        mean_err = np.mean(errors)
        max_err = np.max(np.abs(errors))
        std_err = np.std(errors)
        ax.axvline(mean_err, color='r', linestyle='--', linewidth=1.5, label=f'mean={mean_err:.4f}')
        ax.axvline(mean_err + std_err, color='orange', linestyle=':', linewidth=1, label=f'std={std_err:.4f}')
        ax.axvline(mean_err - std_err, color='orange', linestyle=':', linewidth=1)

        ax.set_title(layer_name, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel('Quantization Error')
        ax.set_ylabel('Count')

    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Per-Layer Quantization Error Distribution', fontsize=14)
    ensure_dir(os.path.dirname(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


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

    default_csv = "results/tables/benchmark_comparison.csv"
    data_path = args.data if args.data else default_csv

    if args.type in ("quantization", "all"):
        if os.path.exists(data_path):
            print(f"Generating Part B charts from: {data_path}")
            plot_accuracy_comparison(data_path)
            plot_latency_comparison(data_path)
            plot_accuracy_vs_latency(data_path)
        else:
            print(f"CSV not found: {data_path}")
            print("Run 'python scripts/bench.py --all --workload 2d' first.")

    if args.type in ("dse", "all"):
        dse_path = args.data if args.data else "results/tables/dse_results.csv"
        if os.path.exists(dse_path):
            print(f"Generating Part D charts from: {dse_path}")
            plot_dse_cycles(dse_path)
            plot_dse_utilization(dse_path)
        else:
            print(f"DSE CSV not found: {dse_path}")
            print("Run 'python scripts/run_arch_eval.py' first (Part D).")
