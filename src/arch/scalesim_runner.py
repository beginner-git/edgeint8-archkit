"""
Module: scalesim_runner.py
Part: D
Purpose: SCALE-Sim configuration, execution, and result parsing.

Learning Goals:
- Understand what SCALE-Sim simulates (systolic array cycle-level behavior)
- Learn to extract Conv layer parameters from a PyTorch model
- Generate SCALE-Sim config and topology files
- Parse and interpret simulation results

Prerequisites:
- pip install scalesim
- Understanding of systolic arrays and dataflow concepts (from paper reading)

什么是 SCALE-Sim？
- Systolic CNN Accelerator Simulator
- 模拟一个 systolic array（脉动阵列）执行 CNN 卷积层的行为
- 输出：周期数（cycles）、带宽需求（bandwidth）、PE 利用率（utilization）
- 是学术界广泛使用的架构评估工具

Systolic Array 基础：
- PE (Processing Element) 阵列：一个 M×N 的计算单元网格
- 每个 PE 做一次乘加（MAC）
- 数据在 PE 之间流动，实现数据复用
- 不同的数据流（dataflow）决定了哪种数据在哪个方向流动：
  - WS (Weight Stationary): 权重固定在 PE 中，输入和输出流动
  - OS (Output Stationary): 输出固定在 PE 中，权重和输入流动
  - IS (Input Stationary):  输入固定在 PE 中，权重和输出流动

Usage:
    python -m src.arch.scalesim_runner
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def generate_topology_csv(model, output_path):
    """
    Extract Conv layer parameters from a PyTorch model and write
    SCALE-Sim topology CSV format.

    SCALE-Sim topology CSV format:
        Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width,
        Channels, Num Filter, Strides

    Args:
        model: PyTorch model
        output_path: Path to save the topology CSV

    Returns:
        output_path
    """
    # =========================================================================
    # TODO [Step 4.2]: Implement topology extraction
    #
    # import torch.nn as nn
    #
    # layers = []
    # # We need to trace input sizes through the model
    # # Use a forward hook to capture input shapes
    #
    # hooks = []
    # layer_info = []
    #
    # def make_hook(name, module):
    #     def hook_fn(mod, inp, out):
    #         if isinstance(mod, nn.Conv2d):
    #             _, _, h, w = inp[0].shape
    #             layer_info.append({
    #                 'name': name,
    #                 'ifmap_h': h,
    #                 'ifmap_w': w,
    #                 'filter_h': mod.kernel_size[0],
    #                 'filter_w': mod.kernel_size[1],
    #                 'channels': mod.in_channels,
    #                 'num_filter': mod.out_channels,
    #                 'stride': mod.stride[0],
    #             })
    #     return hook_fn
    #
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         hooks.append(module.register_forward_hook(make_hook(name, module)))
    #
    # # Run dummy forward pass to trigger hooks
    # import torch
    # model.eval()
    # with torch.no_grad():
    #     model(torch.randn(1, 3, 32, 32))  # CIFAR-10 input
    #
    # for h in hooks:
    #     h.remove()
    #
    # # Write CSV
    # with open(output_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Layer name', 'IFMAP Height', 'IFMAP Width',
    #                      'Filter Height', 'Filter Width', 'Channels',
    #                      'Num Filter', 'Strides'])
    #     for info in layer_info:
    #         writer.writerow([info['name'], info['ifmap_h'], info['ifmap_w'],
    #                          info['filter_h'], info['filter_w'],
    #                          info['channels'], info['num_filter'],
    #                          info['stride']])
    #
    # print(f"Topology saved to {output_path} ({len(layer_info)} layers)")
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.2]: Implement generate_topology_csv")


def generate_scalesim_config(run_name, array_height, array_width,
                             ifmap_sram_kb, filter_sram_kb, ofmap_sram_kb,
                             dataflow, output_dir):
    """
    Generate a SCALE-Sim configuration file (.cfg).

    Args:
        run_name: Name for this simulation run
        array_height: Number of PE rows
        array_width: Number of PE columns
        ifmap_sram_kb: Input feature map SRAM size in KB
        filter_sram_kb: Filter/weight SRAM size in KB
        ofmap_sram_kb: Output feature map SRAM size in KB
        dataflow: "os" (output stationary), "ws" (weight stationary),
                  or "is" (input stationary)
        output_dir: Directory for output files

    Returns:
        config_path: Path to generated config file
    """
    # =========================================================================
    # TODO [Step 4.2]: Implement SCALE-Sim config generation
    #
    # SCALE-Sim config format (INI-style):
    #
    # config_content = f"""[general]
    # run_name = {run_name}
    #
    # [architecture_presets]
    # ArrayHeight:    {array_height}
    # ArrayWidth:     {array_width}
    # IfmapSramSzkB:  {ifmap_sram_kb}
    # FilterSramSzkB: {filter_sram_kb}
    # OfmapSramSzkB:  {ofmap_sram_kb}
    # Dataflow:       {dataflow}
    # """
    #
    # config_path = os.path.join(output_dir, f"{run_name}.cfg")
    # with open(config_path, 'w') as f:
    #     f.write(config_content)
    #
    # Dataflow 选项说明：
    # - "os": Output Stationary — 每个 PE 计算一个输出元素，适合大 output
    # - "ws": Weight Stationary — 权重固定在 PE 中，适合权重复用多的情况
    # - "is": Input Stationary  — 输入固定在 PE 中，适合输入复用多的情况
    #
    # SRAM 大小的影响：
    # - 更大的 SRAM = 更多的片上数据复用 = 更少的 DRAM 访问 = 更低的能耗
    # - 但 SRAM 面积和功耗也会增加
    # - 这就是 DSE 要探索的 trade-off
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.2]: Implement generate_scalesim_config")


def run_scalesim(config_path, topology_path, output_dir):
    """
    Run SCALE-Sim simulation.

    Args:
        config_path: Path to config file
        topology_path: Path to topology CSV
        output_dir: Directory for results

    Returns:
        results_dir: Path to simulation results
    """
    # =========================================================================
    # TODO [Step 4.2]: Implement SCALE-Sim execution
    #
    # Option 1: Programmatic API
    #   from scalesim.scale_sim import scalesim
    #   s = scalesim(save_disk_space=True, verbose=True,
    #                config=config_path, topology=topology_path)
    #   s.run_scale(top_path=topology_path)
    #
    # Option 2: Command line (fallback)
    #   import subprocess
    #   subprocess.run(['python', '-m', 'scalesim.scale_sim',
    #                   '-c', config_path, '-t', topology_path])
    #
    # SCALE-Sim 会在 output_dir 下生成：
    # - COMPUTE_REPORT.csv: 每层的周期数、利用率
    # - BANDWIDTH_REPORT.csv: 每层的带宽需求
    # - 其他详细报告
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.2]: Implement run_scalesim")


def parse_compute_report(report_path):
    """
    Parse SCALE-Sim COMPUTE_REPORT.csv.

    Returns:
        list of dicts with per-layer cycle counts and utilization
    """
    # =========================================================================
    # TODO [Step 4.2]: Implement result parsing
    #
    # Read the CSV and extract:
    # - Layer name
    # - Total cycles
    # - Compute cycles (useful work)
    # - Stall cycles (waiting for data)
    # - PE utilization = compute_cycles / total_cycles
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.2]: Implement parse_compute_report")


def parse_bandwidth_report(report_path):
    """Parse SCALE-Sim BANDWIDTH_REPORT.csv."""
    # =========================================================================
    # TODO [Step 4.2]: Parse bandwidth data
    #
    # Extract: ifmap reads, filter reads, ofmap writes (in bytes or elements)
    # =========================================================================
    raise NotImplementedError("TODO [Step 4.2]: Implement parse_bandwidth_report")


if __name__ == "__main__":
    print("SCALE-Sim Runner")
    print("=" * 40)
    print("Prerequisites: pip install scalesim")
    print("\nSteps to use:")
    print("1. Implement generate_topology_csv() with your model")
    print("2. Implement generate_scalesim_config() with desired arch params")
    print("3. Run run_scalesim() and parse results")
