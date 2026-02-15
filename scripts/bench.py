"""
Module: bench.py
Part: A + B (unified benchmarking)
Purpose: Benchmark ONNX models for latency, accuracy, and model size.

Learning Goals:
- Use ONNX Runtime InferenceSession for CPU inference
- Measure latency with proper warmup and statistical reporting
- Compare FP32 vs INT8 quantized models systematically

Prerequisites:
- ONNX models exported via scripts/export_onnx.py
- For INT8 comparison: quantized models from scripts/quantize_ptq.py

这是整个项目最常用的脚本——用它来回答 "FP32 和 INT8 差多少？" 的问题。
每次做完量化或修改模型后，都应该重新跑一次 bench.py 来更新结果。

Usage:
    python scripts/bench.py --model models/tiny_cnn_2d.onnx --workload 2d
    python scripts/bench.py --all --workload 2d
    python scripts/bench.py --compare models/fp32.onnx models/int8.onnx --workload 2d
"""

import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import set_seed, ensure_dir


def measure_latency(onnx_path, dummy_input_np, num_warmup=10, num_runs=100):
    """
    Measure inference latency of an ONNX model.

    Args:
        onnx_path: Path to ONNX model file
        dummy_input_np: Numpy array with correct input shape
        num_warmup: Number of warmup runs (not measured)
        num_runs: Number of measured runs

    Returns:
        dict with keys: 'mean_ms', 'std_ms', 'p50_ms', 'p99_ms', 'min_ms'
    """
    # =========================================================================
    # TODO [Step 1.6]: Implement latency measurement
    #
    # Steps:
    #   1. Create ORT session:
    #      import onnxruntime as ort
    #      session = ort.InferenceSession(onnx_path,
    #          providers=['CPUExecutionProvider'])
    #      input_name = session.get_inputs()[0].name
    #
    #   2. Warmup (critical for accurate measurement!):
    #      for _ in range(num_warmup):
    #          session.run(None, {input_name: dummy_input_np})
    #
    #   3. Measure:
    #      latencies = []
    #      for _ in range(num_runs):
    #          start = time.perf_counter()
    #          session.run(None, {input_name: dummy_input_np})
    #          latencies.append((time.perf_counter() - start) * 1000)  # ms
    #
    #   4. Compute stats:
    #      return {
    #          'mean_ms': np.mean(latencies),
    #          'std_ms':  np.std(latencies),
    #          'p50_ms':  np.percentile(latencies, 50),
    #          'p99_ms':  np.percentile(latencies, 99),
    #          'min_ms':  np.min(latencies),
    #      }
    #
    # 为什么需要 warmup？
    # - 首次运行时，ORT 会做图优化、内存分配、JIT 编译等
    # - 这些一次性开销不应该计入稳态延迟
    # - 通常 10 次 warmup 就足够了
    #
    # 为什么用 time.perf_counter()？
    # - 比 time.time() 精度更高（纳秒级）
    # - 适合测量短时间间隔
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.6]: Implement latency measurement")


def measure_accuracy(onnx_path, test_loader):
    """
    Measure classification accuracy of an ONNX model.

    Args:
        onnx_path: Path to ONNX model file
        test_loader: PyTorch DataLoader for test set

    Returns:
        accuracy: float (0.0 ~ 1.0)
    """
    # =========================================================================
    # TODO [Step 1.6]: Implement accuracy measurement
    #
    # Steps:
    #   1. Create ORT session
    #   2. For each batch in test_loader:
    #      a. Convert batch to numpy: data_np = data.numpy()
    #      b. Run inference: output = session.run(None, {input_name: data_np})
    #      c. Get predictions: pred = np.argmax(output[0], axis=1)
    #      d. Count correct: correct += (pred == target.numpy()).sum()
    #   3. Return accuracy = correct / total
    #
    # Note: ORT 接受 numpy array 作为输入，不是 torch tensor！
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.6]: Implement accuracy measurement")


def measure_model_size(onnx_path):
    """Get ONNX model file size in MB."""
    return os.path.getsize(onnx_path) / (1024 * 1024)


def benchmark_single(onnx_path, test_loader, dummy_input_np):
    """
    Run all benchmarks on a single model.

    Returns:
        dict with: 'accuracy', 'latency' (sub-dict), 'size_mb', 'model_path'
    """
    # =========================================================================
    # TODO [Step 1.6]: Combine all measurements into one function
    #
    # result = {
    #     'model_path': onnx_path,
    #     'size_mb': measure_model_size(onnx_path),
    #     'accuracy': measure_accuracy(onnx_path, test_loader),
    #     'latency': measure_latency(onnx_path, dummy_input_np),
    # }
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.6]: Implement benchmark_single")


def benchmark_comparison(model_paths, test_loader, dummy_input_np):
    """
    Benchmark multiple models and produce a comparison table.

    Args:
        model_paths: dict of {name: onnx_path}
        test_loader: Test DataLoader
        dummy_input_np: Numpy dummy input

    Returns:
        List of result dicts
    """
    # =========================================================================
    # TODO [Step 2.4]: Implement comparison benchmarking
    #
    # For each model in model_paths:
    #   1. result = benchmark_single(path, test_loader, dummy_input_np)
    #   2. result['name'] = name
    #   3. Append to results list
    #
    # Print formatted comparison table using tabulate:
    #   from tabulate import tabulate
    #   headers = ["Model", "Accuracy", "Latency(ms)", "Size(MB)", "Speedup"]
    #   # Calculate speedup relative to first model (FP32 baseline)
    #
    # Save results to CSV: results/tables/benchmark_comparison.csv
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.4]: Implement benchmark_comparison")


def format_results_table(results):
    """
    Format benchmark results as a pretty table.

    Args:
        results: List of result dicts from benchmark_single

    Returns:
        Formatted string table
    """
    # =========================================================================
    # TODO [Step 1.6]: Implement table formatting
    #
    # Use tabulate library:
    #   from tabulate import tabulate
    #   rows = []
    #   for r in results:
    #       rows.append([
    #           r['name'],
    #           f"{r['accuracy']:.4f}",
    #           f"{r['latency']['mean_ms']:.2f} ± {r['latency']['std_ms']:.2f}",
    #           f"{r['size_mb']:.2f}",
    #       ])
    #   print(tabulate(rows, headers=["Model", "Acc", "Latency(ms)", "Size(MB)"]))
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.6]: Implement format_results_table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX models")
    parser.add_argument("--model", type=str, help="Single ONNX model to benchmark")
    parser.add_argument("--workload", choices=["1d", "2d"], required=True,
                        help="Which workload (determines test data)")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all models in models/ directory")
    parser.add_argument("--compare", nargs="+",
                        help="List of ONNX model paths to compare")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of latency measurement runs")
    args = parser.parse_args()

    set_seed(42)

    # =========================================================================
    # TODO [Step 1.6 / 2.4]: Implement the main benchmarking script
    #
    # 1. Load test data based on workload choice
    # 2. Create appropriate dummy input (numpy)
    # 3. If --model: benchmark single model
    # 4. If --compare: benchmark and compare multiple models
    # 5. If --all: find all .onnx files in models/ and compare
    # 6. Save results to results/tables/
    # =========================================================================
    print("TODO: Implement benchmarking main script")
