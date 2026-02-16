"""
Module: profiling.py
Part: A
Purpose: Timing and memory profiling utilities.

Learning Goals:
- Accurate latency measurement with warmup
- Simple memory usage estimation
- ONNX model introspection (op counts, parameter counts)

Usage:
    from src.utils.profiling import Timer, profile_ort_session
"""

import time
import os

import numpy as np


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("inference"):
            result = model(input)
        # Prints: "[inference] 12.34 ms"
    """

    def __init__(self, name="", verbose=True):
        self.name = name
        self.verbose = verbose
        self.elapsed_ms = 0

    # =========================================================================
    # TODO [Step 1.6]: Implement Timer context manager
    #
    # def __enter__(self):
    #     self.start = time.perf_counter()
    #     return self
    #
    # def __exit__(self, *args):
    #     self.elapsed_ms = (time.perf_counter() - self.start) * 1000
    #     if self.verbose:
    #         print(f"[{self.name}] {self.elapsed_ms:.2f} ms")
    # =========================================================================

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        if self.verbose:
            print(f"[{self.name}] {self.elapsed_ms:.2f} ms")


def profile_ort_session(onnx_path, dummy_input_np, num_warmup=10, num_runs=100):
    """
    Profile an ONNX Runtime session with detailed latency statistics.

    Args:
        onnx_path: Path to ONNX model
        dummy_input_np: Numpy input array
        num_warmup: Warmup iterations
        num_runs: Measurement iterations

    Returns:
        dict with 'mean_ms', 'std_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'min_ms'
    """
    # =========================================================================
    # TODO [Step 1.6]: Implement ORT profiling
    #
    # See scripts/bench.py for reference — this is a reusable version.
    # =========================================================================
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path,
                                   providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input_np})

    # Measure
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input_np})
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms':  float(np.std(latencies)),
        'p50_ms':  float(np.percentile(latencies, 50)),
        'p95_ms':  float(np.percentile(latencies, 95)),
        'p99_ms':  float(np.percentile(latencies, 99)),
        'min_ms':  float(np.min(latencies)),
    }


def get_model_info(onnx_path):
    """
    Get model information from an ONNX file.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        dict with 'num_ops', 'num_params', 'size_mb', 'op_types', 'layers'
    """
    # =========================================================================
    # TODO [Step 1.6]: Implement model info extraction
    #
    # import onnx
    # model = onnx.load(onnx_path)
    # graph = model.graph
    #
    # # Count ops
    # op_types = {}
    # for node in graph.node:
    #     op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    #
    # # Count parameters (from initializers)
    # num_params = sum(
    #     np.prod(init.dims) for init in graph.initializer
    # )
    #
    # return {
    #     'num_ops': len(graph.node),
    #     'num_params': num_params,
    #     'size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
    #     'op_types': op_types,
    # }
    # =========================================================================
    import onnx

    model = onnx.load(onnx_path)
    graph = model.graph

    # Count ops
    op_types = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    # Count parameters (from initializers)
    num_params = sum(
        int(np.prod(init.dims)) for init in graph.initializer
    )

    return {
        'num_ops': len(graph.node),
        'num_params': num_params,
        'size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
        'op_types': op_types,
    }
