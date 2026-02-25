"""
Module: quantize_ptq.py
Part: B
Purpose: INT8 Post-Training Quantization using ONNX Runtime.

Learning Goals:
- Understand static quantization vs dynamic quantization
- Implement CalibrationDataReader for ORT
- Compare different quantization strategies systematically
- Learn what each configuration choice means for accuracy and performance

Prerequisites:
- Step 1.5: ONNX models exported
- Step 2.1-2.2: Understand quantization math (src/quant/)

什么是 Post-Training Quantization (PTQ)？
- 训练后量化：模型已经训练好了，我们在不重新训练的情况下把它从 FP32 转成 INT8
- 与 QAT（量化感知训练）的区别：PTQ 不需要重新训练，速度快但精度可能稍低
- 核心步骤：用一小批校准数据（calibration data）统计每层的数值范围，
  然后计算 scale 和 zero_point

Static vs Dynamic Quantization:
- Static: 提前校准激活范围，推理时直接用固定的 scale/zp （本项目使用这种）
- Dynamic: 每次推理时动态计算激活的 scale/zp （更灵活但略慢）

Usage:
    python scripts/quantize_ptq.py --workload 2d
    python scripts/quantize_ptq.py --workload 1d
    python scripts/quantize_ptq.py --workload all --sweep
"""

import os
import sys
import argparse
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import set_seed, ensure_dir
from src.utils.data import get_cifar10_loaders, get_synthetic_signal_loaders


class CalibrationDataReaderBase:
    """
    Base class for ONNX Runtime calibration data reader.

    ORT 的量化工具需要一个实现了 get_next() 方法的对象，
    用来逐批喂校准数据。校准数据应该是训练集的一个小子集
    （通常 100-500 个样本就够了）。

    为什么需要校准数据？
    - 权重是固定的，范围可以直接从权重计算
    - 但激活（中间层输出）的范围取决于输入数据
    - 需要用代表性的数据跑一遍网络，统计激活范围
    """
    pass


class CifarCalibrationDataReader(CalibrationDataReaderBase):
    """Calibration data reader for CIFAR-10 (Workload-2)."""

    def __init__(self, calibration_loader, input_name="input"):
        """
        Args:
            calibration_loader: DataLoader with calibration samples
            input_name: ONNX model input tensor name
        """
        # =====================================================================
        # TODO [Step 2.3]: Implement calibration data reader for CIFAR-10
        #
        # ORT 需要实现 onnxruntime.quantization.CalibrationDataReader 接口：
        #   from onnxruntime.quantization import CalibrationDataReader
        #
        # 你需要：
        #   1. 在 __init__ 中：
        #      - 将 calibration_loader 中的所有数据预加载到一个列表
        #      - 每个元素是 {input_name: numpy_array} 格式
        #      - 记录当前索引 self.iter = 0
        #
        #   2. 实现 get_next() 方法：
        #      - 返回下一个 {input_name: data} 字典
        #      - 没有更多数据时返回 None
        #
        # 注意：ORT 需要 numpy array，不是 torch tensor！
        # 转换：data_np = data.numpy().astype(np.float32)
        # =====================================================================
        self.data = []
        for data, _ in calibration_loader:
            for i in range(data.shape[0]):
                data_np = data[i:i+1].numpy().astype(np.float32)
                self.data.append({input_name: data_np})
        self.iter = 0

        # raise NotImplementedError("TODO [Step 2.3]: Init calibration reader")

    def get_next(self):
        """Return next calibration sample, or None if exhausted."""
        if self.iter >= len(self.data):
            return None
        result = self.data[self.iter]
        self.iter += 1
        return result
        # raise NotImplementedError("TODO [Step 2.3]: Implement get_next")


class SignalCalibrationDataReader(CalibrationDataReaderBase):
    """Calibration data reader for synthetic signal (Workload-1)."""

    def __init__(self, calibration_loader, input_name="input"):
        # =====================================================================
        # TODO [Step 2.3]: Same as CifarCalibrationDataReader but for 1D data
        # =====================================================================
        self.data = []
        for data, _ in calibration_loader:
            for i in range(data.shape[0]):
                data_np = data[i:i + 1].numpy().astype(np.float32)
                self.data.append({input_name: data_np})
        self.iter = 0
        # raise NotImplementedError("TODO [Step 2.3]: Init signal calibration reader")

    def get_next(self):
        if self.iter >= len(self.data):
            return None
        result = self.data[self.iter]
        self.iter += 1
        return result
        # raise NotImplementedError("TODO [Step 2.3]: Implement get_next")


def quantize_model_static(model_path, output_path, calibration_reader,
                          per_channel=True, calibrate_method="MinMax",
                          activation_type="QUInt8", weight_type="QInt8"):
    """
    Apply static INT8 quantization to an ONNX model.

    Args:
        model_path: Path to FP32 ONNX model
        output_path: Path to save quantized model
        calibration_reader: CalibrationDataReader instance
        per_channel: If True, use per-channel weight quantization
        calibrate_method: "MinMax", "Entropy", or "Percentile"
        activation_type: "QUInt8" (asymmetric) or "QInt8" (symmetric)
        weight_type: "QInt8" (standard for weights)

    Returns:
        output_path
    """
    # =========================================================================
    # TODO [Step 2.3]: Implement static quantization using ORT
    #
    # Steps:
    #   from onnxruntime.quantization import (
    #       quantize_static,
    #       CalibrationMethod,
    #       QuantType,
    #       QuantFormat,
    #   )
    #
    #   # Map string args to ORT enums:
    #   calibrate_method_map = {
    #       "MinMax": CalibrationMethod.MinMax,
    #       "Entropy": CalibrationMethod.Entropy,
    #       "Percentile": CalibrationMethod.Percentile,
    #   }
    #   quant_type_map = {
    #       "QUInt8": QuantType.QUInt8,
    #       "QInt8": QuantType.QInt8,
    #   }
    #
    #   quantize_static(
    #       model_input=model_path,
    #       model_output=output_path,
    #       calibration_data_reader=calibration_reader,
    #       quant_format=QuantFormat.QDQ,        # QDQ format (recommended)
    #       per_channel=per_channel,
    #       calibrate_method=calibrate_method_map[calibrate_method],
    #       activation_type=quant_type_map[activation_type],
    #       weight_type=quant_type_map[weight_type],
    #   )
    #
    # 关键概念解释：
    #
    # 1. per_channel vs per_tensor:
    #    - per_tensor: 整个权重张量共享一个 scale/zp → 简单但精度可能低
    #    - per_channel: 每个输出通道有自己的 scale/zp → 精度更高但实现更复杂
    #    - 对硬件的影响：per_channel 需要在每个通道用不同的 scale 做 requantize
    #
    # 2. activation_type:
    #    - QUInt8 (0~255): 非对称量化，需要 zero_point，适合 ReLU 后的激活（总是>=0）
    #    - QInt8 (-128~127): 对称量化，zero_point=0，硬件实现更简单
    #
    # 3. calibrate_method:
    #    - MinMax: 用 min/max 值确定范围。简单但对 outlier 敏感
    #    - Entropy: 用 KL 散度最小化量化前后分布差异。对 outlier 更鲁棒
    #    - Percentile: 用百分位数（如 99.99%）截断 outlier
    #
    # 4. QDQ vs QOperator format:
    #    - QDQ: 在图中插入 QuantizeLinear/DequantizeLinear 节点（推荐）
    #    - QOperator: 直接使用量化算子如 QLinearConv（老格式）
    # =========================================================================
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationMethod,
        QuantType,
        QuantFormat
    )

    calibration_method_map = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    quant_type_map = {
        "QUInt8": QuantType.QUInt8,
        "QInt8": QuantType.QInt8,
    }

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        calibrate_method=calibration_method_map[calibrate_method],
        activation_type=quant_type_map[activation_type],
        weight_type=quant_type_map[weight_type],
    )
    return output_path

    raise NotImplementedError("TODO [Step 2.3]: Implement quantize_model_static")


def run_quantization_sweep(model_path, calibration_reader, output_dir):
    """
    Run a sweep of quantization configurations.

    Tests all combinations of:
    - per_channel: [True, False]
    - calibrate_method: ["MinMax", "Percentile"]
    - activation_type: ["QUInt8", "QInt8"]

    Args:
        model_path: Path to FP32 ONNX model
        calibration_reader_factory: Callable that creates a fresh CalibrationDataReader
        output_dir: Directory to save quantized models

    Returns:
        List of (config_name, output_path) tuples
    """
    # =========================================================================
    # TODO [Step 2.3]: Implement quantization sweep
    #
    # Generate all combinations and run quantize_model_static for each:
    #
    #   configs = list(itertools.product(
    #       [True, False],           # per_channel
    #       ["MinMax", "Percentile"],  # calibrate_method
    #       ["QUInt8", "QInt8"],      # activation_type
    #   ))
    #
    #   results = []
    #   for per_channel, calib_method, act_type in configs:
    #       name = f"int8_{'perchan' if per_channel else 'pertensor'}"
    #              f"_{calib_method.lower()}_{act_type.lower()}"
    #       output_path = os.path.join(output_dir, f"{name}.onnx")
    #       # Note: Need a FRESH calibration reader for each run!
    #       quantize_model_static(model_path, output_path, ...)
    #       results.append((name, output_path))
    #
    # Print summary table of all generated models
    #
    # 注意：每次调用 quantize_static 都会消耗 calibration reader 的数据，
    # 所以需要每次创建一个新的 reader，或者在 reader 中实现 reset 功能。
    # =========================================================================
    ensure_dir(output_dir)

    configs = list(itertools.product(
        [True, False],
        ["MinMax", "Percentile"],
        ["QUInt8", "QInt8"],
    ))

    results = []
    for per_channel, calib_method, act_type in configs:
        name = (f"int8_{'perchan' if per_channel else 'pertensor'}"
                f"_{calib_method.lower()}_{act_type.lower()}")
        output_path = os.path.join(output_dir, f"{name}.onnx")

        fresh_reader = calibration_reader()

        quantize_model_static(
            model_path=model_path,
            output_path=output_path,
            calibration_reader=fresh_reader,
            per_channel=per_channel,
            calibrate_method=calib_method,
            activation_type=act_type,
        )
        results.append((name, output_path))
        print(f"Generated {name}")

    return results

    # raise NotImplementedError("TODO [Step 2.3]: Implement quantization sweep")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INT8 PTQ Quantization")
    parser.add_argument("--workload", choices=["1d", "2d", "all"], default="all")
    parser.add_argument("--model", type=str, help="Specific ONNX model to quantize")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full quantization sweep")
    parser.add_argument("--calibration-samples", type=int, default=100,
                        help="Number of calibration samples")
    args = parser.parse_args()

    set_seed(42)

    # =========================================================================
    # TODO [Step 2.3]: Implement main quantization script
    #
    # 1. Load calibration data (subset of training data)
    # 2. If --sweep: run_quantization_sweep()
    #    Else: run single quantization with default settings
    # 3. Print summary of generated models
    # =========================================================================
    calibration_loader, _ = get_cifar10_loaders(data_dir="data/cifar10")
    # DataReader = CifarCalibrationDataReader(calibration_loader)
    reader_facroty = lambda: CifarCalibrationDataReader(calibration_loader)
    if args.sweep:
        results = run_quantization_sweep(args.model, reader_facroty, output_dir="models")

    print("TODO: Implement quantization main script")
