"""
Module: quantize_utils.py
Part: B
Purpose: Core quantization math — hand-written implementations for learning.

Learning Goals:
- Derive and implement scale/zero_point computation from scratch
- Understand symmetric vs asymmetric quantization
- Understand the quantize → dequantize roundtrip and its error
- Build intuition BEFORE using ORT's built-in quantization

Prerequisites:
- Basic linear algebra and numerical precision concepts
- Understanding of integer representation (int8: -128~127, uint8: 0~255)

这个文件是整个量化学习的基础。在使用 ORT 的 quantize_static 之前，
你应该先手写这些函数，理解每一步的数学原理。

量化的核心思想：
  把浮点数 x 映射到整数 q：
    q = round(x / scale) + zero_point
    x ≈ (q - zero_point) * scale

  其中 scale 和 zero_point 决定了映射关系：
    - scale: 每个量化步长对应的浮点值
    - zero_point: 浮点数 0 对应的整数值

Usage (self-test):
    python -m src.quant.quantize_utils
"""

import numpy as np


def compute_scale_zp_symmetric(tensor, num_bits=8):
    """
    Compute scale and zero_point for SYMMETRIC quantization.

    Symmetric quantization maps [-max_abs, +max_abs] to [-2^(b-1)+1, 2^(b-1)-1].
    Zero_point is always 0 (float 0.0 maps to integer 0).

    Args:
        tensor: numpy array of float values (weights or activations)
        num_bits: Quantization bit-width (default 8)

    Returns:
        scale: float
        zero_point: int (always 0 for symmetric)

    对称量化公式推导：
        qmin = -(2^(b-1) - 1) = -127  (for int8)
        qmax = 2^(b-1) - 1   = 127   (for int8)
        注意：不用 -128，是为了保持对称（|-127| == |127|）

        max_abs = max(|tensor|)
        scale = max_abs / qmax = max_abs / 127

        quantize:   q = clamp(round(x / scale), qmin, qmax)
        dequantize: x_hat = q * scale

    对称量化的优势：
        - zero_point = 0，硬件实现更简单（不需要减去偏移量）
        - MAC 时不需要额外的 zero_point 补偿项
        - 常用于 权重 的量化

    对称量化的劣势：
        - 如果数据分布不对称（如 ReLU 后全是正数），会浪费一半的量化范围
        - 这就是为什么 激活 通常用非对称量化
    """
    # =========================================================================
    # TODO [Step 2.1]: Implement symmetric quantization parameters
    #
    # Steps:
    #   1. qmax = 2**(num_bits - 1) - 1      # 127 for int8
    #   2. max_abs = np.max(np.abs(tensor))
    #   3. scale = max_abs / qmax if max_abs > 0 else 1.0  # Avoid division by 0
    #   4. zero_point = 0
    #   5. return scale, zero_point
    #
    # Edge case: if tensor is all zeros, scale should be 1.0 (or any positive)
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.1]: Implement compute_scale_zp_symmetric")


def compute_scale_zp_asymmetric(tensor, num_bits=8):
    """
    Compute scale and zero_point for ASYMMETRIC quantization.

    Asymmetric quantization maps [min, max] to [0, 2^b - 1] (for uint8).
    Zero_point is generally non-zero.

    Args:
        tensor: numpy array of float values
        num_bits: Quantization bit-width (default 8)

    Returns:
        scale: float
        zero_point: int

    非对称量化公式推导：
        qmin = 0               (for uint8)
        qmax = 2^b - 1 = 255   (for uint8)

        min_val = min(tensor)
        max_val = max(tensor)
        scale = (max_val - min_val) / (qmax - qmin)
              = (max_val - min_val) / 255

        zero_point = round(qmin - min_val / scale)
                   = round(-min_val / scale)  (since qmin=0)
        zero_point = clamp(zero_point, qmin, qmax)

        quantize:   q = clamp(round(x / scale) + zero_point, qmin, qmax)
        dequantize: x_hat = (q - zero_point) * scale

    非对称量化的优势：
        - 完整利用量化范围（no waste）
        - 适合单侧分布（如 ReLU 后的激活，全 >= 0）

    非对称量化的劣势：
        - 需要存储和处理 zero_point
        - MAC 时需要额外计算 zero_point 的补偿项：
          sum(w * (x - zp_x)) = sum(w*x) - zp_x * sum(w)
          这增加了硬件复杂度
    """
    # =========================================================================
    # TODO [Step 2.1]: Implement asymmetric quantization parameters
    #
    # Steps:
    #   1. qmin, qmax = 0, 2**num_bits - 1   # 0, 255 for uint8
    #   2. min_val = np.min(tensor)
    #   3. max_val = np.max(tensor)
    #   4. scale = (max_val - min_val) / (qmax - qmin)
    #      if scale == 0: scale = 1.0  # Avoid division by 0
    #   5. zero_point = int(np.round(qmin - min_val / scale))
    #   6. zero_point = np.clip(zero_point, qmin, qmax)
    #   7. return scale, int(zero_point)
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.1]: Implement compute_scale_zp_asymmetric")


def quantize_tensor(tensor, scale, zero_point, num_bits=8, symmetric=True):
    """
    Quantize a float tensor to integer.

    Args:
        tensor: numpy array of float values
        scale: Quantization scale
        zero_point: Quantization zero point
        num_bits: Bit width
        symmetric: If True, use signed int range; if False, use unsigned int range

    Returns:
        q_tensor: numpy array of quantized integer values
    """
    # =========================================================================
    # TODO [Step 2.1]: Implement tensor quantization
    #
    # Formula:
    #   q = clamp(round(x / scale) + zero_point, qmin, qmax)
    #
    # For symmetric (signed int8):
    #   qmin = -(2**(num_bits-1) - 1)  = -127
    #   qmax = 2**(num_bits-1) - 1     = 127
    #
    # For asymmetric (unsigned uint8):
    #   qmin = 0
    #   qmax = 2**num_bits - 1         = 255
    #
    # Steps:
    #   1. Determine qmin, qmax based on symmetric flag
    #   2. q = np.round(tensor / scale) + zero_point
    #   3. q = np.clip(q, qmin, qmax).astype(np.int8 or np.uint8)
    #
    # 注意 round 的行为：np.round 默认是"银行家舍入"（round half to even），
    # 这与 C/硬件中常用的 "round half away from zero" 不同。
    # 实际硬件中使用哪种 rounding 策略会影响量化误差。
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.1]: Implement quantize_tensor")


def dequantize_tensor(q_tensor, scale, zero_point):
    """
    Dequantize an integer tensor back to float.

    Args:
        q_tensor: numpy array of quantized integers
        scale: Quantization scale
        zero_point: Quantization zero point

    Returns:
        tensor: numpy array of dequantized float values
    """
    # =========================================================================
    # TODO [Step 2.1]: Implement tensor dequantization
    #
    # Formula:
    #   x_hat = (q - zero_point) * scale
    #
    # This is a simple linear mapping. The result is a "staircase" approximation
    # of the original values — each quantized level maps back to the midpoint
    # of its range.
    #
    # 反量化后的值 x_hat 不会完全等于原始值 x，差异就是"量化误差"。
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.1]: Implement dequantize_tensor")


def compute_quantization_error(original, reconstructed):
    """
    Compute error statistics between original and quantize-dequantized values.

    Args:
        original: Original float tensor (numpy array)
        reconstructed: Dequantized tensor (numpy array)

    Returns:
        dict with: 'mse', 'max_abs_error', 'mean_abs_error', 'sqnr_db'
    """
    # =========================================================================
    # TODO [Step 2.1]: Implement error statistics
    #
    # Metrics:
    #   error = original - reconstructed
    #   mse = np.mean(error**2)
    #   max_abs_error = np.max(np.abs(error))
    #   mean_abs_error = np.mean(np.abs(error))
    #
    #   SQNR (Signal-to-Quantization-Noise Ratio):
    #     SQNR = 10 * log10(signal_power / noise_power)
    #          = 10 * log10(mean(x^2) / mean(error^2))
    #     单位是 dB，越高越好（量化噪声越小）
    #     经验法则：理想的 N-bit 量化 SQNR ≈ 6.02*N dB
    #     所以 8-bit 理想 SQNR ≈ 48 dB
    #
    # 这些指标帮助你量化地评估量化策略的好坏。
    # =========================================================================
    raise NotImplementedError("TODO [Step 2.1]: Implement compute_quantization_error")


# =============================================================================
# Self-test: Run this file to verify implementations
# Usage: python -m src.quant.quantize_utils
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Quantization Utils Self-Test")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: Symmetric quantization of a weight-like tensor
    print("\n--- Test 1: Symmetric Quantization ---")
    weights = np.random.randn(64, 32).astype(np.float32)  # Gaussian weights
    scale_s, zp_s = compute_scale_zp_symmetric(weights)
    q_weights = quantize_tensor(weights, scale_s, zp_s, symmetric=True)
    deq_weights = dequantize_tensor(q_weights, scale_s, zp_s)
    err_s = compute_quantization_error(weights, deq_weights)
    print(f"  Scale: {scale_s:.6f}, ZP: {zp_s}")
    print(f"  MSE: {err_s['mse']:.8f}, Max Error: {err_s['max_abs_error']:.6f}")
    print(f"  SQNR: {err_s['sqnr_db']:.1f} dB")

    # Test 2: Asymmetric quantization of an activation-like tensor (ReLU output)
    print("\n--- Test 2: Asymmetric Quantization (ReLU activations) ---")
    activations = np.abs(np.random.randn(1, 64, 8, 8).astype(np.float32))  # All positive
    scale_a, zp_a = compute_scale_zp_asymmetric(activations)
    q_act = quantize_tensor(activations, scale_a, zp_a, symmetric=False)
    deq_act = dequantize_tensor(q_act, scale_a, zp_a)
    err_a = compute_quantization_error(activations, deq_act)
    print(f"  Scale: {scale_a:.6f}, ZP: {zp_a}")
    print(f"  MSE: {err_a['mse']:.8f}, Max Error: {err_a['max_abs_error']:.6f}")
    print(f"  SQNR: {err_a['sqnr_db']:.1f} dB")

    # Test 3: Compare symmetric vs asymmetric on same data
    print("\n--- Test 3: Symmetric vs Asymmetric on ReLU activations ---")
    scale_s2, zp_s2 = compute_scale_zp_symmetric(activations)
    q_act_s = quantize_tensor(activations, scale_s2, zp_s2, symmetric=True)
    deq_act_s = dequantize_tensor(q_act_s, scale_s2, zp_s2)
    err_s2 = compute_quantization_error(activations, deq_act_s)
    print(f"  Symmetric  SQNR: {err_s2['sqnr_db']:.1f} dB")
    print(f"  Asymmetric SQNR: {err_a['sqnr_db']:.1f} dB")
    print(f"  -> Asymmetric should be better for ReLU activations (all positive)")
    print(f"     because symmetric wastes half the range on negative values.")

    print("\nAll tests passed!" if True else "Some tests failed!")
