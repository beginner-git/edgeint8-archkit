"""
Module: bit_growth.py
Part: C
Purpose: MAC bit growth analysis and accumulator width derivation.

Learning Goals:
- Derive how many bits are needed for multiply-accumulate operations
- Understand why INT32 accumulators are standard for INT8 inference
- Analyze per-layer accumulator requirements for your models

Prerequisites:
- Understanding of binary integer representation
- Understanding of Conv/GEMM operations (multiply-accumulate)

什么是 Bit Growth？
当你做定点数的乘法和加法时，结果的位宽会"增长"：
  - 乘法：M-bit × N-bit = (M+N)-bit result
  - 加法：accumulating K products needs extra ceil(log2(K)) bits
  - 总位宽 = M + N + ceil(log2(K))

举例：INT8 × INT8 with K accumulations
  - 乘法：8 + 8 = 16 bits per product
  - 累加：如果 K = 576 (3×3 kernel, 64 channels)
    → ceil(log2(576)) = 10 bits
  - 总共需要：16 + 10 = 26 bits → 使用 INT32 累加器（32 bits，有余量）

为什么 INT32 是标准选择？
  - 大多数 Conv 层的 K 值不超过 2^16 = 65536
  - 16 + 16 = 32 bits，刚好用 INT32 覆盖
  - INT32 累加器在 hardware 中也是标准配置（NVDLA, Gemmini 等）
  - 如果 K 更大（如大型 FC 层），可能需要 INT64 或分段累加

Usage:
    python -m src.fixed_point.bit_growth
"""

import math
import numpy as np


def compute_mac_bit_growth(input_bits, weight_bits, accumulation_count):
    """
    Compute the total bit-width required for multiply-accumulate.

    Args:
        input_bits: Bit-width of input operand (e.g., 8)
        weight_bits: Bit-width of weight operand (e.g., 8)
        accumulation_count: Number of products accumulated (K)

    Returns:
        dict with:
            'product_bits': Bits for a single multiplication result
            'accumulation_bits': Extra bits needed for accumulation
            'total_bits': Minimum bits for the accumulator
            'recommended_hw_bits': Nearest standard HW width (16/32/64)

    位增长推导（核心公式）：

        假设 input 是 M-bit signed integer: 范围 [-2^(M-1), 2^(M-1)-1]
        假设 weight 是 N-bit signed integer: 范围 [-2^(N-1), 2^(N-1)-1]

        单次乘法：
            最大正值 = 2^(M-1) × 2^(N-1) = 2^(M+N-2)
            最大负值 = -2^(M-1) × (2^(N-1)-1) ≈ -2^(M+N-2)
            → 需要 (M+N) bits 来表示乘积（包括符号位）

        K 次累加：
            最大累加值 = K × 2^(M+N-2)
            需要额外 ceil(log2(K)) bits
            → 总共需要 (M+N) + ceil(log2(K)) bits

        Example (INT8 × INT8, K=576):
            product_bits = 8 + 8 = 16
            accumulation_bits = ceil(log2(576)) = 10
            total_bits = 26
            recommended = 32 (INT32)
    """
    # =========================================================================
    # TODO [Step 3.1]: Implement bit growth calculation
    #
    # Steps:
    #   1. product_bits = input_bits + weight_bits
    #   2. accumulation_bits = math.ceil(math.log2(accumulation_count))
    #      if accumulation_count > 0 else 0
    #   3. total_bits = product_bits + accumulation_bits
    #   4. recommended = next power of 2 that is >= total_bits
    #      and is one of [16, 32, 64]
    #
    # Return a dict with all the computed values.
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.1]: Implement compute_mac_bit_growth")


def compute_accumulation_count_conv2d(kernel_h, kernel_w, in_channels):
    """
    Compute the number of MAC operations per output element for Conv2d.

    For Conv2d(C_in, C_out, (K_h, K_w)):
        Each output element = sum over C_in × K_h × K_w multiply-adds
        K = C_in × K_h × K_w

    Args:
        kernel_h: Kernel height
        kernel_w: Kernel width
        in_channels: Number of input channels

    Returns:
        K: Number of accumulations per output element
    """
    # =========================================================================
    # TODO [Step 3.1]: Compute K for Conv2d
    #
    # K = kernel_h * kernel_w * in_channels
    #
    # 直觉：对于一个 3×3 Conv with 64 input channels，
    # 每个输出像素需要 3×3×64 = 576 次乘加。
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.1]: Implement accumulation count")


def compute_accumulation_count_conv1d(kernel_size, in_channels):
    """Compute K for Conv1d. K = kernel_size × in_channels."""
    # =========================================================================
    # TODO [Step 3.1]: K = kernel_size * in_channels
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.1]: Implement 1D accumulation count")


def recommend_accumulator_width(conv_params):
    """
    Recommend accumulator bit-width for a Conv layer.

    Args:
        conv_params: dict with 'kernel_size', 'in_channels', 'type' ('1d' or '2d')

    Returns:
        dict with 'K', 'min_bits', 'recommended_bits', 'margin'
    """
    # =========================================================================
    # TODO [Step 3.1]: Implement accumulator recommendation
    #
    # 1. Compute K based on conv type
    # 2. Compute bit growth for INT8 × INT8
    # 3. Recommend standard HW width
    # 4. Compute margin (how many bits of headroom)
    #
    # margin = recommended_bits - total_bits
    # A margin of 4+ bits means there's comfortable headroom.
    # A margin of 0-2 bits means you're at risk of overflow in edge cases.
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.1]: Implement recommend_accumulator_width")


def analyze_model_bit_growth(model):
    """
    Analyze bit growth for all Conv layers in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        analysis: List of dicts, one per Conv layer
    """
    # =========================================================================
    # TODO [Step 3.1]: Implement model-wide bit growth analysis
    #
    # import torch.nn as nn
    #
    # analysis = []
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         K = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
    #         bg = compute_mac_bit_growth(8, 8, K)
    #         analysis.append({
    #             'layer': name,
    #             'type': 'Conv2d',
    #             'kernel': f"{module.kernel_size[0]}×{module.kernel_size[1]}",
    #             'in_ch': module.in_channels,
    #             'out_ch': module.out_channels,
    #             'K': K,
    #             **bg
    #         })
    #     elif isinstance(module, nn.Conv1d):
    #         K = module.kernel_size[0] * module.in_channels
    #         bg = compute_mac_bit_growth(8, 8, K)
    #         analysis.append({'layer': name, 'type': 'Conv1d', 'K': K, **bg})
    #     elif isinstance(module, nn.Linear):
    #         K = module.in_features
    #         bg = compute_mac_bit_growth(8, 8, K)
    #         analysis.append({'layer': name, 'type': 'Linear', 'K': K, **bg})
    #
    # Print as table using tabulate
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.1]: Implement analyze_model_bit_growth")


if __name__ == "__main__":
    print("=" * 60)
    print("  Bit Growth Analysis Self-Test")
    print("=" * 60)

    # Example: INT8 Conv2d(64, 128, 3×3)
    print("\n--- Example: Conv2d(64, 128, 3×3) ---")
    K = compute_accumulation_count_conv2d(kernel_h=3, kernel_w=3, in_channels=64)
    print(f"  Accumulation count K = {K}")
    bg = compute_mac_bit_growth(input_bits=8, weight_bits=8, accumulation_count=K)
    print(f"  Product bits:      {bg['product_bits']}")
    print(f"  Accumulation bits: {bg['accumulation_bits']}")
    print(f"  Total bits needed: {bg['total_bits']}")
    print(f"  Recommended HW:    INT{bg['recommended_hw_bits']}")

    # Example: INT8 Conv1d(1, 16, kernel=7)
    print("\n--- Example: Conv1d(1, 16, kernel=7) ---")
    K1d = compute_accumulation_count_conv1d(kernel_size=7, in_channels=1)
    print(f"  Accumulation count K = {K1d}")
    bg1d = compute_mac_bit_growth(8, 8, K1d)
    print(f"  Total bits needed: {bg1d['total_bits']}")
    print(f"  Recommended HW:    INT{bg1d['recommended_hw_bits']}")

    # Full model analysis
    print("\n--- Full Model Analysis (TinyCNN2D) ---")
    try:
        from models.tiny_cnn_2d import get_tiny_cnn_2d
        model = get_tiny_cnn_2d()
        analyze_model_bit_growth(model)
    except NotImplementedError:
        print("  (Implement the model first, then run this analysis)")
