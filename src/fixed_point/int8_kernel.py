"""
Module: int8_kernel.py
Part: C (CORE DELIVERABLE — 最重要的文件)
Purpose: Reference INT8 convolution kernel in pure Python/NumPy.

Learning Goals:
- Implement the complete INT8 inference pipeline for a Conv layer
- Understand quantize → int8 MAC → requantize → int8 output
- Validate against FP32 reference with bounded error
- This is what separates "I used a quantization tool" from
  "I understand what happens at the hardware level"

Prerequisites:
- Step 2.1: quantize_utils.py (scale/zp computation)
- Step 3.1: bit_growth.py (accumulator width understanding)
- Step 3.2: overflow.py (saturation strategy)

INT8 推理的完整数值流程（以 Conv2d 为例）：

    输入：x_fp32 (float32 input tensor)
    权重：w_fp32 (float32 weight tensor)

    Step 1: Quantize input and weights
        x_int8 = round(x_fp32 / s_x) + zp_x
        w_int8 = round(w_fp32 / s_w)          (weights are usually symmetric, zp=0)

    Step 2: INT8 MAC (multiply-accumulate) in INT32
        For each output position (n, co, h, w):
            acc_int32 = bias_int32[co]        (bias pre-computed in int32)
            for ci, kh, kw:
                acc_int32 += int32(x_int8[n, ci, h+kh, w+kw]) *
                             int32(w_int8[co, ci, kh, kw])

        这一步是纯整数运算，不涉及任何浮点数！
        硬件上就是 INT8 乘法器 + INT32 累加器。

    Step 3: Requantize (INT32 → INT8)
        Why: 累加结果是 INT32，但下一层需要 INT8 输入
        How:
            M = s_x * s_w / s_y     (combined scale, a float number)
            y_int8 = round(acc_int32 * M) + zp_y
            y_int8 = clamp(y_int8, -128, 127)

        在硬件中，M 通常被分解为 M = M0 * 2^(-shift)，其中 M0 是定点数，
        这样 requantize 就不需要浮点乘法，只需要定点乘法 + 移位。
        但在本项目中，我们先用浮点实现 M，理解概念后再优化。

    Step 4: Dequantize output (for accuracy comparison)
        y_fp32_approx = (y_int8 - zp_y) * s_y

        这个值应该接近"用 FP32 直接做 Conv"的结果。
        差异就是量化引入的数值误差。

Bias 处理：
    在 INT8 推理中，bias 被预转换为 INT32：
    bias_int32 = round(bias_fp32 / (s_x * s_w))
    这样 bias 可以直接加到 INT32 累加器中。

Usage (self-test):
    python -m src.fixed_point.int8_kernel
"""

import numpy as np


def requantize(acc_int32, input_scale, weight_scale, output_scale, output_zp):
    """
    Requantize INT32 accumulation result to INT8 output.

    This is the bridge between layers: each layer's INT32 accumulation
    must be converted back to INT8 for the next layer.

    Args:
        acc_int32: INT32 accumulation result (numpy array)
        input_scale: Scale of the input quantization (s_x)
        weight_scale: Scale of the weight quantization (s_w)
        output_scale: Scale of the output quantization (s_y)
        output_zp: Zero point of the output quantization (zp_y)

    Returns:
        output_int8: INT8 output (numpy array of int8)

    Requantize 公式：
        M = (s_x * s_w) / s_y       # Combined scale factor
        output = round(acc_int32 * M) + zp_y
        output = clamp(output, -128, 127)

    M 的物理意义：
        acc_int32 的单位是 (s_x * s_w)，即每个整数 1 对应的浮点值是 s_x * s_w
        输出的单位是 s_y
        所以 M = (s_x * s_w) / s_y 就是单位转换因子

    硬件优化（了解即可，本项目不要求实现）：
        M 通常满足 0 < M < 1（因为 s_y 通常 > s_x * s_w）
        可以表示为 M = M0 * 2^(-shift)，其中 M0 是 [0.5, 1) 的定点数
        这样 requantize 就变成：定点乘法 + 右移 + 加 zp_y
    """
    # =========================================================================
    # TODO [Step 3.3]: Implement requantization
    #
    # Steps:
    #   1. M = (input_scale * weight_scale) / output_scale
    #   2. output = np.round(acc_int32.astype(np.float64) * M) + output_zp
    #      (use float64 to avoid precision issues)
    #   3. output = np.clip(output, -128, 127).astype(np.int8)
    #
    # 注意使用 float64：如果 acc_int32 的值很大（几十万），
    # 用 float32 做乘法可能会丢失低位精度。
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.3]: Implement requantize")


def quantize_bias(bias_fp32, input_scale, weight_scale):
    """
    Pre-quantize bias to INT32 for fixed-point accumulation.

    Args:
        bias_fp32: Float32 bias vector
        input_scale: Input quantization scale
        weight_scale: Weight quantization scale (can be per-channel array)

    Returns:
        bias_int32: INT32 bias vector
    """
    # =========================================================================
    # TODO [Step 3.3]: Implement bias quantization
    #
    # bias_int32 = round(bias_fp32 / (input_scale * weight_scale))
    #
    # 为什么 bias 的 scale 是 s_x * s_w？
    # 因为 bias 要加到累加器中，累加器的单位是 s_x * s_w。
    # 所以 bias 也要用同样的单位，这样加法才有意义。
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.3]: Implement quantize_bias")


def int8_conv2d_reference(input_int8, weight_int8, bias_int32,
                          input_scale, input_zp,
                          weight_scale, weight_zp,
                          output_scale, output_zp,
                          stride=1, padding=0):
    """
    Reference INT8 Conv2d implementation in pure NumPy.

    This implements the COMPLETE INT8 inference pipeline for one Conv2d layer:
    INT8 input × INT8 weight → INT32 accumulation → INT8 output

    Args:
        input_int8:    INT8 input, shape [N, C_in, H, W]
        weight_int8:   INT8 weight, shape [C_out, C_in, K_h, K_w]
        bias_int32:    INT32 bias, shape [C_out] (or None)
        input_scale:   Input quantization scale (float)
        input_zp:      Input zero point (int)
        weight_scale:  Weight scale (float or array of [C_out] for per-channel)
        weight_zp:     Weight zero point (int, usually 0 for symmetric)
        output_scale:  Output quantization scale (float)
        output_zp:     Output zero point (int)
        stride:        Convolution stride
        padding:       Zero-padding

    Returns:
        output_int8:   INT8 output, shape [N, C_out, H_out, W_out]
    """
    # =========================================================================
    # TODO [Step 3.3]: Implement reference INT8 Conv2d kernel
    #
    # This is the most important function in the entire project.
    # Implement it step by step:
    #
    # Step 0: Setup
    #   N, C_in, H_in, W_in = input_int8.shape
    #   C_out, _, K_h, K_w = weight_int8.shape
    #   H_out = (H_in + 2*padding - K_h) // stride + 1
    #   W_out = (W_in + 2*padding - K_w) // stride + 1
    #   output = np.zeros((N, C_out, H_out, W_out), dtype=np.int32)
    #
    # Step 1: Apply padding (if needed)
    #   if padding > 0:
    #       # Pad with input_zp (NOT 0!) because quantized "zero" is input_zp
    #       # 注意：量化后的 "0" 不是整数 0，而是 input_zp！
    #       # 如果 input_zp = 128 (uint8 asymmetric)，padding 应该填 128
    #       input_padded = np.pad(input_int8,
    #           ((0,0), (0,0), (padding,padding), (padding,padding)),
    #           mode='constant', constant_values=input_zp)
    #   else:
    #       input_padded = input_int8
    #
    # Step 2: Compute INT32 accumulation (core loop)
    #   for n in range(N):                    # Batch
    #     for co in range(C_out):             # Output channel
    #       for h in range(H_out):            # Output height
    #         for w in range(W_out):          # Output width
    #           acc = np.int32(0)
    #           # Add bias
    #           if bias_int32 is not None:
    #               acc = np.int32(bias_int32[co])
    #           # MAC loop
    #           for ci in range(C_in):        # Input channel
    #             for kh in range(K_h):       # Kernel height
    #               for kw in range(K_w):     # Kernel width
    #                 h_in = h * stride + kh
    #                 w_in = w * stride + kw
    #                 # INT8 × INT8 → INT16, accumulated in INT32
    #                 acc += np.int32(input_padded[n, ci, h_in, w_in]) * \
    #                        np.int32(weight_int8[co, ci, kh, kw])
    #           output[n, co, h, w] = acc
    #
    #   注意：所有运算都是整数！没有浮点！
    #   cast to int32 防止 int8 乘法溢出。
    #
    # Step 3: Requantize INT32 → INT8
    #   output_int8 = requantize(output, input_scale, weight_scale,
    #                            output_scale, output_zp)
    #
    # 优化提示（了解即可，先写正确的慢版本）：
    # - 上面的 7 层 for 循环非常慢，但清晰地展示了算法
    # - 可以用 im2col + GEMM 代替，但会掩盖算法细节
    # - 本项目追求清晰，不追求速度
    #
    # Zero-point 补偿（高级，可选）：
    # 如果 input_zp ≠ 0，上面的 MAC 结果包含了 zp 的贡献：
    #   acc = Σ(x_q * w_q)
    #       = Σ((x_real/s_x + zp_x) * (w_real/s_w + zp_w))
    #       = Σ(x_real*w_real)/(s_x*s_w) + zp_x*Σ(w_q) + zp_w*Σ(x_q) + K*zp_x*zp_w
    # 中间三项是 zp 补偿项，可以预计算。
    # 但如果你的权重是 symmetric (zp_w=0)，只需要处理 zp_x*Σ(w_q) 这一项。
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.3]: Implement int8_conv2d_reference")


def int8_conv1d_reference(input_int8, weight_int8, bias_int32,
                          input_scale, input_zp,
                          weight_scale, weight_zp,
                          output_scale, output_zp,
                          stride=1, padding=0):
    """
    Reference INT8 Conv1d implementation.

    Same logic as Conv2d but in 1D. Input shape: [N, C_in, L].

    Args:
        input_int8:    [N, C_in, L_in]
        weight_int8:   [C_out, C_in, K]
        bias_int32:    [C_out] or None
        ... (same scale/zp args as conv2d)

    Returns:
        output_int8:   [N, C_out, L_out]
    """
    # =========================================================================
    # TODO [Step 3.3]: Implement reference INT8 Conv1d kernel
    #
    # Same as Conv2d but simpler (only 1 spatial dimension):
    #   L_out = (L_in + 2*padding - K) // stride + 1
    #
    # The MAC loop has 3 levels instead of 5:
    #   for co, ci, k: acc += x[ci, l*stride+k] * w[co, ci, k]
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.3]: Implement int8_conv1d_reference")


def validate_against_float(fp32_output, int8_output, output_scale, output_zp,
                           tolerance_pct=5.0):
    """
    Compare INT8 kernel output against FP32 reference.

    Args:
        fp32_output: FP32 convolution output (from PyTorch)
        int8_output: INT8 kernel output (from int8_conv2d_reference)
        output_scale: Output quantization scale
        output_zp: Output zero point
        tolerance_pct: Acceptable percentage of outputs differing by >1 quant step

    Returns:
        dict with: 'mse', 'max_error', 'within_1_step_pct', 'passed'
    """
    # =========================================================================
    # TODO [Step 3.3]: Implement validation against float reference
    #
    # Steps:
    #   1. Dequantize int8 output: y_approx = (int8_output - output_zp) * output_scale
    #   2. Compute error: error = fp32_output - y_approx
    #   3. Statistics:
    #      mse = np.mean(error**2)
    #      max_error = np.max(np.abs(error))
    #      # Count outputs within 1 quantization step
    #      within_1_step = np.abs(error) <= output_scale
    #      within_1_step_pct = 100.0 * np.mean(within_1_step)
    #   4. passed = within_1_step_pct >= (100.0 - tolerance_pct)
    #
    # 误差来源分析：
    # - 输入量化误差（x_fp32 → x_int8 的舍入误差）
    # - 权重量化误差（w_fp32 → w_int8 的舍入误差）
    # - 累加舍入误差（如果累加器位宽不足）
    # - Requantize 舍入误差（INT32 → INT8 的最后一次 round）
    # 通常最后的 requantize 贡献了主要误差。
    # =========================================================================
    raise NotImplementedError("TODO [Step 3.3]: Implement validate_against_float")


if __name__ == "__main__":
    print("=" * 60)
    print("  INT8 Reference Kernel Self-Test")
    print("=" * 60)

    np.random.seed(42)

    # Simple test: 1×1 conv (just a dot product)
    print("\n--- Simple test: Conv2d(2, 1, 1×1) on 1×1 input ---")
    # This is the simplest possible case: no spatial dims, just a dot product
    # input: [1, 2, 1, 1], weight: [1, 2, 1, 1]
    x_fp32 = np.array([[[[0.5]], [[1.0]]]], dtype=np.float32)  # [1, 2, 1, 1]
    w_fp32 = np.array([[[[0.3]], [[0.7]]]], dtype=np.float32)  # [1, 2, 1, 1]
    b_fp32 = np.array([0.1], dtype=np.float32)

    fp32_output = np.sum(x_fp32 * w_fp32) + b_fp32[0]
    print(f"  FP32 result: {fp32_output:.6f}")
    print(f"  (0.5*0.3 + 1.0*0.7 + 0.1 = {0.5*0.3 + 1.0*0.7 + 0.1})")

    # TODO: After implementing, quantize and run int8_conv2d_reference
    # and compare with fp32_output
    print("\n  (Implement int8_conv2d_reference first, then this test will work)")
