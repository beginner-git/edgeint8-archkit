"""
Module: analysis.py
Part: B
Purpose: Per-layer quantization error analysis and sensitivity study.

Learning Goals:
- Understand which layers are most sensitive to quantization
- Learn how to locate the source of accuracy degradation
- Build intuition for mixed-precision quantization decisions

Prerequisites:
- Step 2.1: quantize_utils.py implemented
- Step 2.2: calibration.py implemented
- Trained model from Part A

为什么需要逐层分析？
- 整体精度下降告诉你 "量化伤了精度"，但不告诉你 "伤在哪里"
- 逐层分析帮你定位：是第一层？最后一层？还是某个特定的 Conv 层？
- 这直接影响你的量化决策：敏感层可以保留 FP32 或用更高精度

常见规律：
- 第一层和最后一层通常最敏感（输入分布变化大，输出直接影响预测）
- 深层 Conv 的激活分布通常更集中，量化友好
- 有 skip connection 的网络通常更鲁棒（误差被"分摊"了）

Usage:
    python -m src.quant.analysis
"""

import numpy as np

from src.quant import compute_scale_zp_symmetric, quantize_tensor


def analyze_per_layer_sensitivity(model, test_loader, num_batches=10):
    """
    Analyze quantization sensitivity of each layer independently.

    Method: For each quantizable layer:
      1. Quantize only that one layer (keep others in FP32)
      2. Measure accuracy drop
      3. Rank layers by sensitivity

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        num_batches: Number of test batches to evaluate

    Returns:
        sensitivity: dict of {layer_name: accuracy_drop}
    """
    # =========================================================================
    # TODO [Step 2.5]: Implement per-layer sensitivity analysis
    #
    # This is a "leave-one-out" analysis:
    #
    #   import torch
    #   import copy
    #
    #   # Get baseline accuracy (all FP32)
    #   baseline_acc = evaluate_model(model, test_loader)
    #
    #   sensitivity = {}
    #   for name, module in model.named_modules():
    #       if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
    #           # Quantize only this layer's weights
    #           model_copy = copy.deepcopy(model)
    #           quantize_single_layer(model_copy, name)  # You need to implement this
    #           acc = evaluate_model(model_copy, test_loader)
    #           sensitivity[name] = baseline_acc - acc  # Accuracy drop
    #
    #   # Sort by sensitivity (most sensitive first)
    #   sensitivity = dict(sorted(sensitivity.items(), key=lambda x: -x[1]))
    #
    # A simpler approach: just compute the quantization error (MSE) of each
    # layer's weights and activations, without actually re-evaluating accuracy.
    #
    # 简单版本：不重跑推理，只计算每层权重的量化误差
    #   for name, param in model.named_parameters():
    #       if 'weight' in name:
    #           scale, zp = compute_scale_zp_symmetric(param.data.numpy())
    #           q = quantize_tensor(param.data.numpy(), scale, zp)
    #           deq = dequantize_tensor(q, scale, zp)
    #           error = compute_quantization_error(param.data.numpy(), deq)
    #           sensitivity[name] = error['mse']
    # =========================================================================
    from src.quant.quantize_utils import (
        compute_scale_zp_symmetric,
        quantize_tensor,
        dequantize_tensor,
        compute_quantization_error
    )

    sensitivity = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_np = param.data.cpu().numpy()
            scale, zp = compute_scale_zp_symmetric(weight_np)
            q = quantize_tensor(weight_np, scale, zp)
            deq = dequantize_tensor(q, scale, zp)
            error = compute_quantization_error(weight_np, deq)
            sensitivity[name] = error["mse"]

    sensitivity = dict(sorted(sensitivity.items(), key=lambda x: x[-1]))
    return sensitivity

    raise NotImplementedError("TODO [Step 2.5]: Implement per-layer sensitivity analysis")


def generate_error_report(model, data_loader, num_batches=5):
    """
    Generate a comprehensive error report for all layers.

    Args:
        model: PyTorch model
        data_loader: DataLoader

    Returns:
        report: dict of {layer_name: {
            'weight_mse': float,
            'weight_sqnr': float,
            'activation_mse': float,
            'activation_sqnr': float,
            'weight_range': (min, max),
            'activation_range': (min, max),
        }}
    """
    # =========================================================================
    # TODO [Step 2.5]: Implement comprehensive error report
    #
    # For each layer:
    #   1. Compute weight quantization error (symmetric, per-channel)
    #   2. Collect activation data and compute activation quantization error
    #   3. Record ranges, MSE, SQNR
    #   4. Return as structured dict
    #
    # Use calibration.collect_layer_activations() to get activations
    # Use quantize_utils.compute_quantization_error() for error stats
    #
    # Output format should be printable as a table:
    #   Layer | Weight MSE | Weight SQNR | Act MSE | Act SQNR | Sensitivity
    # =========================================================================
    from src.quant.quantize_utils import (
        compute_scale_zp_symmetric,
        quantize_tensor,
        dequantize_tensor,
        compute_quantization_error
    )
    from src.quant.calibration import collect_layer_activations

    report = {}

    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.data.cpu().numpy()
            scale, zp = compute_scale_zp_symmetric(w)
            q = quantize_tensor(w, scale, zp)
            deq = dequantize_tensor(q, scale, zp)
            error = compute_quantization_error(w, deq)
            report[name] = {
                'weight_mse': error["mse"],
                'weight_sqnr': error["sqnr_db"],
                'weight_range': (float(w.min()), float(w.max())),
            }

    activations = collect_layer_activations(model, data_loader, num_batches=num_batches)
    for layer_name, batches in activations.items():
        all_data = np.concatenate([b.flatten() for b in batches])
        scale, zp = compute_scale_zp_symmetric(all_data)
        q = quantize_tensor(all_data, scale, zp)
        deq = dequantize_tensor(q, scale, zp)
        error = compute_quantization_error(all_data, deq)

        wight_key = layer_name + ".weight"
        if wight_key not in report:
            report[wight_key]['activation_mse'] = error["mse"]
            report[wight_key]['activation_sqnr'] = error["sqnr_db"]
            report[wight_key]['weight_range'] = (float(all_data.min()), float(all_data.max()))

    return report

    raise NotImplementedError("TODO [Step 2.5]: Implement generate_error_report")


def identify_sensitive_layers(error_report, threshold_sqnr=40.0):
    """
    Identify layers that are sensitive to quantization.

    Args:
        error_report: Output from generate_error_report
        threshold_sqnr: Layers with SQNR below this are flagged (default 40 dB)

    Returns:
        sensitive_layers: List of layer names with SQNR below threshold
    """
    # =========================================================================
    # TODO [Step 2.5]: Identify sensitive layers
    #
    # A layer is "sensitive" if its quantization SQNR is low:
    #   - SQNR < 40 dB: potentially problematic
    #   - SQNR < 30 dB: likely to cause accuracy degradation
    #
    # For reference, ideal 8-bit SQNR ≈ 48 dB.
    # If a layer's SQNR is much lower than 48 dB, the issue might be:
    #   - Weight distribution has large outliers
    #   - Activation distribution is very non-uniform
    #   - Layer has very few parameters (quantization "bins" are too coarse)
    #
    # 对于敏感层，可能的应对策略：
    # - 使用 per-channel 量化（权重）
    # - 使用更好的校准方法（Percentile/Entropy）
    # - 保留 FP32（mixed precision）
    # - 使用 QAT（量化感知训练）重新训练
    # =========================================================================
    threshold_sqnr = 40

    sensitive_layers = []
    for layer_name, errors in error_report.items():
        if errors.get('weight_sqnr', 100) < threshold_sqnr:
            sensitive_layers.append(layer_name)

    return sensitive_layers

    raise NotImplementedError("TODO [Step 2.5]: Implement identify_sensitive_layers")


if __name__ == "__main__":
    # print("Per-layer analysis requires a trained model.")
    # print("Run after completing Step 1.4 (training).")
    # print("\nUsage:")
    # print("  from src.quant.analysis import analyze_per_layer_sensitivity")
    # print("  sensitivity = analyze_per_layer_sensitivity(model, test_loader)")

    import torch
    from models import get_tiny_cnn_2d
    from src.utils.data import get_cifar10_loaders
    # from src.quant.analysis import analyze_per_layer_sensitivity, generate_error_report, identify_sensitive_layers

    model = get_tiny_cnn_2d()
    model.load_state_dict(torch.load('models/tiny_cnn_2d.pth', map_location='cpu'))
    model.eval()

    _, test_loader = get_cifar10_loaders()

    # 1. 敏感度排名
    print('=== Sensitivity ===')
    sens = analyze_per_layer_sensitivity(model, test_loader)
    for name, mse in sens.items():
        print(f'  {name}: MSE={mse:.6f}')

    # 2. 完整报告
    print('\n=== Error Report ===')
    report = generate_error_report(model, test_loader)
    for name, errors in report.items():
        print(f'  {name}: SQNR={errors.get("weight_sqnr", "N/A"):.1f} dB')

    # 3. 敏感层
    print('\n=== Sensitive Layers (SQNR < 40 dB) ===')
    sensitive = identify_sensitive_layers(report)
    print(f'  {sensitive if sensitive else "None"}')
