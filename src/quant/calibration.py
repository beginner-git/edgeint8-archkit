"""
Module: calibration.py
Part: B
Purpose: Activation calibration strategies — hand-written implementations.

Learning Goals:
- Understand why activation calibration is needed (vs weight quantization)
- Implement MinMax and Percentile calibration from scratch
- Understand the trade-off between range coverage and quantization resolution

Prerequisites:
- Step 2.1: quantize_utils.py implemented
- Understanding of neural network inference (activations are data-dependent)

为什么需要校准（Calibration）？
- 权重是固定的，可以直接计算 scale/zp
- 但激活（每层的输出）取决于输入数据，范围事先不知道
- 校准就是用一批代表性数据跑一遍网络，统计每层激活的范围
- 然后用这个范围来计算 scale/zp

校准方法的核心 trade-off：
- 范围太大（如用 min/max）：量化分辨率低，精度差
- 范围太小（如截断 outlier）：超出范围的值被 clip，引入 clipping error
- 好的校准方法要在两者之间找到平衡

Usage (self-test):
    python -m src.quant.calibration
"""

import numpy as np
from tabulate import tabulate

from src.quant import compute_scale_zp_symmetric


def calibrate_minmax(activation_batches):
    """
    MinMax calibration: use global min/max across all calibration batches.

    Args:
        activation_batches: List of numpy arrays (each is a batch of activations)

    Returns:
        range_min: float (global minimum)
        range_max: float (global maximum)

    MinMax 校准：
        range_min = min(所有校准数据的激活值)
        range_max = max(所有校准数据的激活值)

    优点：保证所有值都在量化范围内，不会有 clipping error
    缺点：如果有 outlier（极端值），会扩大范围，降低其他值的量化精度
          一个极端值就能 "毒害" 整个量化范围

    这是最简单的方法，也是 ORT 默认的 CalibrationMethod.MinMax。
    """
    # =========================================================================
    # TODO [Step 2.2]: Implement MinMax calibration
    #
    # Steps:
    #   global_min = float('inf')
    #   global_max = float('-inf')
    #   for batch in activation_batches:
    #       global_min = min(global_min, np.min(batch))
    #       global_max = max(global_max, np.max(batch))
    #   return global_min, global_max
    # =========================================================================
    global_min = np.min(activation_batches)
    global_max = np.max(activation_batches)
    for batch in activation_batches:
        global_min = min(global_min, np.min(batch))
        global_max = max(global_max, np.max(batch))
    return global_min, global_max

    raise NotImplementedError("TODO [Step 2.2]: Implement calibrate_minmax")


def calibrate_percentile(activation_batches, percentile=99.99):
    """
    Percentile calibration: clip outliers by using percentile values.

    Args:
        activation_batches: List of numpy arrays
        percentile: Percentile to use (e.g., 99.99 clips top/bottom 0.01%)

    Returns:
        range_min: float (lower percentile)
        range_max: float (upper percentile)

    Percentile 校准：
        把所有激活值拼在一起，用百分位数代替 min/max
        range_min = percentile(all_values, 100 - percentile)  # e.g., 0.01%
        range_max = percentile(all_values, percentile)        # e.g., 99.99%

    优点：对 outlier 鲁棒，不会因为极端值扩大范围
    缺点：超出 percentile 范围的值会被 clip 到边界（引入 clipping error）

    percentile 的选择：
        - 99.99%: 非常保守，几乎等于 MinMax
        - 99.9%:  稍微激进，clip 掉 0.1% 的极端值
        - 99%:    比较激进，适合分布有长尾的情况

    实际中，需要在"clipping error"和"quantization resolution"之间找平衡。
    """
    # =========================================================================
    # TODO [Step 2.2]: Implement Percentile calibration
    #
    # Steps:
    #   all_values = np.concatenate([b.flatten() for b in activation_batches])
    #   range_min = np.percentile(all_values, 100.0 - percentile)
    #   range_max = np.percentile(all_values, percentile)
    #   return range_min, range_max
    #
    # Note: For ReLU activations (all >= 0), range_min will be close to 0.
    # =========================================================================
    all_values = np.concatenate([b.flatten() for b in activation_batches])
    range_min = np.percentile(all_values, 100 - percentile)
    range_max = np.percentile(all_values, percentile)
    return range_min, range_max

    raise NotImplementedError("TODO [Step 2.2]: Implement calibrate_percentile")


def calibrate_entropy(activation_batches, num_bins=2048):
    """
    Entropy (KL-divergence) calibration: find the range that minimizes
    the KL divergence between original and quantized distributions.

    This is the method used by TensorRT. It's more sophisticated but
    also more compute-intensive.

    Args:
        activation_batches: List of numpy arrays
        num_bins: Number of histogram bins

    Returns:
        range_min: float
        range_max: float

    KL 散度校准（高级，可选）：
        1. 统计激活值的直方图
        2. 对不同的截断阈值，计算量化前后分布的 KL 散度
        3. 选择 KL 散度最小的阈值

    KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
    其中 P 是原始分布，Q 是量化后的分布

    这个方法比 MinMax 和 Percentile 更精确，但计算量也更大。
    TensorRT 默认使用这种方法。
    """
    # =========================================================================
    # TODO [Step 2.2] (Advanced, optional): Implement entropy calibration
    #
    # This is a stretch goal. The algorithm is:
    #   1. Compute histogram of all activation values with num_bins bins
    #   2. For each candidate threshold t (from num_bins/2 to num_bins):
    #      a. Clip histogram to [-t, t]
    #      b. Quantize the clipped histogram to 256 bins (INT8)
    #      c. Compute KL divergence between original and quantized histograms
    #   3. Choose the threshold t that minimizes KL divergence
    #
    # Reference: TensorRT developer guide on calibration
    # =========================================================================
    num_bits = 8
    num_qbins = 2 ** (num_bits - 1)
    all_values = np.abs(np.concatenate([b.flatten() for b in activation_batches]))
    abs_max = np.max(all_values)
    hist, bins = np.histogram(all_values, bins=num_bins, range=(0, abs_max))

    best_kl = float("inf")
    best_threshold_bin = num_bins

    for i in range(num_bins//2, num_bins):
        p = hist[:i].copy().astype(np.float32)
        p[i-1] += np.sum(hist[i:])

        # num_merged = int(i / 128)
        q = np.zeros(i, dtype=np.float32)
        for j in range(num_qbins):
            start = int(round(j * i / num_qbins))
            end = int(round((j + 1) * i / num_qbins))

            merged_count = np.sum(p[start:end])

            num_orig_bins = end - start
            if num_orig_bins > 0:
                nonzero = np.count_nonzero(p[start:end])
                if nonzero > 0:
                    q[start:end] = np.where(
                        p[start:end] > 0,
                        merged_count / nonzero, 0
                    )

        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum == 0 or q_sum == 0:
            continue
        p = p / p_sum
        q = q / q_sum

        mask = (p>0) & (q>0)
        kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))

        if kl < best_kl:
            best_kl = kl
            best_threshold_bin = i

    threshold =bins[best_threshold_bin + 1]

    return -threshold, threshold


    raise NotImplementedError("TODO [Step 2.2]: Implement calibrate_entropy (optional)")


def collect_layer_activations(model, data_loader, num_batches=10):
    """
    Collect intermediate layer activations using PyTorch hooks.

    Args:
        model: PyTorch model (in eval mode)
        data_loader: DataLoader with calibration data
        num_batches: Number of batches to collect

    Returns:
        activations: dict of {layer_name: [batch1, batch2, ...]}
    """
    # =========================================================================
    # TODO [Step 2.2]: Implement activation collection using hooks
    #
    # PyTorch hooks 允许你在不修改模型代码的情况下，捕获中间层的输出。
    #
    # Steps:
    #   import torch
    #   activations = {}
    #   hooks = []
    #
    #   def make_hook(name):
    #       def hook_fn(module, input, output):
    #           if name not in activations:
    #               activations[name] = []
    #           activations[name].append(output.detach().cpu().numpy())
    #       return hook_fn
    #
    #   # Register hooks on all Conv and Linear layers
    #   for name, module in model.named_modules():
    #       if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d,
    #                              torch.nn.Linear)):
    #           hooks.append(module.register_forward_hook(make_hook(name)))
    #
    #   # Run calibration data through the model
    #   model.eval()
    #   with torch.no_grad():
    #       for i, (data, _) in enumerate(data_loader):
    #           if i >= num_batches:
    #               break
    #           model(data)
    #
    #   # Remove hooks
    #   for h in hooks:
    #       h.remove()
    #
    #   return activations
    # =========================================================================
    import torch
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu().numpy())
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            model(data)

    for h in hooks:
        h.remove()

    return activations

    raise NotImplementedError("TODO [Step 2.2]: Implement collect_layer_activations")


def compare_calibration_methods(model, data_loader, num_batches=10):
    """
    Compare different calibration methods on the same data.

    Args:
        model: PyTorch model
        data_loader: DataLoader

    Returns:
        comparison: dict of {layer_name: {method_name: (range_min, range_max)}}
    """
    # =========================================================================
    # TODO [Step 2.2]: Compare MinMax vs Percentile calibration
    #
    # 1. Collect activations for all layers
    # 2. For each layer, compute:
    #    - MinMax range
    #    - Percentile(99.99%) range
    #    - Percentile(99.9%) range
    # 3. Print comparison table
    # 4. Compute quantization error for each method
    #
    # 这个对比帮助你理解：不同校准方法对哪些层影响最大？
    # =========================================================================
    from src.quant.quantize_utils import (
        compute_scale_zp_symmetric,
        quantize_tensor,
        dequantize_tensor,
        compute_quantization_error
    )

    comparison = {}
    all_rows = []

    model_activations = collect_layer_activations(model, data_loader, num_batches)
    for layer_name, activations in model_activations.items():
        comparison[layer_name] = {}
        all_data = np.concatenate([b.flatten() for b in activations])
        methods = {
            "MinMax": calibrate_minmax(activations),
            "Percentile_99.99": calibrate_percentile(activations, percentile=99.99),
            "Percentile_99.9": calibrate_percentile(activations, percentile=99.9),
        }

        for method_name, (x_min, x_max) in methods.items():
            comparison[layer_name][method_name] = (x_min, x_max)
            act_method = np.clip(all_data, x_min, x_max)

            scale, zero_point = compute_scale_zp_symmetric(tensor=act_method, num_bits=8)
            act_q = quantize_tensor(tensor=act_method, scale=scale, zero_point=zero_point)
            act_hat = dequantize_tensor(q_tensor=act_q, scale=scale, zero_point=zero_point)
            error = compute_quantization_error(original=all_data, reconstructed=act_hat)

            all_rows.append([
                layer_name,
                method_name,
                f"[{x_min:.4f}, {x_max:.4f}]",
                f"{error['mse']:.6f}",
                f"{error['max_error']:.4f}",
                f"{error['sqnr_db']:.1f}",
            ])
    print("\n" + tabulate(
        all_rows,
        headers=["Layer", "Method", "Range", "MSE", "MaxErr", "SQNR(dB)"],
        tablefmt="grid"
    ))

    return comparison

    raise NotImplementedError("TODO [Step 2.2]: Implement compare_calibration_methods")


if __name__ == "__main__":
    print("=" * 60)
    print("  Calibration Self-Test")
    print("=" * 60)

    np.random.seed(42)

    # Simulate ReLU activations with a few outliers
    normal_data = np.abs(np.random.randn(10, 64, 8, 8).astype(np.float32))
    # Add some outliers
    normal_data[0, 0, 0, 0] = 50.0  # Extreme outlier
    normal_data[5, 10, 3, 3] = 30.0

    batches = [normal_data[i:i+1] for i in range(10)]

    print("\n--- MinMax Calibration ---")
    rmin, rmax = calibrate_minmax(batches)
    print(f"  Range: [{rmin:.4f}, {rmax:.4f}]")
    print(f"  -> Outlier at 50.0 expands the range significantly!")

    print("\n--- Percentile(99.99%) Calibration ---")
    rmin_p, rmax_p = calibrate_percentile(batches, percentile=99.99)
    print(f"  Range: [{rmin_p:.4f}, {rmax_p:.4f}]")

    print("\n--- Percentile(99.9%) Calibration ---")
    rmin_p2, rmax_p2 = calibrate_percentile(batches, percentile=99.9)
    print(f"  Range: [{rmin_p2:.4f}, {rmax_p2:.4f}]")
    print(f"  -> Tighter range = better resolution for most values")
    print(f"     but outliers get clipped to the range boundary")
