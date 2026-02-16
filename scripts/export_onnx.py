"""
Module: export_onnx.py
Part: A
Purpose: Export trained PyTorch models to ONNX format.

Learning Goals:
- Understand what ONNX is and why it matters for deployment
- Learn torch.onnx.export parameters
- Verify ONNX model correctness by comparing outputs

Prerequisites:
- Trained model checkpoints (.pth files) from models/train.py

什么是 ONNX？
- Open Neural Network Exchange，开放的神经网络交换格式
- 让你可以在 PyTorch 中训练，然后在 ORT/TensorRT/TFLite 等推理引擎中运行
- 是 "训练框架" 和 "推理引擎" 之间的桥梁
- ONNX 文件是一个计算图（graph），包含算子（op）和权重（tensor）

为什么需要导出到 ONNX？
- PyTorch 的动态图不适合部署优化
- ONNX Runtime 可以做图优化（算子融合、常量折叠等）
- 量化工具（ORT quantize_static）需要 ONNX 模型作为输入
- 跨平台：同一个 ONNX 文件可以在 CPU/GPU/NPU 上运行

Usage:
    python scripts/export_onnx.py --workload 1d
    python scripts/export_onnx.py --workload 2d
    python scripts/export_onnx.py --workload all
"""

import os
import sys
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.signal_cnn_1d import get_signal_cnn_1d
from models.tiny_cnn_2d import get_tiny_cnn_2d
from src.utils.helpers import set_seed


def export_to_onnx(model, dummy_input, output_path, opset_version=13,
                   input_names=None, output_names=None):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model (must be in eval mode)
        dummy_input: Example input tensor with correct shape
        output_path: Path to save the .onnx file
        opset_version: ONNX opset version (13 is widely supported)
        input_names: List of input tensor names (for readability)
        output_names: List of output tensor names

    Returns:
        output_path: Path to the saved ONNX file
    """
    # =========================================================================
    # TODO [Step 1.5]: Implement ONNX export
    #
    # Steps:
    #   1. Ensure model is in eval mode: model.eval()
    #   2. Call torch.onnx.export:
    #      torch.onnx.export(
    #          model,
    #          dummy_input,
    #          output_path,
    #          opset_version=opset_version,
    #          input_names=input_names or ["input"],
    #          output_names=output_names or ["output"],
    #          dynamic_axes={"input": {0: "batch_size"},
    #                        "output": {0: "batch_size"}},
    #      )
    #   3. Print file size
    #
    # 关键参数解释：
    # - opset_version: ONNX 算子集版本。不同版本支持的算子不同。
    #   opset 13 支持大部分常用算子，是比较安全的选择。
    # - dynamic_axes: 指定哪些维度是动态的。这里 batch_size 维度是动态的，
    #   意味着导出后的模型可以接受任意 batch size 的输入。
    # - input_names/output_names: 给输入输出命名，方便后续引用。
    #   在 ONNX Runtime 中推理时需要用 input_names 来传入数据。
    #
    # Hint: torch.onnx.export 内部会用 dummy_input 跑一次 forward，
    #       追踪（trace）所有操作，生成 ONNX 计算图。
    # =========================================================================
    model.eval()

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    dynamic_axes = {input_names[0]: {0: "batch_size"},
                    output_names[0]: {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model exported to: {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def verify_onnx(onnx_path, model, dummy_input, atol=1e-5):
    """
    Verify ONNX model by comparing its output with PyTorch model output.

    Args:
        onnx_path: Path to the ONNX file
        model: Original PyTorch model
        dummy_input: Same input used for export
        atol: Absolute tolerance for comparison

    Returns:
        True if outputs match within tolerance
    """
    # =========================================================================
    # TODO [Step 1.5]: Implement ONNX verification
    #
    # Steps:
    #   1. Check model validity:
    #      import onnx
    #      onnx_model = onnx.load(onnx_path)
    #      onnx.checker.check_model(onnx_model)
    #
    #   2. Run PyTorch inference:
    #      model.eval()
    #      with torch.no_grad():
    #          pt_output = model(dummy_input).numpy()
    #
    #   3. Run ONNX Runtime inference:
    #      import onnxruntime as ort
    #      session = ort.InferenceSession(onnx_path)
    #      input_name = session.get_inputs()[0].name
    #      ort_output = session.run(None, {input_name: dummy_input.numpy()})[0]
    #
    #   4. Compare outputs:
    #      np.testing.assert_allclose(pt_output, ort_output, atol=atol)
    #
    # 为什么要做这个验证？
    # - 确保 ONNX 导出没有引入数值误差
    # - 有些 PyTorch 操作在 ONNX 中的实现可能略有不同
    # - 这是一个良好的工程实践：每次转换后都验证一致性
    # =========================================================================
    try:
        import onnx
        import onnxruntime as ort

        # 1. Check model validity
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model check passed: {onnx_path}")

        # 2. Run PyTorch inference
        model.eval()
        with torch.no_grad():
            pt_output = model(dummy_input).numpy()

        # 3. Run ONNX Runtime inference
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        ort_output = session.run(None, {input_name: dummy_input.numpy()})[0]

        # 4. Compare outputs
        np.testing.assert_allclose(pt_output, ort_output, atol=atol)
        print(f"Verification OK: PyTorch vs ONNX Runtime outputs match (atol={atol})")
        return True

    except Exception as e:
        print(f"Verification FAIL: {e}")
        return False


def get_model_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--workload", choices=["1d", "2d", "all"], default="all")
    parser.add_argument("--models-dir", default="models",
                        help="Directory containing .pth checkpoints")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version")
    args = parser.parse_args()

    set_seed(42)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # =========================================================================
    # TODO [Step 1.5]: Implement the main export script
    #
    # For workload "1d" or "all":
    #   1. Load model: model = get_signal_cnn_1d(); model.load_state_dict(...)
    #   2. Create dummy input: torch.randn(1, 1, 128)
    #   3. Export: export_to_onnx(model, dummy, "models/signal_cnn_1d.onnx")
    #   4. Verify: verify_onnx("models/signal_cnn_1d.onnx", model, dummy)
    #
    # For workload "2d" or "all":
    #   1. Load model: model = get_tiny_cnn_2d(); model.load_state_dict(...)
    #   2. Create dummy input: torch.randn(1, 3, 32, 32)
    #   3. Export: export_to_onnx(model, dummy, "models/tiny_cnn_2d.onnx")
    #   4. Verify: verify_onnx("models/tiny_cnn_2d.onnx", model, dummy)
    #
    # Print summary: model name, ONNX file size, verification status
    # =========================================================================
    results = []

    # --- Workload 1D: Signal CNN ---
    if args.workload in ("1d", "all"):
        pth_path = os.path.join(project_root, args.models_dir, "signal_cnn_1d.pth")
        onnx_path = os.path.join(project_root, args.models_dir, "signal_cnn_1d.onnx")

        if not os.path.exists(pth_path):
            print(f"WARNING: {pth_path} not found, skipping 1D model export.")
        else:
            print("\n" + "=" * 60)
            print("Exporting Signal CNN 1D")
            print("=" * 60)
            model = get_signal_cnn_1d()
            model.load_state_dict(torch.load(pth_path, map_location="cpu"))
            model.eval()

            dummy = torch.randn(1, 1, 128)
            export_to_onnx(model, dummy, onnx_path, opset_version=args.opset)
            verified = verify_onnx(onnx_path, model, dummy)
            results.append(("signal_cnn_1d", get_model_size_mb(onnx_path), verified))

    # --- Workload 2D: Tiny CNN ---
    if args.workload in ("2d", "all"):
        pth_path = os.path.join(project_root, args.models_dir, "tiny_cnn_2d.pth")
        onnx_path = os.path.join(project_root, args.models_dir, "tiny_cnn_2d.onnx")

        if not os.path.exists(pth_path):
            print(f"WARNING: {pth_path} not found, skipping 2D model export.")
        else:
            print("\n" + "=" * 60)
            print("Exporting Tiny CNN 2D")
            print("=" * 60)
            model = get_tiny_cnn_2d()
            model.load_state_dict(torch.load(pth_path, map_location="cpu"))
            model.eval()

            dummy = torch.randn(1, 3, 32, 32)
            export_to_onnx(model, dummy, onnx_path, opset_version=args.opset)
            verified = verify_onnx(onnx_path, model, dummy)
            results.append(("tiny_cnn_2d", get_model_size_mb(onnx_path), verified))

    # --- Summary ---
    if results:
        print("\n" + "=" * 60)
        print("Export Summary")
        print("=" * 60)
        for name, size_mb, verified in results:
            status = "OK" if verified else "FAIL"
            print(f"  {name:20s}  {size_mb:.2f} MB  Verification: {status}")
