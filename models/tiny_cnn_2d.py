"""
Module: tiny_cnn_2d.py
Part: A (Workload-2)
Purpose: Define a tiny 2D CNN for CIFAR-10 image classification.

Learning Goals:
- Understand 2D convolution and its parameter count
- Use BatchNorm for training stability
- Design a model small enough for quantization study yet representative for architecture analysis

Prerequisites:
- Basic PyTorch nn.Module knowledge
- Understanding of 2D convolution: output_size = (input - kernel + 2*padding) / stride + 1

2D CNN 是计算机视觉的基础。本模型用于 Workload-2（CIFAR-10 分类），
它比 Workload-1 更"典型"，因为 Conv2d/GEMM 层是 NPU/加速器最常处理的算子。
后续会用它来做量化对比和架构映射（SCALE-Sim）。

Architecture:
    Input [B, 3, 32, 32]  (CIFAR-10: 3-channel, 32x32 RGB images)
      ↓
    Conv2d(3, 32, 3, padding=1) + BN + ReLU   → [B, 32, 32, 32]
      ↓ MaxPool2d(2)                           → [B, 32, 16, 16]
    Conv2d(32, 64, 3, padding=1) + BN + ReLU   → [B, 64, 16, 16]
      ↓ MaxPool2d(2)                           → [B, 64, 8, 8]
    Conv2d(64, 128, 3, padding=1) + BN + ReLU  → [B, 128, 8, 8]
      ↓ AdaptiveAvgPool2d(1)                   → [B, 128, 1, 1]
      ↓ Flatten                                → [B, 128]
    Linear(128, num_classes)                    → [B, 10]
"""

import torch
import torch.nn as nn


class TinyCNN2D(nn.Module):
    """
    A tiny 2D CNN for CIFAR-10.

    为什么这样设计：
    - 3 层 Conv2d + BatchNorm + ReLU 是经典的 CNN building block
    - BatchNorm 加速训练收敛，在量化时也有特殊处理（BN folding）
    - MaxPool2d 降低空间分辨率，减少计算量
    - 通道数 32→64→128 逐步增加，提取更高级的特征
    - AdaptiveAvgPool2d(1) 取代全连接展平，支持任意输入尺寸

    这个模型大约有 ~120K 参数，足够小以快速训练，
    但包含了典型的 Conv-BN-ReLU 结构，非常适合量化分析。

    关于 BatchNorm 和量化的关系：
    在推理（eval mode）时，BN 可以被"折叠"（fold）到 Conv 中，
    变成一个等效的 Conv 层。ORT 量化工具会自动处理这一点。
    """

    def __init__(self, in_channels=3, num_classes=10):
        """
        Args:
            in_channels: Number of input channels (3 for CIFAR-10 RGB)
            num_classes: Number of classes (10 for CIFAR-10)
        """
        super().__init__()

        # =====================================================================
        # TODO [Step 1.2]: Implement the network layers
        #
        # Build a 3-block 2D CNN. Each block = Conv2d + BatchNorm2d + ReLU:
        #
        #   Block 1:
        #     conv1: Conv2d(in_channels, 32, kernel_size=3, padding=1)
        #     bn1:   BatchNorm2d(32)
        #     pool1: MaxPool2d(kernel_size=2, stride=2)
        #
        #   Block 2:
        #     conv2: Conv2d(32, 64, kernel_size=3, padding=1)
        #     bn2:   BatchNorm2d(64)
        #     pool2: MaxPool2d(kernel_size=2, stride=2)
        #
        #   Block 3:
        #     conv3: Conv2d(64, 128, kernel_size=3, padding=1)
        #     bn3:   BatchNorm2d(128)
        #
        #   Head:
        #     global_pool: AdaptiveAvgPool2d(output_size=1)
        #     fc: Linear(128, num_classes)
        #     relu: ReLU (shared)
        #
        # 为什么用 kernel_size=3？
        # - 3×3 是现代 CNN 最常用的 kernel，VGG 论文证明了堆叠 3×3 比用大 kernel 更高效
        # - 两个 3×3 的感受野 = 一个 5×5，但参数更少
        # - 对 NPU 硬件来说，3×3 也是最常优化的 kernel size
        #
        # 为什么用 BatchNorm？
        # - 加速训练收敛
        # - 推理时可以折叠到 Conv 里（BN folding），不增加推理成本
        # - 量化时需要处理 BN folding 后的权重分布
        #
        # Hint: Use nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
        #       nn.AdaptiveAvgPool2d, nn.Linear
        # =====================================================================
        raise NotImplementedError("TODO [Step 1.2]: Define network layers")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
               Example: [32, 3, 32, 32] for CIFAR-10

        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # =====================================================================
        # TODO [Step 1.2]: Implement the forward pass
        #
        # Sequence (trace the shapes!):
        #   x = relu(bn1(conv1(x)))   # [B, 3, 32, 32]  -> [B, 32, 32, 32]
        #   x = pool1(x)              # [B, 32, 32, 32]  -> [B, 32, 16, 16]
        #   x = relu(bn2(conv2(x)))   # [B, 32, 16, 16]  -> [B, 64, 16, 16]
        #   x = pool2(x)              # [B, 64, 16, 16]  -> [B, 64, 8, 8]
        #   x = relu(bn3(conv3(x)))   # [B, 64, 8, 8]   -> [B, 128, 8, 8]
        #   x = global_pool(x)        # [B, 128, 8, 8]   -> [B, 128, 1, 1]
        #   x = x.flatten(1)          # [B, 128, 1, 1]   -> [B, 128]
        #   x = fc(x)                 # [B, 128]         -> [B, num_classes]
        #
        # 练习：计算每一层的 FLOPs（乘加次数）。
        # 例如 Conv2d(3, 32, 3, padding=1) on 32×32 input:
        #   FLOPs = 2 × Cout × Cin × K × K × Hout × Wout
        #         = 2 × 32 × 3 × 3 × 3 × 32 × 32 = 1,769,472
        # =====================================================================
        raise NotImplementedError("TODO [Step 1.2]: Implement forward pass")


def get_tiny_cnn_2d(num_classes=10, in_channels=3):
    """Factory function to create a TinyCNN2D model."""
    return TinyCNN2D(in_channels=in_channels, num_classes=num_classes)


# =============================================================================
# Self-test: Run this file directly to verify the model
# Usage: python -m models.tiny_cnn_2d
# =============================================================================
if __name__ == "__main__":
    model = get_tiny_cnn_2d(num_classes=10, in_channels=3)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with CIFAR-10 shaped input
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
    print("Forward pass OK!")
