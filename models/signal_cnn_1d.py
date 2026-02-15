"""
Module: signal_cnn_1d.py
Part: A (Workload-1)
Purpose: Define a small 1D CNN for synthetic signal classification.

Learning Goals:
- Understand how 1D convolution works on sequential/signal data
- Design a minimal model suitable for quantization study
- Trace tensor shapes through the network

Prerequisites:
- Basic PyTorch nn.Module knowledge
- Understanding of 1D convolution: output_len = (input_len - kernel_size + 2*padding) / stride + 1

1D CNN 适用于时间序列、传感器信号等一维数据。与 2D CNN 的区别在于卷积核
只在一个维度上滑动。本模型用于 Workload-1（合成信号分类），后续会对它做
量化、定点分析和架构映射。

Architecture:
    Input [B, 1, 128]
      ↓
    Conv1d(1, 16, kernel_size=7, padding=3)  → [B, 16, 128]
      ↓ ReLU
    Conv1d(16, 32, kernel_size=5, padding=2) → [B, 32, 128]
      ↓ ReLU
    Conv1d(32, 64, kernel_size=3, padding=1) → [B, 64, 128]
      ↓ ReLU
    AdaptiveAvgPool1d(1)                     → [B, 64, 1]
      ↓ Flatten
    Linear(64, num_classes)                  → [B, num_classes]
"""

import torch
import torch.nn as nn


class SignalCNN1D(nn.Module):
    """
    A small 1D CNN for signal classification.

    为什么这样设计：
    - 3 层 Conv1d 逐步增加通道数（1→16→32→64），提取从低级到高级的特征
    - 使用 padding 保持序列长度不变，方便跟踪 shape
    - AdaptiveAvgPool1d(1) 将任意长度的特征压缩为固定维度
    - 最后一个 Linear 层做分类

    这个模型大约有 ~25K 参数，非常适合做量化实验。
    """

    def __init__(self, in_channels=1, num_classes=5):
        """
        Args:
            in_channels: Number of input channels (1 for single-channel signal)
            num_classes: Number of classification categories
        """
        super().__init__()

        # =====================================================================
        # TODO [Step 1.1]: Implement the network layers
        #
        # Build a 3-layer 1D CNN with the following structure:
        #   conv1: Conv1d(in_channels, 16, kernel_size=7, padding=3)
        #   conv2: Conv1d(16, 32, kernel_size=5, padding=2)
        #   conv3: Conv1d(32, 64, kernel_size=3, padding=1)
        #   pool:  AdaptiveAvgPool1d(output_size=1)
        #   fc:    Linear(64, num_classes)
        #   relu:  ReLU (shared activation, 可复用同一个)
        #
        # 为什么选这些 kernel size？
        # - 第一层 kernel=7 捕获较宽的时域模式（如频率特征）
        # - 后续层 kernel 逐渐变小，因为感受野已经通过堆叠扩大了
        # - padding = (kernel_size - 1) // 2 保持序列长度不变（same padding）
        #
        # Hint: Use nn.Conv1d, nn.ReLU, nn.AdaptiveAvgPool1d, nn.Linear
        # =====================================================================
        raise NotImplementedError("TODO [Step 1.1]: Define network layers")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, in_channels, seq_len]
               Example: [32, 1, 128] for batch=32, single channel, length=128

        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # =====================================================================
        # TODO [Step 1.1]: Implement the forward pass
        #
        # Sequence:
        #   x = relu(conv1(x))   # [B, 1, 128] -> [B, 16, 128]
        #   x = relu(conv2(x))   # [B, 16, 128] -> [B, 32, 128]
        #   x = relu(conv3(x))   # [B, 32, 128] -> [B, 64, 128]
        #   x = pool(x)          # [B, 64, 128] -> [B, 64, 1]
        #   x = x.flatten(1)     # [B, 64, 1]   -> [B, 64]
        #   x = fc(x)            # [B, 64]      -> [B, num_classes]
        #
        # 练习：在每一步后加 print(x.shape) 验证 shape 变化是否符合预期。
        # =====================================================================
        raise NotImplementedError("TODO [Step 1.1]: Implement forward pass")


def get_signal_cnn_1d(num_classes=5):
    """Factory function to create a SignalCNN1D model."""
    return SignalCNN1D(in_channels=1, num_classes=num_classes)


# =============================================================================
# Self-test: Run this file directly to verify the model
# Usage: python -m models.signal_cnn_1d
# =============================================================================
if __name__ == "__main__":
    model = get_signal_cnn_1d(num_classes=5)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 1, 128)  # [batch=1, channels=1, seq_len=128]
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 5), f"Expected (1, 5), got {output.shape}"
    print("Forward pass OK!")
