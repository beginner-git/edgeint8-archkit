"""
Module: train.py
Part: A (prerequisite for all subsequent parts)
Purpose: Train both workload models and save checkpoints.

Learning Goals:
- Standard PyTorch training loop
- Evaluation and checkpointing
- Understanding that for this project, high accuracy is NOT the goal —
  we just need a reasonably trained model to study quantization effects

Prerequisites:
- models/signal_cnn_1d.py and models/tiny_cnn_2d.py implemented
- src/utils/data.py implemented (data loaders)

训练脚本是整个项目的第一步。训练好模型后，才能做 ONNX 导出、量化、定点分析。
注意：本项目不追求 SOTA，5-10 epochs 足够。重点是拿到一个"能用"的权重。

Usage:
    python -m models.train --workload 1d --epochs 10
    python -m models.train --workload 2d --epochs 10
    python -m models.train --workload all --epochs 10
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.signal_cnn_1d import get_signal_cnn_1d
from models.tiny_cnn_2d import get_tiny_cnn_2d
from src.utils.data import get_cifar10_loaders, get_synthetic_signal_loaders
from src.utils.helpers import set_seed, ensure_dir


def train_one_epoch(model, train_loader, criterion, optimizer, device="cpu"):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss: Average training loss for this epoch
        accuracy: Training accuracy for this epoch
    """
    # =========================================================================
    # TODO [Step 1.4]: Implement the training loop for one epoch
    #
    # Standard PyTorch training pattern:
    #   model.train()
    #   for batch_idx, (data, target) in enumerate(train_loader):
    #       data, target = data.to(device), target.to(device)
    #       optimizer.zero_grad()
    #       output = model(data)
    #       loss = criterion(output, target)
    #       loss.backward()
    #       optimizer.step()
    #       # Track running loss and correct predictions
    #
    # Return average loss and accuracy.
    #
    # 为什么用 CrossEntropyLoss？
    # - 它内部包含 LogSoftmax + NLLLoss，是多分类任务的标准选择
    # - 输入是 raw logits（model 输出），不需要先过 softmax
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.4]: Implement train_one_epoch")


def evaluate(model, test_loader, criterion, device="cpu"):
    """
    Evaluate the model on test data.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss: Average test loss
        accuracy: Test accuracy (0.0 ~ 1.0)
    """
    # =========================================================================
    # TODO [Step 1.4]: Implement the evaluation function
    #
    # Key differences from training:
    #   model.eval()            # Switch to eval mode (affects BN, Dropout)
    #   with torch.no_grad():  # Disable gradient computation (saves memory)
    #       for data, target in test_loader:
    #           output = model(data)
    #           loss = criterion(output, target)
    #           pred = output.argmax(dim=1)
    #           correct += pred.eq(target).sum().item()
    #
    # 为什么需要 model.eval()？
    # - BatchNorm 在 eval 模式下使用 running mean/var 而不是 batch 统计量
    # - Dropout 在 eval 模式下不丢弃任何神经元
    # - 量化和导出 ONNX 时也必须先 eval()
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.4]: Implement evaluate")


def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3,
                device="cpu", save_path=None):
    """
    Full training loop.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
        save_path: Path to save the best model checkpoint (.pth)

    Returns:
        model: Trained model
        history: Dict with 'train_loss', 'test_loss', 'test_acc' lists
    """
    # =========================================================================
    # TODO [Step 1.4]: Implement the full training loop
    #
    # Steps:
    #   1. Define criterion (nn.CrossEntropyLoss) and optimizer (Adam)
    #   2. For each epoch:
    #      a. Call train_one_epoch()
    #      b. Call evaluate()
    #      c. Print epoch summary
    #      d. Save model if test accuracy improved (best model checkpoint)
    #   3. Return model and training history
    #
    # 为什么用 Adam optimizer？
    # - Adam 是最常用的自适应优化器，对学习率不太敏感
    # - 对小模型和小数据集来说，Adam 通常比 SGD 更快收敛
    # - lr=1e-3 是 Adam 的经典默认值
    #
    # Hint: Use torch.save(model.state_dict(), save_path) to save
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.4]: Implement train_model")


# =============================================================================
# Main: Train both workloads from command line
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EdgeINT8-ArchKit models")
    parser.add_argument("--workload", choices=["1d", "2d", "all"], default="all",
                        help="Which workload to train")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu"
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    ensure_dir(model_dir)

    # =========================================================================
    # TODO [Step 1.4]: Implement the main training script
    #
    # For workload "1d" or "all":
    #   1. model = get_signal_cnn_1d(num_classes=5)
    #   2. train_loader, test_loader = get_synthetic_signal_loaders(batch_size)
    #   3. train_model(model, ..., save_path="models/signal_cnn_1d.pth")
    #
    # For workload "2d" or "all":
    #   1. model = get_tiny_cnn_2d(num_classes=10, in_channels=3)
    #   2. train_loader, test_loader = get_cifar10_loaders(batch_size)
    #   3. train_model(model, ..., save_path="models/tiny_cnn_2d.pth")
    #
    # Print summary after training (params count, final accuracy)
    # =========================================================================
    print("TODO: Implement training main script")
