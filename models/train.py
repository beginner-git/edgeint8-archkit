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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")

        if test_acc > best_acc and save_path is not None:
            best_acc = test_acc
            ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (acc={best_acc:.4f}) to {save_path}")

    return model, history


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    if args.workload in ("1d", "all"):
        print("=" * 60)
        print("  Training Workload-1: 1D Signal CNN")
        print("=" * 60)
        model_1d = get_signal_cnn_1d(num_classes=5)
        train_loader_1d, test_loader_1d = get_synthetic_signal_loaders(batch_size=args.batch_size)
        save_path_1d = os.path.join(model_dir, "signal_cnn_1d.pth")
        model_1d, history_1d = train_model(
            model_1d, train_loader_1d, test_loader_1d,
            epochs=args.epochs, lr=args.lr, device=device,
            save_path=save_path_1d,
        )
        from src.utils.helpers import count_parameters
        print(f"\n1D CNN Summary:")
        print(f"  Parameters:     {count_parameters(model_1d):,}")
        print(f"  Final Test Acc: {history_1d['test_acc'][-1]:.4f}")
        print()

    if args.workload in ("2d", "all"):
        print("=" * 60)
        print("  Training Workload-2: 2D Tiny CNN (CIFAR-10)")
        print("=" * 60)
        model_2d = get_tiny_cnn_2d(num_classes=10, in_channels=3)
        train_loader_2d, test_loader_2d = get_cifar10_loaders(batch_size=args.batch_size)
        save_path_2d = os.path.join(model_dir, "tiny_cnn_2d.pth")
        model_2d, history_2d = train_model(
            model_2d, train_loader_2d, test_loader_2d,
            epochs=args.epochs, lr=args.lr, device=device,
            save_path=save_path_2d,
        )
        from src.utils.helpers import count_parameters
        print(f"\n2D CNN Summary:")
        print(f"  Parameters:     {count_parameters(model_2d):,}")
        print(f"  Final Test Acc: {history_2d['test_acc'][-1]:.4f}")
        print()
