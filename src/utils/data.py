"""
Module: data.py
Part: A (used by all parts)
Purpose: Data loading utilities for both workloads.

Learning Goals:
- Load CIFAR-10 with proper transforms
- Generate synthetic signal data for 1D classification
- Create calibration subsets for quantization

Prerequisites:
- torch, torchvision installed

Usage:
    python -m src.utils.data
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_cifar10_loaders(batch_size=64, data_dir="data/cifar10", num_workers=0):
    """
    Get CIFAR-10 train and test DataLoaders.

    Args:
        batch_size: Batch size
        data_dir: Directory to download/store CIFAR-10
        num_workers: DataLoader workers (0 for Windows compatibility)

    Returns:
        train_loader, test_loader
    """
    # =========================================================================
    # TODO [Step 1.3]: Implement CIFAR-10 data loading
    #
    # Steps:
    #   from torchvision import datasets, transforms
    #
    #   # Standard CIFAR-10 transforms
    #   # For training: augmentation is optional (not the focus of this project)
    #   # For testing: just normalize
    #   transform = transforms.Compose([
    #       transforms.ToTensor(),
    #       transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                            std=[0.2470, 0.2435, 0.2616]),
    #   ])
    #
    #   train_set = datasets.CIFAR10(root=data_dir, train=True,
    #                                download=True, transform=transform)
    #   test_set = datasets.CIFAR10(root=data_dir, train=False,
    #                               download=True, transform=transform)
    #
    #   train_loader = DataLoader(train_set, batch_size=batch_size,
    #                             shuffle=True, num_workers=num_workers)
    #   test_loader = DataLoader(test_set, batch_size=batch_size,
    #                            shuffle=False, num_workers=num_workers)
    #
    #   return train_loader, test_loader
    #
    # 关于 Normalize:
    # - mean 和 std 是 CIFAR-10 训练集的统计量（标准值，可直接使用）
    # - 归一化后数据分布更适合训练（零均值，单位方差附近）
    # - 对量化的影响：归一化后的数据范围大约在 [-2, 2]，
    #   与 INT8 的 [-128, 127] 映射后，每个量化步长约 0.03
    # =========================================================================
    from torchvision import datasets, transforms

    # Standard CIFAR-10 transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True,
                                 download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False,
                                download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_synthetic_signal_loaders(batch_size=64, num_samples=5000,
                                  seq_len=128, num_classes=5,
                                  num_channels=1, seed=42):
    """
    Generate synthetic signal classification data.

    Creates a dataset of 1D signals composed of sinusoidal waves with noise.
    Each class has a different characteristic frequency.

    Args:
        batch_size: Batch size
        num_samples: Total number of samples
        seq_len: Length of each signal sequence
        num_classes: Number of classes
        num_channels: Number of input channels
        seed: Random seed

    Returns:
        train_loader, test_loader
    """
    # =========================================================================
    # TODO [Step 1.3]: Implement synthetic signal data generation
    #
    # Generate signals where each class has a different dominant frequency:
    #
    #   np.random.seed(seed)
    #   t = np.linspace(0, 1, seq_len)  # Time axis
    #
    #   X = np.zeros((num_samples, num_channels, seq_len), dtype=np.float32)
    #   y = np.zeros(num_samples, dtype=np.int64)
    #
    #   for i in range(num_samples):
    #       class_idx = i % num_classes
    #       y[i] = class_idx
    #       # Each class has a base frequency
    #       freq = 2.0 + class_idx * 3.0  # Classes: 2Hz, 5Hz, 8Hz, 11Hz, 14Hz
    #       # Signal = sin(2π·freq·t) + noise
    #       signal = np.sin(2 * np.pi * freq * t)
    #       noise = np.random.randn(seq_len) * 0.3
    #       X[i, 0, :] = signal + noise
    #
    #   # Split into train/test (80/20)
    #   split = int(0.8 * num_samples)
    #   # Shuffle
    #   indices = np.random.permutation(num_samples)
    #   X, y = X[indices], y[indices]
    #
    #   train_dataset = TensorDataset(torch.tensor(X[:split]),
    #                                 torch.tensor(y[:split]))
    #   test_dataset = TensorDataset(torch.tensor(X[split:]),
    #                                torch.tensor(y[split:]))
    #
    #   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    #   return train_loader, test_loader
    #
    # 为什么用合成数据？
    # - 零外部依赖，完全可控可复现
    # - 可以精确控制信号特征（频率、噪声、类别）
    # - 对 1D CNN 来说，频率分类是一个非常自然的任务
    # =========================================================================
    np.random.seed(seed)
    t = np.linspace(0, 1, seq_len)  # Time axis

    X = np.zeros((num_samples, num_channels, seq_len), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        class_idx = i % num_classes
        y[i] = class_idx
        # Each class has a base frequency
        freq = 2.0 + class_idx * 3.0  # Classes: 2Hz, 5Hz, 8Hz, 11Hz, 14Hz
        # Signal = sin(2*pi*freq*t) + noise
        signal = np.sin(2 * np.pi * freq * t)
        noise = np.random.randn(seq_len) * 0.3
        X[i, 0, :] = signal + noise

    # Split into train/test (80/20)
    split = int(0.8 * num_samples)
    # Shuffle
    indices = np.random.permutation(num_samples)
    X, y = X[indices], y[indices]

    train_dataset = TensorDataset(torch.tensor(X[:split]),
                                  torch.tensor(y[:split]))
    test_dataset = TensorDataset(torch.tensor(X[split:]),
                                 torch.tensor(y[split:]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_calibration_dataset(data_loader, num_samples=100):
    """
    Create a small calibration dataset from a DataLoader.

    Used for PTQ calibration — only need a representative subset.

    Args:
        data_loader: Full training DataLoader
        num_samples: Number of calibration samples

    Returns:
        calibration_loader: DataLoader with calibration subset
    """
    # =========================================================================
    # TODO [Step 1.3]: Extract calibration subset
    #
    # Collect first num_samples samples:
    #   data_list = []
    #   label_list = []
    #   count = 0
    #   for data, labels in data_loader:
    #       data_list.append(data)
    #       label_list.append(labels)
    #       count += data.shape[0]
    #       if count >= num_samples:
    #           break
    #
    #   all_data = torch.cat(data_list)[:num_samples]
    #   all_labels = torch.cat(label_list)[:num_samples]
    #   calib_dataset = TensorDataset(all_data, all_labels)
    #   return DataLoader(calib_dataset, batch_size=1, shuffle=False)
    #
    # 为什么用 batch_size=1 for calibration？
    # ORT 的 CalibrationDataReader 逐个样本读取（get_next 返回一个样本），
    # 所以 calibration loader 用 batch_size=1 最方便。
    # =========================================================================
    data_list = []
    label_list = []
    count = 0
    for data, labels in data_loader:
        data_list.append(data)
        label_list.append(labels)
        count += data.shape[0]
        if count >= num_samples:
            break

    all_data = torch.cat(data_list)[:num_samples]
    all_labels = torch.cat(label_list)[:num_samples]
    calib_dataset = TensorDataset(all_data, all_labels)
    return DataLoader(calib_dataset, batch_size=1, shuffle=False)


if __name__ == "__main__":
    print("=" * 40)
    print("  Data Loading Self-Test")
    print("=" * 40)

    print("\n--- Synthetic Signal Data ---")
    train_loader, test_loader = get_synthetic_signal_loaders()
    x, y = next(iter(train_loader))
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Batch shape:   {x.shape}")  # Expected: [64, 1, 128]
    print(f"  Labels shape:  {y.shape}")  # Expected: [64]
    print(f"  Classes:       {sorted(set(y.numpy()))}")

    print("\n--- CIFAR-10 ---")
    train_loader, test_loader = get_cifar10_loaders()
    x, y = next(iter(train_loader))
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Batch shape:   {x.shape}")  # Expected: [64, 3, 32, 32]
    print(f"  Labels shape:  {y.shape}")
