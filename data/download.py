"""
Module: download.py
Part: A
Purpose: Data download and verification utility.

This is an optional helper script. CIFAR-10 is downloaded automatically
by torchvision when you first run the training script. This script is
useful for pre-downloading data or verifying the dataset.

Usage:
    python data/download.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_cifar10(data_dir="data/cifar10"):
    """
    Download CIFAR-10 dataset via torchvision.

    Args:
        data_dir: Directory to save the dataset
    """
    # =========================================================================
    # TODO [Step 1.3]: Implement CIFAR-10 download
    #
    # Steps:
    #   from torchvision import datasets
    #   train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    #   test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
    #   print(f"Train samples: {len(train_set)}")
    #   print(f"Test samples:  {len(test_set)}")
    #   print(f"Classes: {train_set.classes}")
    #   print(f"Image shape: {train_set[0][0].size}")  # PIL Image size
    #
    # CIFAR-10 包含 10 个类别：
    # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.3]: Implement CIFAR-10 download")


def verify_dataset(data_dir="data/cifar10"):
    """Print dataset statistics for verification."""
    # =========================================================================
    # TODO [Step 1.3]: Load dataset and print statistics
    #
    # Print: num samples, image shape, num classes, class distribution
    # =========================================================================
    raise NotImplementedError("TODO [Step 1.3]: Implement dataset verification")


if __name__ == "__main__":
    print("Downloading CIFAR-10...")
    download_cifar10()
    print("\nVerifying dataset...")
    verify_dataset()
    print("\nDone!")
