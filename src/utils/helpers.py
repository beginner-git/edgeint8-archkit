"""
Module: helpers.py
Part: All
Purpose: Miscellaneous helper functions shared across the project.

These are simple utility functions. Implement them first as they are
used by almost every other module.
"""

import os
import random
import csv

import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def format_size(size_bytes):
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def save_dict_to_csv(data_list, path, fieldnames=None):
    """
    Save a list of dicts to CSV.

    Args:
        data_list: List of dicts (each dict is a row)
        path: Output CSV path
        fieldnames: Column names (auto-detected if None)
    """
    if not data_list:
        return

    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(data_list[0].keys())

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)


def load_csv_to_dict(path):
    """
    Load a CSV file into a list of dicts.

    Args:
        path: CSV file path

    Returns:
        List of dicts
    """
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def count_parameters(model):
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
