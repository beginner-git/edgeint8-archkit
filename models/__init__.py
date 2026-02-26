"""
EdgeINT8-ArchKit - Model Definitions
=====================================
Part A: End-to-end inference pipeline

This package contains two workload models:
- SignalCNN1D: 1D CNN for synthetic signal classification (Workload-1)
- TinyCNN2D:   Tiny 2D CNN for CIFAR-10 classification (Workload-2)

两个模型都故意设计得很小（<1M 参数），因为本项目的重点不是追求 SOTA 精度，
而是用它们来学习量化、定点、架构评估的完整流程。
"""

# from .signal_cnn_1d import SignalCNN1D, get_signal_cnn_1d
from .tiny_cnn_2d import TinyCNN2D, get_tiny_cnn_2d
