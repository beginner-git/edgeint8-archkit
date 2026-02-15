"""
Fixed-Point Module (Part C — Core Differentiator)
==================================================
This is the most important module for differentiating your portfolio.
It bridges the gap between "using quantization tools" and
"understanding the hardware-level numerical behavior."

模块说明：
- bit_growth.py:   MAC 位增长推导与分析
- overflow.py:     饱和（saturation）vs 环绕（wrap-around）策略对比
- int8_kernel.py:  参考 INT8 卷积核（最核心的文件）
- tests.py:        Golden reference 单元测试
"""

from .bit_growth import compute_mac_bit_growth, recommend_accumulator_width
from .overflow import saturate, wrap_around
from .int8_kernel import int8_conv2d_reference, requantize
