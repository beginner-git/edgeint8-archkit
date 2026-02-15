"""
Quantization Module (Part B)
============================
Hand-written quantization implementations for learning,
plus calibration and analysis utilities.

模块说明：
- quantize_utils.py: 量化数学的手写实现（先理解原理，再用 ORT 工具）
- calibration.py:    激活校准策略（minmax, percentile）
- analysis.py:       逐层量化误差分析
"""

from .quantize_utils import (
    compute_scale_zp_symmetric,
    compute_scale_zp_asymmetric,
    quantize_tensor,
    dequantize_tensor,
    compute_quantization_error,
)
