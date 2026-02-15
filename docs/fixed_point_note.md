# Fixed-Point Derivation & Verification Notes

> Part C 核心交付物：3-5 页定点推导与验证笔记
> 目标：把 "量化" 从 "会用工具" 讲到 "会决策 + 可实现"

---

## 1. Introduction to Fixed-Point Arithmetic

<!-- TODO [Step 3.5]: Write introduction
     Cover:
     - What is fixed-point vs floating-point?
     - Why does hardware prefer fixed-point (INT8) over floating-point (FP32)?
       → Lower area, lower power, higher throughput
     - The fundamental trade-off: precision vs efficiency

     定点数 vs 浮点数：
     - FP32: 1-bit sign + 8-bit exponent + 23-bit mantissa → 动态范围大，但硬件昂贵
     - INT8: 8-bit signed integer → 范围小（-128~127），但硬件极其简单
     - INT8 MAC 面积约是 FP32 MAC 的 1/30，能效约是 30×
     - 这就是为什么所有 NPU 都支持 INT8：同样的芯片面积可以放 30× 更多的 MAC
-->

---

## 2. MAC Bit Growth Derivation

<!-- TODO [Step 3.5]: Write bit growth derivation
     Include:
     - Multiplication: INT8 × INT8 = INT16 (prove why M+N bits are needed)
     - Accumulation: K products → need ceil(log2(K)) extra bits
     - Total: 16 + ceil(log2(K)) bits
     - Table showing K values for each layer in both workloads

     公式推导：
       signed INT8 范围：[-128, 127]
       最大乘积：(-128) × (-128) = 16384 → 需要 15 bits + 1 sign bit = 16 bits
       K 次累加最大值：K × 16384
       需要 ceil(log2(K × 16384)) bits = 16 + ceil(log2(K)) bits

     Example table:
     | Layer | Type | Kernel | C_in | K | Product | Acc | Total | HW |
     |-------|------|--------|------|---|---------|-----|-------|-----|
     | conv1 | 2D   | 3×3    | 3    | 27  | 16  | 5   | 21    | 32  |
     | conv2 | 2D   | 3×3    | 32   | 288 | 16  | 9   | 25    | 32  |
     | conv3 | 2D   | 3×3    | 64   | 576 | 16  | 10  | 26    | 32  |
-->

---

## 3. Accumulator Bit-Width Analysis

<!-- TODO [Step 3.5]: Analyze accumulator requirements
     Cover:
     - Why INT32 is the de facto standard
     - When might INT32 not be enough? (very large FC layers)
     - When could a narrower accumulator (INT24, INT16) be used?
     - Table: layer → K → min bits → INT32 margin

     Key insight:
     对于本项目的两个 workload，所有 Conv 层的 K 值都远小于 2^16=65536，
     所以 INT32 累加器绰绰有余（至少有 6 bits 的余量）。
     但如果你设计的 NPU 要处理 K > 65536 的 FC 层，就需要考虑 INT64 或分段累加。
-->

---

## 4. Saturation vs Wrap-Around

<!-- TODO [Step 3.5]: Document overflow strategy comparison
     Include:
     - Experimental results from overflow.py
     - Table: bit-width × strategy × error statistics
     - Figure: error distribution for saturation vs wrap

     Expected conclusions:
     - INT32: 两种策略没有区别（因为不会溢出）
     - INT16: wrap-around 误差是 saturation 的 100-10000×
     - Saturation 在所有情况下都更好或不更差
     - 硬件代价：saturation 需要 2 个比较器 + 1 个 MUX（非常便宜）
-->

---

## 5. Requantization Pipeline

<!-- TODO [Step 3.5]: Explain the requantization process
     Include:
     - Diagram of the full INT8 inference pipeline:
       x_fp32 → quantize → x_int8
                              ↓
       w_fp32 → quantize → w_int8
                              ↓
                          INT8 MAC → acc_int32
                                       ↓
                                  requantize → y_int8
                                       ↓
                              (next layer input)

     - The combined scale factor M = (s_x × s_w) / s_y
     - How M is implemented in hardware (fixed-point multiply + shift)
     - Where the rounding error comes from in requantization
     - Why requantization is often the dominant error source
-->

---

## 6. Experimental Validation

<!-- TODO [Step 3.5]: Document test results from tests.py
     Include:
     - Simple test results (1×1 conv)
     - PyTorch comparison results (random 3×3 conv)
     - End-to-end test results (real model weights)

     Table format:
     | Test Case | MSE | Max Error | Within 1 Step (%) | Status |
     |-----------|-----|-----------|-------------------|--------|
     | 1×1 conv  |     |           |                   |        |
     | 3×3 random|     |           |                   |        |
     | Real model|     |           |                   |        |

     Key conclusions:
     - 大部分输出（>95%）与 FP32 参考值的差异在 1 个量化步长以内
     - 最大误差通常出现在激活值接近量化边界的地方
     - Requantization 的 rounding 贡献了主要误差
-->

---

## 7. Conclusions & Design Guidelines

<!-- TODO [Step 3.5]: Summarize conclusions
     Answer these questions:
     1. 对于本项目的 workload，INT32 累加器是否足够？为什么？
     2. Saturation 和 wrap-around 的实际影响有多大？
     3. 量化误差主要来自哪里？如何减小？
     4. 如果要用更窄的累加器（INT24/INT16），哪些层会受影响？

     Design guidelines:
     - 默认使用 INT32 累加器（几乎没有溢出风险）
     - 如果功耗敏感，可以对 K < 256 的层使用 INT24（20% 面积节省）
     - 权重用对称量化（zero_point=0），简化硬件
     - 激活用非对称量化（利用 ReLU 的单侧分布）
     - Per-channel 权重量化通常值得硬件成本的增加
-->

---

## References

<!-- TODO: List references
     - Jacob et al., "Quantization and Training of Neural Networks for Efficient
       Integer-Arithmetic-Only Inference", CVPR 2018
     - Krishnamoorthi, "Quantizing deep convolutional networks for efficient
       inference", arXiv 2018
     - ONNX Runtime Quantization documentation
-->
