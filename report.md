# EdgeINT8-ArchKit: Project Report

> INT8 Edge Inference + Fixed-Point Closure + Architecture Mapping Evaluation

---

## 1. Executive Summary

<!-- TODO [Final]: Write a 1-paragraph summary of the entire project.
     Cover: what you did, key findings, and main conclusions.
     这是整个项目的一句话总结，面试时可以直接引用。 -->

---

## 2. Part A: End-to-End Inference Pipeline

### 2.1 Model Architecture

<!-- TODO [Step 1.1-1.2]: Describe both workload models.
     Include: architecture table (layers, params, FLOPs), input/output shapes.
     建议放一个简单的架构示意图或表格。 -->

### 2.2 FP32 Baseline Results

<!-- TODO [Step 1.6]: Insert FP32 benchmark results.
     Include: accuracy, latency (mean/std/p50/p99), model size (MB).
     从 results/tables/fp32_baseline.csv 提取数据。 -->

| Metric | Workload-1 (1D) | Workload-2 (2D) |
|--------|-----------------|-----------------|
| Accuracy | | |
| Latency (ms) | | |
| Model Size (MB) | | |

---

## 3. Part B: INT8 Post-Training Quantization

### 3.1 Quantization Strategy Comparison

<!-- TODO [Step 2.3-2.4]: Insert quantization comparison table.
     Columns: Strategy, Calibration, Per-channel?, Accuracy, Accuracy Drop, Latency, Speedup.
     从 results/tables/quantization_comparison.csv 提取数据。 -->

### 3.2 Per-Layer Sensitivity Analysis

<!-- TODO [Step 2.5]: Describe which layers are most sensitive to quantization and why.
     Include: error distribution figure, sensitive layer identification.
     哪一层最敏感？为什么？怎么定位的？ -->

### 3.3 Key Findings

<!-- TODO [Step 2.6]: Summarize quantization conclusions.
     Questions to answer:
     - 为什么选这种量化策略？
     - per-channel 比 per-tensor 好多少？在什么条件下？
     - 哪种校准方法最好？为什么？ -->

---

## 4. Part C: Fixed-Point Closure

### 4.1 MAC Bit Growth Analysis

<!-- TODO [Step 3.1]: Show bit growth derivation for key layers.
     Include: formula, per-layer analysis table.
     参考 docs/fixed_point_note.md 中的推导。 -->

### 4.2 Accumulator Bit-Width and Overflow Strategy

<!-- TODO [Step 3.2]: Compare saturation vs wrap-around results.
     Include: error statistics table, recommendation.
     饱和 vs 环绕，在不同累加器位宽下的误差对比。 -->

### 4.3 Reference INT8 Kernel Validation

<!-- TODO [Step 3.3-3.4]: Show golden check test results.
     Include: MSE, max error, percentage within tolerance.
     定点核与浮点参考的一致性验证结果。 -->

---

## 5. Part D: Architecture Evaluation

### 5.1 Design Space Explored

<!-- TODO [Step 4.2-4.3]: Describe the architecture configurations evaluated.
     Include: PE array sizes, SRAM sizes, dataflow types.
     你扫了哪些参数？为什么选这些范围？ -->

### 5.2 Cycle/Energy Trends

<!-- TODO [Step 4.4-4.5]: Insert DSE result figures.
     Include: cycle count vs array size, utilization vs SRAM size.
     从 results/figures/ 引用图表。 -->

### 5.3 Bottleneck Analysis

<!-- TODO [Step 4.5]: Identify memory-bound vs compute-bound configurations.
     Answer: 瓶颈在哪里？怎么改？为什么这个配置更好？ -->

---

## 6. Conclusions

<!-- TODO [Final]: Summarize all findings and answer the three key questions:
     1. "为什么选这种量化策略？"
     2. "为什么这个 PE array/buffer/bandwidth 配置更好？"
     3. "瓶颈在哪里？怎么改？"

     以及对未来工作的建议（如果要继续做，下一步做什么？）。 -->

---

## References

<!-- TODO: List key papers and tools referenced.
     1. Eyeriss (Chen et al., 2016)
     2. Timeloop/Accelergy (Parashar et al., 2019)
     3. SCALE-Sim (Samajdar et al., 2018)
     4. ONNX Runtime documentation -->
