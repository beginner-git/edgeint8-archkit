# Paper Reading Notes

> 每篇 10-15 行笔记：问题/方法/关键洞察/与本项目的关系
> 建议顺序阅读，每周 2-3 小时

---

## 1. Eyeriss: An Energy-Efficient Reconfigurable Accelerator for DNNs
**Authors**: Chen et al., ISCA 2016

<!-- TODO [ToDo 5]: Read and summarize Eyeriss

Questions to answer:
- What problem does Eyeriss solve?
- What is "Row Stationary" dataflow and why is it energy-efficient?
- How does Eyeriss handle data reuse across PE array?
- What is the memory hierarchy in Eyeriss?

Key concepts:
- Dataflow taxonomy: WS, OS, IS, RS
- Data reuse: convolutional reuse, filter reuse, ifmap reuse
- Energy breakdown: DRAM access vs computation
  (DRAM access costs ~200× more energy than an INT8 MAC!)

与本项目的关系：
- Part D (架构评估) 的理论基础
- 理解为什么选择不同 dataflow 会导致不同的能效
- SCALE-Sim 的 dataflow 选项直接对应 Eyeriss 的分类
-->

**Problem**:
**Method**:
**Key Insight**:
**Relevance to my project**:

---

## 2. Timeloop: A Systematic Approach to DNN Accelerator Evaluation
**Authors**: Parashar et al., ISPASS 2019

<!-- TODO [ToDo 5]: Read and summarize Timeloop

Questions to answer:
- What is a "mapping" in the context of accelerator evaluation?
- How does Timeloop search the mapping space?
- What metrics does Timeloop output (energy, latency, area)?
- What is Accelergy and how does it complement Timeloop?

Key concepts:
- Loop nest representation of DNN computation
- Mapping = loop order + tiling + spatial mapping
- Energy estimation via Accelergy (technology-dependent)
- The importance of mapping optimization (same arch, different mapping → 10× energy difference!)

与本项目的关系：
- Part D 的备选工具（如果 SCALE-Sim 不够用）
- 理解 "mapping" 的概念对 DSE 至关重要
- 论文中的方法论（固定 workload → 扫映射空间 → 输出趋势）就是我们 DSE 的流程
-->

**Problem**:
**Method**:
**Key Insight**:
**Relevance to my project**:

---

## 3. Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation
**Authors**: Genc et al., DAC 2021

<!-- TODO [ToDo 5]: Read and summarize Gemmini

Questions to answer:
- What is Gemmini's hardware architecture?
- How is it different from Eyeriss?
- What is the role of the RoCC interface?
- How does Gemmini handle different datatypes (INT8, FP16)?

Key concepts:
- RISC-V based accelerator generator
- Systolic array with configurable dimensions
- Software stack: ONNX → Gemmini runtime → hardware
- Open-source and modifiable (useful for research)

与本项目的关系：
- 如果将来想做 FPGA prototyping，Gemmini 是一个好的起点
- 理解一个完整的加速器系统（不只是 PE array，还有控制、内存、接口）
- Gemmini 的 datatype 配置与 Part C 的定点分析直接相关
-->

**Problem**:
**Method**:
**Key Insight**:
**Relevance to my project**:

---

## 4. NVDLA: NVIDIA Deep Learning Accelerator
**Source**: NVDLA documentation & technical reference manual

<!-- TODO [ToDo 5]: Read and summarize NVDLA

Questions to answer:
- What is NVDLA's high-level architecture?
- How does NVDLA handle INT8 quantization?
- What are the key hardware blocks (convolution core, SDP, PDP, CDP)?
- What design trade-offs does NVDLA make for edge deployment?

Key concepts:
- Industrial-grade accelerator design (vs academic prototypes)
- Small vs Large NVDLA configurations
- Convolution Buffer Architecture (CBUF)
- Second-level post-processing pipeline (SDP for quantization, bias, activation)

与本项目的关系：
- 了解工业级加速器如何处理 INT8 推理
- NVDLA 的 SDP 模块就是做 requantization 的（Part C 的硬件实现）
- NVDLA 的设计决策可以为你的 DSE 分析提供参考
-->

**Problem**:
**Method**:
**Key Insight**:
**Relevance to my project**:
