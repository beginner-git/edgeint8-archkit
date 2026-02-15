# EdgeINT8-ArchKit 项目结构详细说明

> 本文档详细介绍项目中每个文件夹和每个文件的作用。
> 项目整体目标：**用一个"最小闭环项目"把量化、定点、推理工具链、架构评估串起来。**

---

## 项目总览

```
EdgeINT8-ArchKit/
├── models/          # Part A：模型定义与训练
├── data/            # 数据集管理
├── scripts/         # 可执行脚本（入口）
├── src/             # 核心源码模块
│   ├── quant/       #   Part B：量化
│   ├── fixed_point/ #   Part C：定点闭环
│   ├── arch/        #   Part D：架构评估
│   └── utils/       #   共享工具
├── results/         # 输出结果（图表、CSV）
│   ├── figures/     #   生成的图表
│   └── tables/      #   CSV 数据表
├── docs/            # 文档
├── report.md        # 最终项目报告
├── environment.yml  # Conda 环境配置
└── requirements.txt # Pip 依赖
```

---

## 一、`models/` — 模型定义与训练（Part A）

这个文件夹负责定义两个 workload 的神经网络模型，以及训练脚本。

### `models/__init__.py`
- **作用**：Python 包初始化文件，导出模型类和工厂函数
- **导出内容**：`SignalCNN1D`, `get_signal_cnn_1d`, `TinyCNN2D`, `get_tiny_cnn_2d`
- **实现状态**：已完成（仅 import 语句）

### `models/signal_cnn_1d.py`
- **作用**：定义 **Workload-1：1D CNN 信号分类模型**
- **模型结构**：3 层 Conv1d（通道数 1→16→32→64）+ AdaptiveAvgPool1d + 全连接层
- **输入/输出**：`[B, 1, 128]` → `[B, 5]`（5 类信号分类）
- **参数量**：约 25K
- **用途**：用于信号处理场景的轻量级推理模型，适合展示端侧部署能力
- **实现状态**：骨架/模板（`__init__` 和 `forward` 待实现）

### `models/tiny_cnn_2d.py`
- **作用**：定义 **Workload-2：Tiny 2D CNN 图像分类模型**
- **模型结构**：3 个 Conv Block（Conv2d + BatchNorm + ReLU + MaxPool），通道数 32→64→128
- **输入/输出**：`[B, 3, 32, 32]` → `[B, 10]`（CIFAR-10 十分类）
- **参数量**：约 120K
- **用途**：标准图像分类 benchmark，Conv/GEMM 操作更典型，方便与架构评估工具对标
- **实现状态**：骨架/模板

### `models/train.py`
- **作用**：**统一训练脚本**，支持两种 workload 的训练
- **核心函数**：
  - `train_one_epoch()` — 单 epoch 训练循环
  - `evaluate()` — 模型评估
  - `train_model()` — 完整训练流程，含 checkpoint 保存
- **命令行参数**：`--workload`（1d/2d/all）、`--epochs`、`--batch-size`、`--lr`、`--seed`
- **输出**：训练好的 `.pth` 权重文件
- **实现状态**：骨架/模板

---

## 二、`data/` — 数据集管理

这个文件夹管理项目使用的数据集，包括数据下载、验证和文档。

### `data/README.md`
- **作用**：**数据源说明文档**
- **内容**：
  - Workload-1 使用代码生成的合成信号数据（5 类不同频率的正弦波 + 噪声）
  - Workload-2 使用 CIFAR-10 数据集（60K 张 32×32 彩色图像，50K 训练 + 10K 测试）
  - 提供数据加载的代码示例
- **实现状态**：已完成

### `data/download.py`
- **作用**：**数据下载与验证工具**
- **核心函数**：
  - `download_cifar10()` — 下载 CIFAR-10 数据集
  - `verify_dataset()` — 验证数据集完整性
- **用途**：确保实验可复现，其他人 clone 仓库后可一键下载数据
- **实现状态**：骨架/模板

---

## 三、`scripts/` — 可执行脚本（入口）

这个文件夹包含所有顶层可执行脚本，是用户运行实验的入口。

### `scripts/run_all.py`
- **作用**：**主流水线编排脚本**，一键运行整个项目
- **执行顺序**：
  1. Part A：训练模型 → 导出 ONNX → FP32 基线测试
  2. Part B：INT8 量化 → 量化策略对比
  3. Part C：定点测试验证
  4. Part D：架构评估
- **核心函数**：`run_step()` — 执行一个命令并报告状态（成功/失败/跳过）
- **用途**：项目的"一键复现"入口
- **实现状态**：骨架/模板

### `scripts/export_onnx.py`
- **作用**：**PyTorch 模型导出为 ONNX 格式**（Part A）
- **核心函数**：
  - `export_to_onnx()` — 导出模型，支持动态 batch size
  - `verify_onnx()` — 验证导出的 ONNX 模型与 PyTorch 输出一致
  - `get_model_size_mb()` — 获取 ONNX 文件大小
- **命令行参数**：`--workload`、`--models-dir`、`--opset`
- **用途**：将训练好的 PyTorch 模型转换为 ONNX 格式，为后续 ORT 推理和量化做准备
- **实现状态**：骨架/模板

### `scripts/bench.py`
- **作用**：**统一基准测试脚本**（Part A + Part B）
- **核心函数**：
  - `measure_latency()` — 测量推理延迟（含 warmup）
  - `measure_accuracy()` — 在测试集上评估准确率
  - `measure_model_size()` — 获取模型文件大小
  - `benchmark_single()` — 对单个模型做综合测试
  - `benchmark_comparison()` — 多模型对比测试
  - `format_results_table()` — 格式化结果表格
- **命令行参数**：`--model`、`--workload`、`--all`、`--compare`、`--num-runs`
- **输出**：延迟/准确率/模型大小的对比表格
- **用途**：量化评估 FP32 vs INT8 的性能差异
- **实现状态**：骨架/模板

### `scripts/quantize_ptq.py`
- **作用**：**INT8 训练后量化（PTQ）脚本**（Part B）
- **核心类**：
  - `CalibrationDataReaderBase` — ORT 校准数据读取器基类
  - `CifarCalibrationDataReader` — CIFAR-10 校准数据读取器
  - `SignalCalibrationDataReader` — 合成信号校准数据读取器
- **核心函数**：
  - `quantize_model_static()` — 应用 ORT 静态 INT8 量化
  - `run_quantization_sweep()` — 扫描多种量化配置（对称/非对称 × per-tensor/per-channel × 不同校准方法）
- **命令行参数**：`--workload`、`--model`、`--sweep`、`--calibration-samples`
- **用途**：用 ONNX Runtime 对模型做 INT8 量化，并扫描不同策略组合
- **实现状态**：骨架/模板

### `scripts/visualize.py`
- **作用**：**生成可视化图表**（Part B + Part D）
- **Part B 图表函数**：
  - `plot_accuracy_comparison()` — 不同量化策略的准确率对比柱状图
  - `plot_latency_comparison()` — 延迟对比图
  - `plot_accuracy_vs_latency()` — 准确率-延迟散点图
  - `plot_error_distribution()` — 量化误差分布图
- **Part D 图表函数**：
  - `plot_dse_cycles()` — DSE 周期趋势图
  - `plot_dse_utilization()` — PE 利用率图
- **输出目录**：`results/figures/`
- **用途**：为报告生成专业级图表
- **实现状态**：骨架/模板

### `scripts/run_arch_eval.py`
- **作用**：**架构评估入口脚本**（Part D）
- **流程**：加载模型 → 生成 topology CSV → 运行 DSE → 分析结果
- **用途**：使用 SCALE-Sim 对模型的卷积层做架构级性能评估
- **实现状态**：骨架/模板

---

## 四、`src/` — 核心源码模块

这是项目的核心代码库，按功能分为四个子包。

### `src/__init__.py`
- **作用**：源码包初始化，包含模块结构说明文档
- **实现状态**：已完成

---

### 4.1 `src/quant/` — 量化模块（Part B）

负责实现量化相关的数学计算、校准策略和误差分析。

#### `src/quant/__init__.py`
- **作用**：导出量化核心函数
- **导出**：`compute_scale_zp_symmetric`, `compute_scale_zp_asymmetric`, `quantize_tensor`, `dequantize_tensor`, `compute_quantization_error`
- **实现状态**：已完成

#### `src/quant/quantize_utils.py`
- **作用**：**手写量化数学实现**（核心学习模块，不依赖框架）
- **核心函数**：
  - `compute_scale_zp_symmetric()` — 计算对称量化的 scale 和 zero_point（zp=0），适用于权重
  - `compute_scale_zp_asymmetric()` — 计算非对称量化的 scale 和 zero_point，适用于激活值（利用 ReLU 单侧分布）
  - `quantize_tensor()` — 将 FP32 张量量化为 INT8
  - `dequantize_tensor()` — 将 INT8 张量反量化为 FP32
  - `compute_quantization_error()` — 计算量化误差（MSE、最大误差、SQNR）
- **设计理念**：手写而非调用框架，展示对量化原理的深入理解
- **实现状态**：骨架/模板

#### `src/quant/calibration.py`
- **作用**：**激活校准策略实现**（手写，学习用）
- **核心函数**：
  - `calibrate_minmax()` — MinMax 校准：使用全局最小/最大值
  - `calibrate_percentile()` — 百分位校准：裁剪 outlier
  - `calibrate_entropy()` — KL 散度校准（进阶）
  - `collect_layer_activations()` — 使用 PyTorch hook 收集中间层激活值
  - `compare_calibration_methods()` — 对比不同校准策略的效果
- **关键知识点**：为什么需要校准（激活是数据依赖的）、MinMax vs Percentile 的权衡、Outlier 处理
- **实现状态**：骨架/模板

#### `src/quant/analysis.py`
- **作用**：**逐层量化误差分析与敏感度研究**
- **核心函数**：
  - `analyze_per_layer_sensitivity()` — Leave-one-out 敏感度分析（逐层量化，观察精度下降）
  - `generate_error_report()` — 生成综合的逐层误差报告
  - `identify_sensitive_layers()` — 识别 SQNR 低的敏感层
- **关键知识点**：判断哪些层对量化最敏感、为混合精度量化提供决策依据
- **实现状态**：骨架/模板

---

### 4.2 `src/fixed_point/` — 定点闭环模块（Part C，核心差异化）

这是项目最重要的差异化模块。目标是把"量化"从"会用工具"提升到"理解硬件实现"。

#### `src/fixed_point/__init__.py`
- **作用**：导出定点运算核心函数
- **导出**：`compute_mac_bit_growth`, `recommend_accumulator_width`, `saturate`, `wrap_around`, `int8_conv2d_reference`, `requantize`
- **实现状态**：已完成

#### `src/fixed_point/bit_growth.py`
- **作用**：**MAC 位宽增长分析与累加器宽度推导**
- **核心函数**：
  - `compute_mac_bit_growth()` — 计算乘累加操作需要的总位宽
  - `compute_accumulation_count_conv2d()` — 计算 2D 卷积的累加次数 K = Cin × Kh × Kw
  - `compute_accumulation_count_conv1d()` — 计算 1D 卷积的累加次数 K = Cin × Kw
  - `recommend_accumulator_width()` — 推荐硬件累加器位宽（16/32/64 bit）
  - `analyze_model_bit_growth()` — 分析模型所有 Conv 层的位宽需求
- **核心公式**：乘积位宽 = input_bits + weight_bits；累加位宽 = ceil(log2(K))；总位宽 = 16 + ceil(log2(K))
- **实现状态**：骨架/模板

#### `src/fixed_point/overflow.py`
- **作用**：**溢出处理策略对比（饱和 vs 回绕）**
- **核心函数**：
  - `saturate()` — 饱和策略：超出范围则钳位到最大/最小值（安全但需要硬件比较器）
  - `wrap_around()` — 回绕策略：补码取模（零硬件开销但可能导致灾难性误差）
  - `get_range_for_bits()` — 获取 N 位有符号整数的表示范围
  - `compare_overflow_strategies()` — 在不同累加器位宽下对比两种策略的误差
  - `simulate_conv_accumulation()` — 模拟单个输出像素的累加过程
- **关键结论**：INT32 累加器几乎不会溢出；INT16 下 wrap-around 误差是 saturation 的 100-10000 倍
- **实现状态**：骨架/模板

#### `src/fixed_point/int8_kernel.py`
- **作用**：**参考级 INT8 卷积核实现**（纯 Python/NumPy，核心交付物）
- **核心函数**：
  - `requantize()` — INT32 累加器结果重新量化为 INT8 输出
  - `quantize_bias()` — 将 bias 预量化为 INT32
  - `int8_conv2d_reference()` — 完整的 INT8 2D 卷积核（7 层嵌套循环：N, Co, H, W, Ci, Kh, Kw）
  - `int8_conv1d_reference()` — 完整的 INT8 1D 卷积核
  - `validate_against_float()` — 与 FP32 参考实现做对比验证
- **核心流程**：quantize → int8 MAC/acc → requantize（完整的 INT8 推理 pipeline）
- **重量化公式**：`y_int8 = round(acc_int32 × M) + zp_y`，其中 `M = (s_x × s_w) / s_y`
- **设计理念**：这个文件把你从"我用过量化工具"提升到"我理解硬件如何做 INT8 推理"
- **实现状态**：骨架/模板

#### `src/fixed_point/tests.py`
- **作用**：**定点运算的 Golden Reference 单元测试**
- **测试用例**：
  - `test_bit_growth_calculation()` — 验证位宽增长公式
  - `test_saturation_vs_wrap()` — 测试溢出策略
  - `test_requantization()` — 验证重量化正确性
  - `test_int8_conv2d_simple()` — 简单 1×1 卷积验证
  - `test_int8_conv2d_against_pytorch()` — 与 PyTorch FP32 对比
  - `test_end_to_end_int8_pipeline()` — 使用真实权重的端到端测试
  - `run_all_tests()` — 测试运行器，统计 pass/fail/skip
- **验收标准**：>95% 的输出与 FP32 参考值差异在 1 个量化步长以内
- **实现状态**：骨架/模板

---

### 4.3 `src/arch/` — 架构评估模块（Part D）

负责将模型映射到硬件加速器架构上，做性能/能效评估。

#### `src/arch/__init__.py`
- **作用**：架构评估模块文档
- **实现状态**：已完成

#### `src/arch/scalesim_runner.py`
- **作用**：**SCALE-Sim（脉动阵列模拟器）封装**
- **核心函数**：
  - `generate_topology_csv()` — 从 PyTorch 模型提取卷积参数，生成 SCALE-Sim 格式的 topology CSV
  - `generate_scalesim_config()` — 生成 SCALE-Sim 配置文件（.cfg），包含 PE 阵列大小、SRAM 大小、数据流等
  - `run_scalesim()` — 执行 SCALE-Sim 仿真
  - `parse_compute_report()` — 解析计算报告（周期数、PE 利用率）
  - `parse_bandwidth_report()` — 解析带宽报告
- **关键概念**：脉动阵列（Systolic Array）、三种数据流（WS/OS/IS）、PE 阵列大小与 SRAM 大小对性能的影响
- **实现状态**：骨架/模板

#### `src/arch/dse.py`
- **作用**：**设计空间探索（Design Space Exploration）**
- **核心函数**：
  - `define_design_space()` — 定义可搜索的参数空间（阵列大小、SRAM 容量、数据流等）
  - `generate_all_configs()` — 生成所有参数组合
  - `run_dse_sweep()` — 对所有配置运行仿真
  - `analyze_dse_results()` — 寻找趋势和最优配置
  - `identify_bottleneck()` — 判断瓶颈是 memory-bound 还是 compute-bound
- **关键概念**：Roofline 模型、收益递减、"为什么这个配置更好"的叙事能力
- **实现状态**：骨架/模板

---

### 4.4 `src/utils/` — 共享工具模块

提供项目各模块共用的基础功能。

#### `src/utils/__init__.py`
- **作用**：工具模块文档
- **实现状态**：已完成

#### `src/utils/data.py`
- **作用**：**数据加载工具**
- **核心函数**：
  - `get_cifar10_loaders()` — 加载 CIFAR-10 数据集（含标准化：mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]）
  - `get_synthetic_signal_loaders()` — 生成合成 1D 信号数据（5 类不同频率正弦波：2/5/8/11/14 Hz + 噪声）
  - `create_calibration_dataset()` — 提取校准子集（batch_size=1，兼容 ORT 校准接口）
- **实现状态**：骨架/模板

#### `src/utils/profiling.py`
- **作用**：**计时与性能分析工具**
- **核心组件**：
  - `Timer` 类 — 上下文管理器，用于代码块计时
  - `profile_ort_session()` — ORT 推理的详细延迟统计（均值/标准差/P50/P95/P99/最小值）
  - `get_model_info()` — ONNX 模型内省（算子数、参数量、大小、算子类型分布）
- **实现状态**：骨架/模板

#### `src/utils/helpers.py`
- **作用**：**通用辅助函数**
- **核心函数**：
  - `set_seed()` — 设置随机种子（random/numpy/torch），确保实验可复现
  - `ensure_dir()` — 创建目录（如果不存在）
  - `format_size()` — 字节数转人类可读格式（KB/MB/GB）
  - `save_dict_to_csv()` — 将字典列表保存为 CSV
  - `load_csv_to_dict()` — 从 CSV 加载为字典列表
  - `count_parameters()` — 统计模型可训练参数量
- **实现状态**：**已完成**（项目中唯一完全实现的源码文件）

---

## 五、`results/` — 输出结果目录

存放项目运行产生的所有结果数据。

### `results/figures/`
- **作用**：**存放生成的图表**
- **预期内容**：
  - Part B：量化策略准确率对比图、延迟对比图、准确率-延迟散点图、误差分布图
  - Part D：DSE 周期趋势图、PE 利用率图
- **当前状态**：空目录

### `results/tables/`
- **作用**：**存放 CSV 数据表**
- **预期内容**：
  - FP32 vs INT8 性能对比表
  - 量化策略扫描结果
  - 逐层敏感度分析结果
  - DSE 扫描结果
- **当前状态**：空目录

---

## 六、`docs/` — 文档目录

存放项目的技术文档和学习笔记。

### `docs/fixed_point_note.md`
- **作用**：**定点推导与验证笔记**（Part C 核心交付物，3-5 页）
- **章节结构**：
  1. 定点算术简介（FP32 vs INT8 的硬件代价对比）
  2. MAC 位宽增长推导（乘法 + 累加的位宽需求公式）
  3. 累加器位宽分析（为什么 INT32 是工业标准）
  4. 饱和 vs 回绕策略对比（实验结果与硬件代价分析）
  5. 重量化流水线（INT8 推理的完整数据路径）
  6. 实验验证（Golden Reference 测试结果）
  7. 结论与设计指南
- **实现状态**：模板（含详细 TODO 指导）

### `docs/paper_notes.md`
- **作用**：**论文阅读笔记模板**
- **包含论文**：
  1. **Eyeriss**（ISCA 2016）— 数据流分类与能效分析的基础
  2. **Timeloop**（ISPASS 2019）— 系统化加速器评估方法
  3. **Gemmini**（DAC 2021）— 开源可配置加速器生成器
  4. **NVDLA** — 工业级加速器设计参考
- **笔记格式**：问题/方法/关键洞察/与本项目的关系
- **实现状态**：模板（仅含 TODO 指导和空字段）

### `docs/project_structure.md`
- **作用**：**本文件**，项目结构的完整说明文档

---

## 七、根目录文件

### `README.md`
- **作用**：**项目主文档**
- **内容**：Quick Start 指南、完整目录结构说明、4 阶段学习路径（含检查点）、Workload 说明、最终交付物清单、简历可用 bullet points
- **实现状态**：已完成

### `report.md`
- **作用**：**最终项目报告模板**
- **章节结构**：
  1. 执行摘要
  2. Part A：端到端推理（模型设计、FP32 基线）
  3. Part B：INT8 量化（策略对比、敏感度分析）
  4. Part C：定点闭环（位宽增长、溢出策略、内核验证）
  5. Part D：架构评估（设计空间、周期/能效趋势、瓶颈分析）
  6. 结论与参考文献
- **实现状态**：模板（含 TODO 占位）

### `environment.yml`
- **作用**：**Conda 环境配置文件**
- **环境名**：`edgeint8`
- **核心依赖**：Python 3.10、PyTorch ≥ 2.0、torchvision ≥ 0.15、ONNX ≥ 1.14、ONNX Runtime ≥ 1.16、matplotlib、pandas、tabulate、tqdm、scipy
- **实现状态**：已完成

### `requirements.txt`
- **作用**：**Pip 依赖文件**（`environment.yml` 的替代方案）
- **实现状态**：已完成

---

## 八、实现状态总结

| 状态 | 文件数 | 说明 |
|------|--------|------|
| **已完成** | 12 | 包初始化文件、配置文件、`helpers.py`、文档 |
| **骨架/模板** | 20 | 含详细 TODO 注释和架构设计，待逐步实现 |

### 按 Part 分类的实现路径

| 阶段 | 涉及文件 | 预计周数 | 关键检查点 |
|------|----------|----------|-----------|
| **Part A** | `models/`, `data/`, `export_onnx.py`, `bench.py` | 1-2 周 | FP32 基线结果输出 |
| **Part B** | `src/quant/`, `quantize_ptq.py`, `visualize.py` | 2-4 周 | FP32 vs INT8 对比表 |
| **Part C** | `src/fixed_point/`, `docs/fixed_point_note.md` | 4-7 周 | 所有 Golden Test 通过 |
| **Part D** | `src/arch/`, `run_arch_eval.py`, `report.md` | 7-12 周 | DSE 趋势图 + 最终报告 |
