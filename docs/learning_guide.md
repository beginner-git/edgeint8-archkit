# EdgeINT8-ArchKit 学习指南：从零到闭环

> 本文档是你做这个项目的"导航手册"。
> 按顺序读、按顺序做，每一步都有**要学什么**、**怎么做**、**怎么验证**、**踩坑提醒**。
> 预计总周期：8-12 周。节奏建议：**1.5× 缓冲**，不给自己不必要的压力。

---

## 0. 开始之前：你需要理解的大图

### 0.1 这个项目到底在做什么？

一句话：**把一个神经网络，从"能跑"变成"能高效地跑在硬件上"，并用数据证明你的每一步决策。**

具体来说，你要走通这条链路：

```
PyTorch 模型 (FP32)
      │
      ▼
  ① 导出 ONNX ──→ ORT 推理 ──→ FP32 基线 (accuracy/latency/size)
      │
      ▼
  ② INT8 量化 ──→ 策略对比 ──→ "为什么选这种量化策略？"
      │
      ▼
  ③ 定点闭环 ──→ 手写 INT8 卷积核 ──→ "硬件里到底发生了什么？"
      │
      ▼
  ④ 架构评估 ──→ DSE 趋势分析 ──→ "为什么这个架构配置更好？瓶颈在哪？"
```

### 0.2 你最终要能回答的三个问题

面试官 / 导师 / 自己复盘时，你要能清晰回答：

1. **"为什么选这种量化策略？"** — 对称 vs 非对称、per-tensor vs per-channel、不同校准方法的 trade-off
2. **"硬件上 INT8 推理到底怎么算的？位宽/溢出怎么处理？"** — bit growth、累加器位宽、saturation vs wrap-around
3. **"为什么这个 PE array/buffer/bandwidth 配置更好？瓶颈在哪？"** — roofline、memory-bound vs compute-bound

### 0.3 学习心态

- **不追求 SOTA**。模型精度不重要，重要的是你能解释"为什么"
- **先跑通，再优化**。每个 Part 都先写出最简单的能工作的版本，再迭代
- **手写优先于调库**。`quantize_utils.py` 和 `int8_kernel.py` 是你手写的，这才是面试能讲的故事
- **每一步都有检查点**。过了检查点再往下走，不要跳步

---

## 1. Phase 0：环境搭建与项目熟悉（第 0 周，1-2 天）

### 1.1 搭建环境

```bash
# 方式一：Conda（推荐）
conda env create -f environment.yml
conda activate edgeint8

# 方式二：Pip
pip install -r requirements.txt

# 验证安装
python -c "import torch; import onnx; import onnxruntime; print('All dependencies OK')"
```

### 1.2 通读项目结构

1. **先读 `README.md`**，了解整体结构
2. **再读 `docs/project_structure.md`**，了解每个文件干什么
3. **翻一遍所有源文件的文件头注释**（每个 `.py` 文件顶部都有 docstring），了解它的 Learning Goals
4. **看一遍所有 TODO 注释**，心里有数每个函数要做什么

### 1.3 Git 初始化

```bash
git init
git add .
git commit -m "Initial skeleton: project structure with TODO templates"
```

> **原则**：每完成一个 Step，就 commit 一次。你的 Git 历史就是你的学习记录。

### 验证检查点

- [ ] `conda activate edgeint8` 成功
- [ ] `import torch; import onnx; import onnxruntime` 无报错
- [ ] 你能说出项目有 4 个 Part，分别是什么

---

## 2. Phase 1：端到端推理 Pipeline — Part A（第 1-2 周）

> **目标**：训练两个小模型，导出 ONNX，用 ORT 跑通 FP32 推理，建立性能基线。
> **这是地基**，后面所有 Part 都依赖这个。

### 2.1 Step 1.1 — 实现 1D CNN 模型

**文件**：`models/signal_cnn_1d.py`

**要学什么**：
- PyTorch `nn.Module` 的基本用法
- Conv1d 的参数含义：`in_channels`, `out_channels`, `kernel_size`
- AdaptiveAvgPool1d 的作用：把不定长输出变成固定长度

**怎么做**：
1. 打开文件，阅读 docstring 和 TODO 注释
2. 注释里已经给出了详细的架构设计：3 层 Conv1d + Pool + FC
3. 按 TODO 的 Steps 一步步填入代码
4. 实现 `__init__` 里定义层，`forward` 里写前向传播

**自测**：
```bash
python -m models.signal_cnn_1d
# 应输出 "Output shape: torch.Size([2, 5])" 和参数量
```

**踩坑提醒**：
- `view()` 或 `flatten()` 把 3D tensor 变成 2D 前，注意 batch 维度
- 参数量应该在 ~25K，如果差太多检查通道数

---

### 2.2 Step 1.2 — 实现 2D CNN 模型

**文件**：`models/tiny_cnn_2d.py`

**要学什么**：
- Conv2d + BatchNorm + ReLU 的标准 block 模式
- BatchNorm 在推理时的行为（fold 到 Conv 里）
- MaxPool2d 对 feature map 尺寸的影响

**怎么做**：
1. 同样阅读 TODO 注释，架构已经设计好
2. 3 个 Conv Block + AdaptiveAvgPool2d + FC
3. 注意 `MaxPool2d(2)` 每次把 spatial 尺寸减半

**自测**：
```bash
python -m models.tiny_cnn_2d
# 应输出 "Output shape: torch.Size([2, 10])" 和参数量 (~120K)
```

---

### 2.3 Step 1.3 — 实现数据加载

**文件**：`src/utils/data.py`

**要学什么**：
- `torchvision.datasets.CIFAR10` 的使用
- 数据标准化（为什么要 normalize，mean/std 怎么来的）
- 合成数据的生成（不同频率的正弦波 + 噪声 → 分类任务）

**怎么做**：
1. CIFAR-10：用 `torchvision.datasets.CIFAR10` 加载，标准化参数已经在注释里给出
2. 合成信号：生成 5 类不同频率的正弦波（2/5/8/11/14 Hz）+ 高斯噪声
3. 校准数据集：从训练集中采样一个子集，batch_size=1

**自测**：
```bash
python -m src.utils.data
# 应输出数据集大小和 batch shape
```

---

### 2.4 Step 1.4 — 训练模型

**文件**：`models/train.py`

**要学什么**：
- 标准训练循环：forward → loss → backward → optimizer.step()
- 交叉熵损失函数 `CrossEntropyLoss`
- 保存/加载 checkpoint（`torch.save` / `torch.load`）

**怎么做**：
1. 实现 `train_one_epoch()`：遍历 dataloader，前向 → 算 loss → 反向 → 更新
2. 实现 `evaluate()`：`with torch.no_grad()` 模式下计算准确率
3. 实现 `train_model()`：循环调用上面两个函数，保存最佳权重

**运行**：
```bash
python models/train.py --workload 2d --epochs 20
python models/train.py --workload 1d --epochs 20
```

**验证**：
- 2D CNN on CIFAR-10 应达到 ~70-80% accuracy（不需要高，够分析就行）
- 1D CNN on 合成信号应达到 >90% accuracy
- `models/` 目录下应出现 `.pth` 权重文件

---

### 2.5 Step 1.5 — 导出 ONNX

**文件**：`scripts/export_onnx.py`

**要学什么**：
- `torch.onnx.export()` 的用法
- 什么是 ONNX（Open Neural Network Exchange）以及为什么要用它
- `dynamic_axes` 的意义（让 batch size 灵活）
- ONNX 验证：导出后重新加载，对比 PyTorch 和 ORT 的输出

**怎么做**：
1. 加载 `.pth` 权重
2. 创建 dummy input（和训练时同 shape）
3. `torch.onnx.export()` 导出
4. `onnx.checker.check_model()` 验证模型合法性
5. 用 ORT 推理同一个输入，对比与 PyTorch 的输出差异（应 < 1e-5）

**运行**：
```bash
python scripts/export_onnx.py --workload 2d
python scripts/export_onnx.py --workload 1d
```

**踩坑提醒**：
- 如果有 BatchNorm，导出前先 `model.eval()`
- `opset_version` 建议用 13 或更高

---

### 2.6 Step 1.6 — FP32 基准测试

**文件**：`scripts/bench.py`

**要学什么**：
- 用 ORT (`InferenceSession`) 做推理
- 如何正确测量延迟（warmup + 多次运行取统计）
- 模型大小的度量（文件大小、参数量）

**怎么做**：
1. `measure_latency()`：先 warmup 10 次，再正式测量 100 次，取均值/P95
2. `measure_accuracy()`：把测试集全部过一遍 ORT，计算 top-1 accuracy
3. `measure_model_size()`：`os.path.getsize()` 获取文件大小

**运行**：
```bash
python scripts/bench.py --workload 2d
```

### Phase 1 检查点

- [ ] 两个模型训练完毕，有 `.pth` 权重
- [ ] ONNX 文件导出成功，ORT 验证通过
- [ ] FP32 基线数据：accuracy / latency / model size
- [ ] 能一句话说出每个 workload 的 FP32 性能

```bash
# 理想输出示例：
# Workload-2 (Tiny 2D CNN, CIFAR-10):
#   Accuracy: 76.3%
#   Latency (batch=1): 0.85 ms
#   Model size: 0.48 MB
```

---

## 3. Phase 2：INT8 量化 — Part B（第 2-4 周）

> **目标**：理解量化数学，手写实现，再用 ORT 做工业级 PTQ，对比多种策略。
> **核心心法**：先手写理解原理，再用工具跑通工程。

### 3.1 Step 2.1 — 手写量化数学（最重要的一步）

**文件**：`src/quant/quantize_utils.py`

**要学什么**（按顺序）：

#### 先理解量化的本质
```
浮点数 x 映射到整数 q：
  q = round(x / scale) + zero_point
  x ≈ (q - zero_point) * scale
```

你要弄明白 `scale` 和 `zero_point` 到底是什么：
- `scale`：一个量化步长对应多大的浮点范围。scale 越大，范围越大但精度越低
- `zero_point`：浮点 0.0 映射到哪个整数。对称量化时 zp=0，非对称时 zp≠0

#### 对称量化 vs 非对称量化

| 特性 | 对称量化 | 非对称量化 |
|------|---------|-----------|
| Zero Point | 固定为 0 | 一般非零 |
| 量化范围 | [-127, 127] (int8) | [0, 255] (uint8) |
| 适用场景 | **权重**（分布近似对称） | **激活**（ReLU 后全正） |
| 硬件优势 | 不需要 zp 补偿，MAC 更简单 | 充分利用量化范围 |
| 硬件代价 | 无额外开销 | MAC 时需要 zp 补偿项 |

**怎么做**：
1. 先在纸上推导公式（文件注释里有完整推导）
2. 逐函数实现：`compute_scale_zp_symmetric` → `compute_scale_zp_asymmetric` → `quantize_tensor` → `dequantize_tensor` → `compute_quantization_error`
3. 每写一个函数就运行 self-test 验证

**自测**：
```bash
python -m src.quant.quantize_utils
# 检查 SQNR 是否合理（8-bit 应该在 40-50 dB 左右）
# 检查 Asymmetric 在 ReLU 激活上是否比 Symmetric 更好
```

**踩坑提醒**：
- `np.round` 用的是银行家舍入（round half to even），这和硬件常用的 round half away from zero 不一样
- scale 为 0 时要特殊处理（tensor 全零的情况）
- 量化后要 `clamp`（裁剪），否则溢出

---

### 3.2 Step 2.2 — 激活校准策略

**文件**：`src/quant/calibration.py`

**要学什么**：
- 为什么需要"校准"：权重是固定的（可以直接算 min/max），但激活是数据依赖的（需要跑一批数据来统计）
- MinMax 校准：最简单，但容易被 outlier 拉偏
- Percentile 校准：裁掉极端值（如去掉最大/最小 0.1%），更鲁棒
- KL 散度校准（进阶）：找最佳裁剪阈值，使量化后分布最接近原始分布

**怎么做**：
1. `collect_layer_activations()`：用 PyTorch 的 `register_forward_hook` 拦截中间层输出
2. `calibrate_minmax()`：取所有 batch 的全局 min/max
3. `calibrate_percentile()`：取 percentile（如 99.9%）作为 max

**自测**：
```bash
python -m src.quant.calibration
```

---

### 3.3 Step 2.3 — ORT 量化扫描

**文件**：`scripts/quantize_ptq.py`

**要学什么**：
- ONNX Runtime 的 `quantize_static` API
- 静态量化 vs 动态量化的区别
- 如何写 CalibrationDataReader（ORT 的校准数据接口）

**怎么做**：
1. 实现 CalibrationDataReader 子类
2. 调用 `onnxruntime.quantization.quantize_static()`
3. 扫描多种配置组合：
   - 对称 vs 非对称
   - per-tensor vs per-channel
   - MinMax vs Entropy（ORT 内置）

**运行**：
```bash
python scripts/quantize_ptq.py --workload 2d --sweep
```

---

### 3.4 Step 2.4-2.5 — INT8 对比测试与逐层分析

**文件**：`scripts/bench.py`（扩展）+ `src/quant/analysis.py`

**要学什么**：
- FP32 vs INT8 的性能对比方法
- 逐层敏感度分析：哪一层量化后精度掉得最多？
- SQNR（信号量化噪声比）作为量化质量指标

**怎么做**：
1. 扩展 `bench.py`，支持对比多个 ONNX 模型（FP32 + 多个 INT8 变体）
2. 实现 `analyze_per_layer_sensitivity()`：逐层量化（Leave-one-out），观察精度变化
3. 找出最敏感的层，思考为什么

---

### 3.5 Step 2.6 — 可视化

**文件**：`scripts/visualize.py`

**怎么做**：
1. `plot_accuracy_comparison()`：柱状图，X 轴是策略名，Y 轴是 accuracy
2. `plot_latency_comparison()`：柱状图，对比 FP32 vs 各 INT8 变体的延迟
3. `plot_accuracy_vs_latency()`：散点图，X 轴 latency，Y 轴 accuracy，每个点是一种策略

### Phase 2 检查点

- [ ] 手写 quantize_utils.py 全部通过 self-test
- [ ] 至少对比 4 种量化策略组合
- [ ] 有一张 FP32 vs INT8 对比表（accuracy + latency + size）
- [ ] 能回答："为什么 per-channel 比 per-tensor 精度更高？"
- [ ] 能回答："为什么 activations 用非对称量化更好？"

```
# 理想输出示例：
# Strategy               | Acc   | Latency | Size
# ─────────────────────────────────────────────────
# FP32 baseline          | 76.3% | 0.85ms  | 0.48MB
# INT8-sym-perchannel    | 75.8% | 0.42ms  | 0.14MB
# INT8-asym-pertensor    | 74.1% | 0.40ms  | 0.14MB
# INT8-sym-pertensor     | 73.2% | 0.39ms  | 0.14MB
```

---

## 4. Phase 3：定点闭环 — Part C（第 4-7 周，核心差异化）

> **目标**：从"会用量化工具"升级到"理解硬件级 INT8 推理的每一步数值运算"。
> **这是项目最重要的部分**。它把你和"只会调 API"的人区分开来。

### 4.1 Step 3.1 — MAC 位宽增长推导

**文件**：`src/fixed_point/bit_growth.py`

**要学什么**：

#### 核心公式
```
INT8 × INT8 → INT16（乘法：M+N 位）
K 次累加 → 需要额外 ceil(log2(K)) 位
总位宽 = 16 + ceil(log2(K))
```

#### 为什么重要
如果你设计一个 NPU，累加器用多少位？
- 用太少位 → 溢出，精度崩溃
- 用太多位 → 面积和功耗浪费

| 层类型 | 卷积核 | C_in | K 值 | 需要位数 | 选择 |
|--------|--------|------|------|---------|------|
| Conv2d 3×3, C_in=3 | 3×3 | 3 | 27 | 21 | INT32 够 |
| Conv2d 3×3, C_in=64 | 3×3 | 64 | 576 | 26 | INT32 够 |
| Conv2d 3×3, C_in=512 | 3×3 | 512 | 4608 | 29 | INT32 够 |
| FC, C_in=4096 | 1×1 | 4096 | 4096 | 28 | INT32 够 |

**结论**：对于绝大多数 CNN 层，INT32 累加器绰绰有余（INT32 有 31 位有效位）。

**怎么做**：
1. 在纸上推导公式
2. 实现 `compute_mac_bit_growth()`、`recommend_accumulator_width()`
3. 用你的两个 workload 验证

**自测**：
```bash
python -m src.fixed_point.bit_growth
```

---

### 4.2 Step 3.2 — 溢出策略对比

**文件**：`src/fixed_point/overflow.py`

**要学什么**：

#### 两种溢出处理
1. **饱和（Saturation）**：超出范围就钳位到最大/最小值
   - 硬件代价：2 个比较器 + 1 个 MUX（很便宜）
   - 效果：误差有界，最多差一个量化范围
2. **回绕（Wrap-around）**：补码取模，自然溢出
   - 硬件代价：零（不需要额外电路）
   - 效果：可能灾难性翻转（正数变负数）

#### 实验预期

| 累加器位宽 | Saturation 误差 | Wrap-around 误差 | 结论 |
|-----------|---------------|-----------------|------|
| INT32 | ≈0 | ≈0 | 无溢出，两者一样 |
| INT24 | 很小 | 较小 | Saturation 略好 |
| INT16 | 小 | **极大** | Saturation 必须，Wrap 灾难 |

**怎么做**：
1. 实现 `saturate()` 和 `wrap_around()`
2. 实现 `simulate_conv_accumulation()`：用真实权重模拟 MAC 过程
3. 在不同位宽下对比两种策略

---

### 4.3 Step 3.3 — 手写 INT8 卷积核（最核心的文件）

**文件**：`src/fixed_point/int8_kernel.py`

**要学什么**：

#### INT8 推理的完整数据路径
```
                 ┌──────────────────────────────────┐
                 │  完整 INT8 Conv 推理流程          │
                 └──────────────────────────────────┘

  x_fp32 ──quantize──→ x_int8 ──┐
                                 │
  w_fp32 ──quantize──→ w_int8 ──┤──→ INT8×INT8 MAC ──→ acc_int32
                                 │
  b_fp32 ──quantize──→ b_int32 ─┘        │
                                          │
                                    requantize
                                          │
                                          ▼
                                      y_int8 ──→ 下一层输入
```

#### 重量化（Requantize）的关键
```
M = (s_x × s_w) / s_y      ← 组合缩放因子
y_int8 = round(acc_int32 × M) + zp_y
y_int8 = clamp(y_int8, -128, 127)
```

M 的物理意义：累加器的单位是 `s_x × s_w`，输出的单位是 `s_y`，M 就是单位转换因子。

#### Bias 处理
```
bias_int32 = round(bias_fp32 / (s_x × s_w))
```
Bias 必须用 `s_x × s_w` 作为 scale，这样才能和 MAC 结果直接相加。

**怎么做**（按顺序，不要跳步）：
1. **先实现 `requantize()`**：最简单，3 行核心代码
2. **再实现 `quantize_bias()`**：也很简单
3. **最后实现 `int8_conv2d_reference()`**：7 层 for 循环，慢但清晰
   - 先用最简单的 1×1 conv 验证
   - 再用 3×3 conv 验证
   - 最后用真实模型权重验证
4. 实现 `validate_against_float()`：对比 INT8 和 FP32 的输出

**自测**：
```bash
python -m src.fixed_point.int8_kernel
```

**踩坑提醒**：
- **Padding 值**：量化后的 "0" 不是整数 0，而是 `input_zp`。如果 padding 填 0 会引入误差
- **数据类型**：MAC 前一定要 cast 到 `int32`，否则 int8 × int8 会溢出
- **requantize 精度**：用 `float64` 做 M 的乘法，`float32` 可能丢精度
- **先写正确的慢版本**：7 层 for 循环很慢，但意图清晰。不要急着优化

---

### 4.4 Step 3.4 — Golden Reference 测试

**文件**：`src/fixed_point/tests.py`

**要学什么**：
- 如何设计数值验证的单元测试
- "Golden Reference" 的概念：用高精度（FP32）结果作为参考，检查定点结果的误差

**怎么做**：
1. 按注释实现每个 test 函数
2. 从简单到复杂：位宽计算 → 溢出策略 → 简单卷积 → 随机卷积 → 真实权重

**运行**：
```bash
python -m src.fixed_point.tests
# 所有测试应该 PASS
```

**验收标准**：>95% 的输出与 FP32 参考值差异在 1 个量化步长以内。

---

### 4.5 Step 3.5 — 撰写定点推导笔记

**文件**：`docs/fixed_point_note.md`

这是作品集的核心交付物之一。把你在 Step 3.1-3.4 学到的东西，用 3-5 页整理成文档。

**写作顺序**：
1. 引言：为什么硬件偏好定点？INT8 MAC 面积是 FP32 的 1/30
2. MAC 位宽增长：推导 + 你的 workload 的实际数据
3. 累加器位宽：为什么 INT32 是标准选择
4. 溢出策略：实验数据 + 结论
5. Requantize 流程：完整数据路径图
6. 实验验证：测试结果表
7. 结论与设计指南

### Phase 3 检查点

- [ ] `python -m src.fixed_point.tests` 所有测试 PASS
- [ ] `docs/fixed_point_note.md` 写了 3-5 页
- [ ] 能回答："INT8 × INT8 累加 576 次，需要多少位？为什么 INT32 够？"
- [ ] 能回答："Saturation 和 Wrap-around 的区别？硬件代价？"
- [ ] 能在白板上画出 INT8 推理的完整数据路径

---

## 5. Phase 4：架构评估 — Part D（第 7-12 周）

> **目标**：用标准工具（SCALE-Sim）把你的模型映射到硬件架构上，做 DSE 分析。
> **核心能力**：从"画图说故事"升级到"用数据做架构决策"。

### 5.1 Step 4.1 — 安装 SCALE-Sim 并理解概念

```bash
pip install scalesim
```

**要学什么**：

#### 脉动阵列（Systolic Array）
```
  W W W W W ──→
  │ │ │ │ │
  PE PE PE PE PE ──→ 输出
  PE PE PE PE PE ──→
  PE PE PE PE PE ──→
  ↓
  输入数据
```
- 每个 PE 做一次 MAC 操作
- 数据在 PE 之间"流动"（systolic = 心脏般脉动）
- 阵列越大，并行度越高

#### 三种数据流（Dataflow）
| 数据流 | 哪个数据固定在 PE 里 | 适合场景 |
|--------|-------------------|---------|
| WS (Weight Stationary) | 权重 | 大卷积核、深 channel |
| OS (Output Stationary) | 部分和（输出） | 大 output feature map |
| IS (Input Stationary) | 输入特征 | 多滤波器共享输入 |

#### Roofline 模型
```
性能                          ┌──── compute-bound
(ops/s)                      │
  │                     ────┘
  │                ───
  │           ───
  │      ───        memory-bound
  │ ───
  └────────────────────────────
        算术强度 (ops/byte)
```

- 算术强度 = 计算量 / 数据搬运量
- 转折点 = peak_compute / peak_bandwidth
- 在转折点左边 → memory-bound（加带宽/SRAM 有用）
- 在转折点右边 → compute-bound（加 PE 有用）

---

### 5.2 Step 4.2 — SCALE-Sim Wrapper

**文件**：`src/arch/scalesim_runner.py`

**怎么做**：
1. `generate_topology_csv()`：遍历 PyTorch 模型的所有 Conv 层，提取参数（输入尺寸、卷积核、通道数等），写入 SCALE-Sim 格式的 CSV
2. `generate_scalesim_config()`：生成 `.cfg` 配置文件（PE 阵列大小、SRAM 大小、数据流）
3. `run_scalesim()`：调用 SCALE-Sim 运行仿真
4. `parse_compute_report()`：解析输出报告，提取周期数和 PE 利用率

---

### 5.3 Step 4.3 — 设计空间探索（DSE）

**文件**：`src/arch/dse.py`

**怎么做**：
1. 定义设计空间：
   - PE 阵列大小：8×8, 16×16, 32×32
   - SRAM 大小：32KB, 64KB, 128KB, 256KB
   - 数据流：WS, OS, IS
2. 生成所有配置组合
3. 逐个运行仿真
4. 分析结果：找趋势、找瓶颈、找最优配置

**分析要回答的问题**：
- "array 从 8×8 加到 16×16，性能提升了多少？到 32×32 呢？为什么有收益递减？"
- "这个 workload 在 WS 下是 memory-bound 还是 compute-bound？"
- "增加 SRAM 到 256KB 后，瓶颈从 memory 转到 compute 了吗？"

---

### 5.4 Step 4.4-4.6 — 运行、可视化、报告

```bash
python scripts/run_arch_eval.py
python scripts/visualize.py  # 生成 DSE 趋势图
```

**最终报告**：填写 `report.md`，包含所有 Part 的结果和分析。

### Phase 4 检查点

- [ ] SCALE-Sim 仿真成功运行
- [ ] 至少 2 个架构配置 + 2 种数据流对比
- [ ] 有 DSE 趋势图（周期 vs 阵列大小 / SRAM 大小）
- [ ] 能回答："为什么 16×16 比 8×8 快，但 32×32 没有快那么多？"
- [ ] `report.md` 完成

---

## 6. 贯穿全程：论文阅读（每周 2-3 小时）

**文件**：`docs/paper_notes.md`

### 阅读顺序与时间建议

| 周次 | 论文 | 为什么读它 | 对应 Part |
|------|------|-----------|----------|
| 1-2 | Eyeriss (ISCA 2016) | 建立 dataflow & 能效的核心直觉 | Part D |
| 3-4 | Timeloop (ISPASS 2019) | 理解 mapping 和系统化评估方法 | Part D |
| 5-7 | Gemmini (DAC 2021) | 了解完整的加速器系统（PE + 控制 + 内存） | Part C+D |
| 8-10 | NVDLA | 了解工业级设计取舍 | Part C+D |

### 笔记模板

每篇论文写 10-15 行笔记，回答四个问题：
1. **Problem**：这篇论文解决什么问题？
2. **Method**：用了什么方法？核心创新是什么？
3. **Key Insight**：最重要的一个洞察是什么？
4. **Relevance**：这和我的项目有什么关系？我能怎么用？

---

## 7. 时间线总览

```
Week 0     ┃ 环境搭建、通读项目
Week 1-2   ┃ Part A：模型训练、ONNX 导出、FP32 基线
            ┃ ├── 论文：开始读 Eyeriss
Week 2-4   ┃ Part B：手写量化、ORT PTQ 扫描、策略对比
            ┃ ├── 论文：读 Timeloop
Week 4-7   ┃ Part C：定点推导、INT8 卷积核、Golden Test
            ┃ ├── 论文：读 Gemmini
            ┃ ├── 撰写 fixed_point_note.md
Week 7-12  ┃ Part D：SCALE-Sim、DSE、趋势分析
            ┃ ├── 论文：读 NVDLA
            ┃ ├── 撰写最终 report.md
            ┃ └── 整理 GitHub repo
```

---

## 8. 每日/每周工作流建议

### 每天开始时
1. `git pull`（如果多设备）
2. 看 TODO list，确认今天要做哪个 Step
3. 打开对应文件，阅读 TODO 注释

### 每个 Step 的工作流
1. **读**：阅读文件头 docstring + TODO 注释（理解要做什么、为什么做）
2. **想**：在纸上画图/推公式（特别是 Part C）
3. **写**：实现代码
4. **测**：运行 self-test 或单元测试
5. **提交**：`git commit`

### 每个 Phase 结束时
1. 回顾检查点清单
2. 自问"面试问这个 Part 的问题，我能答吗？"
3. 把结果整理进 `results/` 和 `report.md`
4. 推送到 GitHub

---

## 9. 常见踩坑与解决方案

### Part A

| 问题 | 原因 | 解决 |
|------|------|------|
| ONNX 导出 shape 不对 | 没有 `model.eval()` | 导出前调用 `model.eval()` |
| ORT 和 PyTorch 输出差异大 | BN 在 training mode | 确保 eval mode |
| 训练 loss 不下降 | 学习率太大/太小 | 2D CNN 建议 lr=0.01, 1D CNN 建议 lr=0.001 |

### Part B

| 问题 | 原因 | 解决 |
|------|------|------|
| 量化后精度暴跌 | Per-tensor 太粗糙 | 换 per-channel |
| SQNR 明显低于 48 dB | Outlier 拉偏 scale | 用 percentile 校准 |
| ORT quantize_static 报错 | CalibrationDataReader 格式 | 确保输出 dict，key 是输入名 |

### Part C

| 问题 | 原因 | 解决 |
|------|------|------|
| INT8 Conv 输出全错 | Padding 用了 0 而不是 zp | Padding 值应为 `input_zp` |
| Requantize 精度差 | 用了 float32 算 M | 改用 float64 |
| MAC 溢出 | 没有 cast 到 int32 | 乘法前 `.astype(np.int32)` |

### Part D

| 问题 | 原因 | 解决 |
|------|------|------|
| SCALE-Sim 安装失败 | Python 版本不兼容 | 用 Python 3.10 |
| 仿真结果全是 0 | topology CSV 格式错误 | 检查列名和分隔符 |
| PE 利用率很低 | Workload 太小配 array 太大 | 减小 array size 或换更大 workload |

---

## 10. 最终交付物清单

完成全部 4 个 Phase 后，你应该拥有：

- [ ] **GitHub Repo**：可复现，README 完整，结构清晰
- [ ] **`report.md`**：最终报告，包含所有 Part 的结果、图表、分析
- [ ] **`docs/fixed_point_note.md`**：3-5 页定点推导与验证笔记
- [ ] **`docs/paper_notes.md`**：4 篇论文的阅读笔记
- [ ] **`results/figures/`**：量化对比图、DSE 趋势图
- [ ] **`results/tables/`**：性能对比 CSV

### 简历可用 Bullet Points

做完这个项目后，你的简历可以写：

> - Built an end-to-end **INT8 inference pipeline** (PyTorch→ONNX→ORT), benchmarking FP32 vs INT8 across calibration and per-channel quantization settings.
> - Derived **fixed-point bit-width/overflow strategy** for INT8 MAC/accumulation (bit growth, saturation vs wrap) and validated numerical consistency with golden references.
> - Conducted **architecture-mapping exploration** using SCALE-Sim, comparing dataflows/tiling to identify memory vs compute bottlenecks.

---

> **最后一句话**：这个项目的价值不在于模型精度多高，而在于你能**解释每一步的 why**。面试时能画出 INT8 推理的数据路径、能说清楚 saturation 和 wrap-around 的区别、能用 roofline 解释瓶颈——这才是真正的竞争力。
