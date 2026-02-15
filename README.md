# EdgeINT8-ArchKit

**INT8 端侧推理 + 定点闭环 + 架构映射评估（Mini Project）**

> 用一个"最小闭环项目"把量化、定点、推理工具链、架构评估串起来。
> 不依赖私有资源；可复现；适合写进简历；适合面试讲故事。

---

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate edgeint8

# OR use pip
pip install -r requirements.txt

# 2. Verify installation
python -c "import torch; import onnx; import onnxruntime; print('All dependencies OK')"

# 3. Run the full pipeline (after implementing all parts)
python scripts/run_all.py
```

---

## Project Structure

```
EdgeINT8-ArchKit/
├── README.md                    # <-- You are here
├── environment.yml              # Conda environment
├── requirements.txt             # Pip dependencies
├── report.md                    # Final project report (template)
│
├── models/                      # [Part A] Model definitions & training
│   ├── signal_cnn_1d.py         #   Workload-1: 1D CNN for signal classification
│   ├── tiny_cnn_2d.py           #   Workload-2: Tiny 2D CNN for CIFAR-10
│   └── train.py                 #   Training script for both workloads
│
├── data/                        # Data directory
│   ├── README.md                #   Data source documentation
│   └── download.py              #   Data download/verification utility
│
├── scripts/                     # Top-level executable scripts
│   ├── export_onnx.py           #   [Part A] PyTorch -> ONNX export
│   ├── bench.py                 #   [Part A+B] Unified benchmarking
│   ├── quantize_ptq.py          #   [Part B] INT8 PTQ with strategy sweep
│   ├── visualize.py             #   [Part B+D] Chart generation
│   ├── run_arch_eval.py         #   [Part D] Architecture evaluation entry
│   └── run_all.py               #   Run entire pipeline end-to-end
│
├── src/                         # Source modules
│   ├── quant/                   #   [Part B] Quantization
│   │   ├── quantize_utils.py    #     Core quantization math (hand-written)
│   │   ├── calibration.py       #     Activation calibration strategies
│   │   └── analysis.py          #     Per-layer error analysis
│   │
│   ├── fixed_point/             #   [Part C] Fixed-point closure
│   │   ├── bit_growth.py        #     MAC bit growth analysis
│   │   ├── overflow.py          #     Saturation vs wrap-around
│   │   ├── int8_kernel.py       #     Reference INT8 convolution kernel
│   │   └── tests.py             #     Golden reference unit tests
│   │
│   ├── arch/                    #   [Part D] Architecture evaluation
│   │   ├── scalesim_runner.py   #     SCALE-Sim wrapper
│   │   └── dse.py               #     Design Space Exploration
│   │
│   └── utils/                   #   Shared utilities
│       ├── data.py              #     Data loading (CIFAR-10 + synthetic signal)
│       ├── profiling.py         #     Timing & memory profiling
│       └── helpers.py           #     Misc helpers (seed, IO, formatting)
│
├── results/                     # Output directory
│   ├── tables/                  #   CSV result tables
│   └── figures/                 #   Generated charts & plots
│
└── docs/                        # Documentation
    ├── fixed_point_note.md      #   [Part C] Fixed-point derivation notes
    └── paper_notes.md           #   Paper reading notes template
```

---

## Learning Path (学习路径)

按以下顺序逐步实现，每个 Phase 都有明确的检查点。

### Phase 1: Part A — End-to-End Pipeline (第 1-2 周)

| Step | File | What to do |
|------|------|------------|
| 1.1 | `models/signal_cnn_1d.py` | Implement 1D CNN architecture |
| 1.2 | `models/tiny_cnn_2d.py` | Implement Tiny 2D CNN architecture |
| 1.3 | `src/utils/data.py` | Implement data loaders |
| 1.4 | `models/train.py` | Train both models, save .pth |
| 1.5 | `scripts/export_onnx.py` | Export to ONNX, verify consistency |
| 1.6 | `scripts/bench.py` | FP32 benchmark (latency/accuracy/size) |

**Checkpoint**: `python scripts/bench.py --workload 2d` outputs FP32 results

### Phase 2: Part B — INT8 PTQ (第 2-4 周)

| Step | File | What to do |
|------|------|------------|
| 2.1 | `src/quant/quantize_utils.py` | Hand-write quantization math |
| 2.2 | `src/quant/calibration.py` | Implement calibration strategies |
| 2.3 | `scripts/quantize_ptq.py` | ORT-based PTQ sweep |
| 2.4 | `scripts/bench.py` (extend) | INT8 vs FP32 comparison |
| 2.5 | `src/quant/analysis.py` | Per-layer sensitivity analysis |
| 2.6 | `scripts/visualize.py` | Accuracy/latency comparison charts |

**Checkpoint**: FP32 vs INT8 comparison table with data-driven conclusions

### Phase 3: Part C — Fixed-Point Closure (第 4-7 周)

| Step | File | What to do |
|------|------|------------|
| 3.1 | `src/fixed_point/bit_growth.py` | Derive MAC bit growth per layer |
| 3.2 | `src/fixed_point/overflow.py` | Compare saturation vs wrap-around |
| 3.3 | `src/fixed_point/int8_kernel.py` | Reference INT8 conv kernel |
| 3.4 | `src/fixed_point/tests.py` | Golden check unit tests |
| 3.5 | `docs/fixed_point_note.md` | 3-5 page derivation notes |

**Checkpoint**: `python -m src.fixed_point.tests` all tests pass

### Phase 4: Part D — Architecture Evaluation (第 7-12 周)

| Step | File | What to do |
|------|------|------------|
| 4.1 | Install SCALE-Sim | `pip install scalesim` |
| 4.2 | `src/arch/scalesim_runner.py` | Extract Conv params -> topology CSV |
| 4.3 | `src/arch/dse.py` | DSE sweep across configurations |
| 4.4 | `scripts/run_arch_eval.py` | Run full DSE |
| 4.5 | `scripts/visualize.py` (extend) | Cycle/energy trend plots |
| 4.6 | `report.md` | Complete final report |

**Checkpoint**: `python scripts/run_arch_eval.py` outputs DSE trend charts

---

## Workloads

| Workload | Type | Input Shape | Dataset | Purpose |
|----------|------|-------------|---------|---------|
| Workload-1 | 1D CNN | `[B, 1, 128]` | Synthetic signals | Signal classification, familiar domain |
| Workload-2 | Tiny 2D CNN | `[B, 3, 32, 32]` | CIFAR-10 | Standard benchmark, Conv/GEMM analysis |

---

## Key Deliverables (作品集输出)

1. **GitHub repo** — reproducible, well-documented
2. **`report.md`** — final project report with figures and tables
3. **`docs/fixed_point_note.md`** — fixed-point derivation and verification
4. **`results/`** — all benchmark data, charts, and analysis

---

## Resume Bullets (简历可用)

- Built an end-to-end **INT8 inference pipeline** (PyTorch->ONNX->ORT), benchmarking FP32 vs INT8 across calibration and per-channel quantization settings.
- Derived **fixed-point bit-width/overflow strategy** for INT8 MAC/accumulation (bit growth, saturation vs wrap) and validated numerical consistency with golden references.
- Conducted **architecture-mapping exploration** using SCALE-Sim / Timeloop+Accelergy, comparing dataflows/tiling to identify memory vs compute bottlenecks.
