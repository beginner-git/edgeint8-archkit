"""
Module: run_all.py
Part: All
Purpose: Master orchestration script — run the entire pipeline end-to-end.

This script runs all phases of the project in sequence:
  Phase 1 (Part A): Train → Export ONNX → FP32 Benchmark
  Phase 2 (Part B): Quantize → INT8 Benchmark → Comparison
  Phase 3 (Part C): Fixed-point analysis → Tests
  Phase 4 (Part D): Architecture evaluation → DSE

一键跑通整个项目的脚本。适合最终验证和演示。

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --skip-train    # Skip if models already trained
    python scripts/run_all.py --part A        # Run only Part A
"""

import os
import sys
import subprocess
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_step(description, command):
    """
    Run a command and report status.

    Args:
        description: Human-readable step description
        command: Shell command to run
    """
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    start = time.time()

    try:
        result = subprocess.run(
            command, shell=True, cwd=PROJECT_ROOT,
            capture_output=True, text=True
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"  [OK] Completed in {elapsed:.1f}s")
            if result.stdout:
                # Print last 10 lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"    {line}")
        else:
            print(f"  [FAIL] Exit code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run EdgeINT8-ArchKit pipeline")
    parser.add_argument("--part", choices=["A", "B", "C", "D", "all"],
                        default="all", help="Which part to run")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training if checkpoints exist")
    args = parser.parse_args()

    print("=" * 60)
    print("  EdgeINT8-ArchKit: Full Pipeline")
    print("=" * 60)

    results = {}

    # ===== Part A: Train → Export → Benchmark =====
    if args.part in ("A", "all"):
        print("\n" + "="*60)
        print("  PART A: End-to-End Pipeline")
        print("="*60)

        if not args.skip_train:
            run_step("A.1 — Train 1D CNN",
                     "python -m models.train --workload 1d --epochs 10")
            run_step("A.2 — Train 2D CNN",
                     "python -m models.train --workload 2d --epochs 10")

        run_step("A.3 — Export ONNX (all workloads)",
                 "python scripts/export_onnx.py --workload all")

        run_step("A.4 — Benchmark FP32 (2D CNN)",
                 "python scripts/bench.py --workload 2d --model models/tiny_cnn_2d.onnx")

        run_step("A.5 — Benchmark FP32 (1D CNN)",
                 "python scripts/bench.py --workload 1d --model models/signal_cnn_1d.onnx")

    # ===== Part B: Quantize → Benchmark → Visualize =====
    if args.part in ("B", "all"):
        print("\n" + "="*60)
        print("  PART B: INT8 Quantization")
        print("="*60)

        run_step("B.1 — Quantize models (sweep)",
                 "python scripts/quantize_ptq.py --workload all --sweep")

        run_step("B.2 — Benchmark all models (2D)",
                 "python scripts/bench.py --workload 2d --all")

        run_step("B.3 — Generate quantization charts",
                 "python scripts/visualize.py --type quantization")

    # ===== Part C: Fixed-Point Tests =====
    if args.part in ("C", "all"):
        print("\n" + "="*60)
        print("  PART C: Fixed-Point Closure")
        print("="*60)

        run_step("C.1 — Run fixed-point tests",
                 "python -m src.fixed_point.tests")

    # ===== Part D: Architecture Evaluation =====
    if args.part in ("D", "all"):
        print("\n" + "="*60)
        print("  PART D: Architecture Evaluation")
        print("="*60)

        run_step("D.1 — Run architecture evaluation & DSE",
                 "python scripts/run_arch_eval.py")

        run_step("D.2 — Generate DSE charts",
                 "python scripts/visualize.py --type dse")

    print("\n" + "="*60)
    print("  Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
