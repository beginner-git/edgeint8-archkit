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

    # =========================================================================
    # TODO: Implement the full pipeline orchestration
    #
    # Part A:
    #   1. Train models:     python -m models.train --workload all --epochs 10
    #   2. Export ONNX:       python scripts/export_onnx.py --workload all
    #   3. FP32 Benchmark:   python scripts/bench.py --workload 2d --model models/tiny_cnn_2d.onnx
    #
    # Part B:
    #   4. Quantize:         python scripts/quantize_ptq.py --workload all --sweep
    #   5. INT8 Benchmark:   python scripts/bench.py --workload 2d --all
    #   6. Visualize:        python scripts/visualize.py --type quantization
    #
    # Part C:
    #   7. Fixed-point tests: python -m src.fixed_point.tests
    #
    # Part D:
    #   8. Architecture eval: python scripts/run_arch_eval.py
    #   9. DSE visualize:     python scripts/visualize.py --type dse
    #
    # Each step should be wrapped with run_step() for status reporting.
    # If any step fails, print a warning but continue (unless critical).
    # =========================================================================
    print("\nTODO: Implement pipeline orchestration")
    print("Run individual scripts manually for now:")
    print("  python -m models.train --workload all")
    print("  python scripts/export_onnx.py --workload all")
    print("  python scripts/bench.py --workload 2d --model models/tiny_cnn_2d.onnx")


if __name__ == "__main__":
    main()
