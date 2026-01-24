#!/usr/bin/env python3
"""
Master Benchmark Runner for SLAY Paper

Runs all benchmarks and generates paper-ready tables and figures.

Usage:
    python tests/run_all_benchmarks.py --quick          # Quick validation (~10 min)
    python tests/run_all_benchmarks.py --full           # Full benchmarks (~2-4 hours)
    python tests/run_all_benchmarks.py --device cuda    # Specify device
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start
    
    status = "[OK] SUCCESS" if result.returncode == 0 else "[X] FAILED"
    print(f"\n{status} ({elapsed:.1f}s)")
    
    return result.returncode == 0


def get_default_device() -> str:
    """Get default device, checking CUDA availability properly."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Run all SLAY benchmarks")
    parser.add_argument("--device", default=None, help="Device to use (auto-detected if not specified)")
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced configs")
    parser.add_argument("--full", action="store_true", help="Full benchmark suite")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["scaling", "kernel", "tasks", "ablation"],
        help="Benchmarks to skip",
    )
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs("tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    print("=" * 70)
    print("SLAY Benchmark Suite")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Mode: {'quick' if args.quick else 'full' if args.full else 'default'}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)
    
    python = sys.executable
    results = {}
    
    # 1. Scaling Benchmark (Latency & Memory)
    if "scaling" not in args.skip:
        cmd = [python, "tests/benchmark_scaling.py", "--device", args.device]
        if args.quick:
            cmd.append("--quick")
        results["scaling_forward"] = run_command(cmd, "Scaling Benchmark (Forward)")
        
        # With backward pass
        cmd_backward = cmd + ["--backward"]
        results["scaling_backward"] = run_command(cmd_backward, "Scaling Benchmark (Forward+Backward)")
    
    # 2. Kernel Approximation Quality
    if "kernel" not in args.skip:
        cmd = [python, "tests/benchmark_kernel_quality.py", "--device", args.device]
        if args.full:
            cmd.append("--sweep")
        results["kernel_quality"] = run_command(cmd, "Kernel Approximation Quality")
    
    # 3. Synthetic Tasks
    if "tasks" not in args.skip:
        cmd = [python, "tests/benchmark_tasks.py", "--device", args.device]
        if args.quick:
            cmd.append("--quick")
        results["tasks"] = run_command(cmd, "Synthetic Task Benchmarks")
    
    # 4. Polynomial Ablation
    if "ablation" not in args.skip:
        cmd = [python, "tests/ablation_poly_approx.py", "--device", args.device, "--sweep"]
        if args.quick:
            cmd.append("--quick")
        results["ablation"] = run_command(cmd, "Polynomial Approximation Ablation")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    for name, success in results.items():
        status = "[OK]" if success else "[X]"
        print(f"  {status} {name}")
    
    # List generated files
    print("\n" + "=" * 70)
    print("GENERATED FILES")
    print("=" * 70)
    
    for folder in ["tables", "artifacts", "figures"]:
        if os.path.exists(folder):
            files = sorted(os.listdir(folder))
            if files:
                print(f"\n{folder}/")
                for f in files:
                    path = os.path.join(folder, f)
                    size = os.path.getsize(path)
                    print(f"  - {f} ({size:,} bytes)")
    
    # Save run metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "mode": "quick" if args.quick else "full" if args.full else "default",
        "results": results,
    }
    with open("artifacts/benchmark_run.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    # Return non-zero if any benchmark failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
