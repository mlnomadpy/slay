#!/usr/bin/env python3
"""
Master Benchmark Runner for SLAY Paper (JAX/Flax)

Runs all benchmarks and generates paper-ready tables.

Usage:
    python kaggle/run_all_benchmarks.py --quick    # Quick validation (~5 min)
    python kaggle/run_all_benchmarks.py --full     # Full benchmarks (~30 min+)
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


def run_command(cmd: list[str], description: str, cwd: str = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd or Path(__file__).parent)
    elapsed = time.time() - start
    
    status = "[OK] SUCCESS" if result.returncode == 0 else "[X] FAILED"
    print(f"\n{status} ({elapsed:.1f}s)")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all SLAY JAX benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced configs")
    parser.add_argument("--full", action="store_true", help="Full benchmark suite")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["scaling", "kernel", "tasks", "ablation", "max_context", "feature_budget", "extreme"],
        help="Benchmarks to skip",
    )
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs("../tables", exist_ok=True)
    os.makedirs("../artifacts", exist_ok=True)
    
    print("=" * 70)
    print("SLAY JAX/Flax Benchmark Suite")
    print("=" * 70)
    print(f"Mode: {'quick' if args.quick else 'full' if args.full else 'default'}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Skipping: {args.skip if args.skip else 'none'}")
    print("=" * 70)
    
    python = sys.executable
    cwd = str(Path(__file__).parent)
    results = {}
    
    # 1. Scaling Benchmark (Latency & Memory)
    if "scaling" not in args.skip:
        cmd = [python, "benchmark_scaling.py"]
        if args.quick:
            cmd.append("--quick")
        results["scaling_forward"] = run_command(
            cmd, "Scaling Benchmark (Forward)", cwd=cwd
        )
        
        # With backward pass
        cmd_backward = cmd + ["--backward"]
        results["scaling_backward"] = run_command(
            cmd_backward, "Scaling Benchmark (Forward+Backward)", cwd=cwd
        )
    
    # 2. Kernel Approximation Quality
    if "kernel" not in args.skip:
        cmd = [python, "benchmark_kernel_quality.py"]
        if args.full:
            cmd.append("--sweep")
        results["kernel_quality"] = run_command(
            cmd, "Kernel Approximation Quality", cwd=cwd
        )
    
    # 3. Polynomial Ablation
    if "ablation" not in args.skip:
        cmd = [python, "ablation_poly_approx.py"]
        if args.full:
            cmd.append("--sweep")
        results["ablation"] = run_command(
            cmd, "Polynomial Ablation Study", cwd=cwd
        )
    
    # 4. Synthetic Tasks
    if "tasks" not in args.skip:
        cmd = [python, "benchmark_tasks.py"]
        if args.quick:
            cmd.append("--quick")
        results["tasks"] = run_command(
            cmd, "Synthetic Task Benchmarks", cwd=cwd
        )
    
    # 5. Maximum Context Length
    if "max_context" not in args.skip:
        cmd = [python, "benchmark_max_context.py"]
        results["max_context"] = run_command(
            cmd, "Maximum Context Length", cwd=cwd
        )
    
    # 6. Feature Budget Trade-off
        if args.full:
            cmd.append("--full")
        results["feature_budget"] = run_command(
            cmd, "Feature Budget Trade-off", cwd=cwd
        )

    # 7. Extreme Classification (Eurlex)
    if "extreme" not in args.skip and args.full:
        cmd = [python, "extreme.py"]
        results["extreme"] = run_command(
            cmd, "Extreme Classification (Eurlex) - SLAY/Yat Kernels", cwd=cwd
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    for name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {status} {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} passed")
    
    # List generated artifacts
    print("\n" + "=" * 70)
    print("GENERATED ARTIFACTS")
    print("=" * 70)
    
    for directory in ["../tables", "../artifacts"]:
        if os.path.exists(directory):
            files = os.listdir(directory)
            if files:
                print(f"\n{directory}/")
                for f in sorted(files):
                    print(f"  - {f}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
