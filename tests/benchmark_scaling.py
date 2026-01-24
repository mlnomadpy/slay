"""
Scaling Benchmark: Latency and Memory vs Sequence Length

Generates the core scaling tables and figures for the SLAY paper:
- Table: Attention-only latency at various sequence lengths
- Table: Peak memory at various sequence lengths
- Figure: Scaling curves (latency and memory vs L)

Run:
    python tests/benchmark_scaling.py --device cuda
    python tests/benchmark_scaling.py --device cuda --quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import get_attention_class, list_attention_types


@dataclass
class ScalingConfig:
    """Configuration for scaling benchmarks."""
    embed_dim: int = 64
    n_heads: int = 4
    batch_size: int = 1
    warmup_iters: int = 10
    bench_iters: int = 30
    
    # Sequence lengths to test
    seq_lengths: Tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 42
    
    # Yat-specific parameters
    yat_num_features: int = 32
    yat_num_quadrature_nodes: int = 2
    yat_poly_dim: int = 32
    yat_prf_dim: int = 16
    yat_poly_sketch_dim: int = 32
    yat_epsilon: float = 1e-6


# Attention groups for benchmarking
BASELINE_ATTENTIONS = ["standard", "performer", "linear", "cosformer"]
YAT_ATTENTIONS = [
    "yat-spherical",
    "yat-performer-anchor", 
    "yat-performer-laplace",
    "yat-performer",
]
ALL_PAPER_ATTENTIONS = BASELINE_ATTENTIONS + YAT_ATTENTIONS


def _torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping.get(dtype_str, torch.float32)


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_attention_kwargs(name: str, cfg: ScalingConfig) -> Dict[str, Any]:
    """Get constructor kwargs for each attention type."""
    if name in {"yat-performer", "yat-performer-hadamard"}:
        return {
            "num_features": cfg.yat_num_features,
            "num_quadrature_nodes": cfg.yat_num_quadrature_nodes,
            "epsilon": cfg.yat_epsilon,
        }
    if name in {"yat-performer-laplace", "yat-performer-laplace-only"}:
        return {
            "num_features": cfg.yat_num_features,
            "num_quadrature_nodes": cfg.yat_num_quadrature_nodes,
            "epsilon": cfg.yat_epsilon,
        }
    if name in {"yat-performer-anchor", "yat-performer-anchors"}:
        return {
            "num_prf_features": cfg.yat_prf_dim,
            "num_quadrature_nodes": cfg.yat_num_quadrature_nodes,
            "poly_dim": cfg.yat_poly_dim,
            "epsilon": cfg.yat_epsilon,
        }
    if name in {"yat-performer-tensor", "yat-performer-tensorsketch"}:
        return {
            "num_prf_features": cfg.yat_prf_dim,
            "num_quadrature_nodes": cfg.yat_num_quadrature_nodes,
            "poly_sketch_dim": cfg.yat_poly_sketch_dim,
            "epsilon": cfg.yat_epsilon,
        }
    if name in {"yat-performer-rm", "yat-performer-random-maclaurin"}:
        return {
            "num_prf_features": cfg.yat_prf_dim,
            "num_quadrature_nodes": cfg.yat_num_quadrature_nodes,
            "poly_dim": cfg.yat_poly_dim,
            "epsilon": cfg.yat_epsilon,
        }
    if name == "yat-performer-nystrom":
        return {
            "num_prf_features": cfg.yat_prf_dim,
            "num_quadrature_nodes": cfg.yat_num_quadrature_nodes,
            "poly_dim": cfg.yat_poly_dim,
            "epsilon": cfg.yat_epsilon,
        }
    if name in {"yat-spherical", "yat-exact-spherical"}:
        return {"epsilon": cfg.yat_epsilon}
    if name == "performer":
        return {"kernel_size": 64}
    return {}


def benchmark_latency(
    module: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    backward: bool = False,
) -> float:
    """Benchmark forward (and optionally backward) latency in milliseconds."""
    module.train() if backward else module.eval()
    
    # Warmup
    for _ in range(warmup):
        if backward:
            y = module(x)
            loss = y.sum()
            loss.backward()
        else:
            with torch.no_grad():
                _ = module(x)
    
    _sync_cuda()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        if backward:
            y = module(x)
            loss = y.sum()
            loss.backward()
        else:
            with torch.no_grad():
                _ = module(x)
    _sync_cuda()
    
    return (time.perf_counter() - start) / iters * 1000.0


def benchmark_memory(
    module: nn.Module,
    x: torch.Tensor,
    backward: bool = False,
) -> Optional[float]:
    """Benchmark peak memory in MB. Returns None if not on CUDA."""
    if not torch.cuda.is_available() or not x.is_cuda:
        return None
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    module.train() if backward else module.eval()
    
    if backward:
        y = module(x)
        loss = y.sum()
        loss.backward()
    else:
        with torch.no_grad():
            _ = module(x)
    
    _sync_cuda()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def run_scaling_benchmark(
    cfg: ScalingConfig,
    attention_names: List[str],
    backward: bool = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Run scaling benchmarks for multiple attention types.
    
    Returns:
        results[attention_name][seq_len] = {
            "latency_ms": float,
            "memory_mb": float or None,
            "status": "ok" | "oom" | "error"
        }
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dtype = _torch_dtype(cfg.dtype)
    
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    for name in attention_names:
        results[name] = {}
        
        try:
            attn_cls = get_attention_class(name)
        except ValueError:
            print(f"[SKIP] Unknown attention: {name}")
            continue
        
        kwargs = get_attention_kwargs(name, cfg)
        
        for seq_len in cfg.seq_lengths:
            print(f"  {name} @ L={seq_len}...", end=" ", flush=True)
            
            try:
                # Create module and input
                module = attn_cls(cfg.embed_dim, cfg.n_heads, **kwargs).to(device)
                x = torch.randn(cfg.batch_size, seq_len, cfg.embed_dim, device=device, dtype=dtype)
                
                if backward:
                    x = x.requires_grad_(True)
                
                # Benchmark latency
                latency_ms = benchmark_latency(module, x, cfg.warmup_iters, cfg.bench_iters, backward)
                
                # Benchmark memory
                memory_mb = benchmark_memory(module, x, backward)
                
                results[name][seq_len] = {
                    "latency_ms": latency_ms,
                    "memory_mb": memory_mb,
                    "status": "ok",
                }
                
                print(f"OK {latency_ms:.2f}ms, {memory_mb:.1f}MB" if memory_mb else f"OK {latency_ms:.2f}ms")
                
                # Cleanup
                del module, x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                results[name][seq_len] = {"latency_ms": float("nan"), "memory_mb": float("nan"), "status": "oom"}
                print("OOM")
                torch.cuda.empty_cache()
                
            except Exception as e:
                results[name][seq_len] = {"latency_ms": float("nan"), "memory_mb": float("nan"), "status": "error"}
                print(f"ERROR: {e}")
    
    return results


def format_latex_scaling_table(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    metric: str = "latency_ms",
    caption: str = "",
    label: str = "",
) -> str:
    """Format results as a LaTeX table."""
    
    # Get all sequence lengths
    all_seq_lens = sorted(set(
        seq_len for attn_results in results.values() for seq_len in attn_results.keys()
    ))
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    
    # Column spec
    col_spec = "l" + "r" * len(all_seq_lens)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header
    header = "Method"
    for seq_len in all_seq_lens:
        if seq_len >= 1024:
            header += f" & {seq_len // 1024}K"
        else:
            header += f" & {seq_len}"
    header += r"\\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    unit = "ms" if metric == "latency_ms" else "MB"
    
    for name, attn_results in results.items():
        row = name.replace("_", "\\_")
        for seq_len in all_seq_lens:
            if seq_len in attn_results:
                val = attn_results[seq_len].get(metric, float("nan"))
                status = attn_results[seq_len].get("status", "ok")
                if status == "oom":
                    row += " & OOM"
                elif status == "error" or math.isnan(val):
                    row += " & --"
                else:
                    if metric == "memory_mb":
                        if val >= 1024:
                            row += f" & {val/1024:.1f}GB"
                        else:
                            row += f" & {val:.0f}"
                    else:
                        row += f" & {val:.1f}"
            else:
                row += " & --"
        row += r"\\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def format_markdown_table(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    metric: str = "latency_ms",
) -> str:
    """Format results as a Markdown table."""
    
    all_seq_lens = sorted(set(
        seq_len for attn_results in results.values() for seq_len in attn_results.keys()
    ))
    
    lines = []
    
    # Header
    header = "| Method |"
    sep = "|---|"
    for seq_len in all_seq_lens:
        if seq_len >= 1024:
            header += f" {seq_len // 1024}K |"
        else:
            header += f" {seq_len} |"
        sep += "---:|"
    lines.append(header)
    lines.append(sep)
    
    # Data rows
    for name, attn_results in results.items():
        row = f"| {name} |"
        for seq_len in all_seq_lens:
            if seq_len in attn_results:
                val = attn_results[seq_len].get(metric, float("nan"))
                status = attn_results[seq_len].get("status", "ok")
                if status == "oom":
                    row += " OOM |"
                elif status == "error" or math.isnan(val):
                    row += " -- |"
                else:
                    if metric == "memory_mb" and val >= 1024:
                        row += f" {val/1024:.1f}GB |"
                    elif metric == "memory_mb":
                        row += f" {val:.0f}MB |"
                    else:
                        row += f" {val:.1f}ms |"
            else:
                row += " -- |"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scaling benchmark for attention mechanisms")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer sequence lengths")
    parser.add_argument("--backward", action="store_true", help="Include backward pass")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--attentions",
        nargs="*",
        default=None,
        help="Attention types to benchmark (default: paper-relevant subset)",
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all attention types")
    args = parser.parse_args()
    
    # Configure
    cfg = ScalingConfig(
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )
    
    if args.quick:
        cfg.seq_lengths = (256, 512, 1024, 2048, 4096)
        cfg.warmup_iters = 3
        cfg.bench_iters = 10
    
    # Select attention types
    if args.attentions:
        attention_names = args.attentions
    elif args.all:
        attention_names = list_attention_types()
    else:
        attention_names = ALL_PAPER_ATTENTIONS
    
    print("=" * 70)
    print("SLAY Scaling Benchmark")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"Dtype: {cfg.dtype}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Embed dim: {cfg.embed_dim}")
    print(f"Heads: {cfg.n_heads}")
    print(f"Sequence lengths: {cfg.seq_lengths}")
    print(f"Backward pass: {args.backward}")
    print(f"Attention types: {attention_names}")
    print("=" * 70)
    
    # Run benchmark
    mode = "forward+backward" if args.backward else "forward-only"
    print(f"\nRunning {mode} benchmark...")
    results = run_scaling_benchmark(cfg, attention_names, backward=args.backward)
    
    # Print results
    print("\n" + "=" * 70)
    print("LATENCY RESULTS (ms)")
    print("=" * 70)
    print(format_markdown_table(results, "latency_ms"))
    
    if cfg.device.startswith("cuda"):
        print("\n" + "=" * 70)
        print("MEMORY RESULTS (MB)")
        print("=" * 70)
        print(format_markdown_table(results, "memory_mb"))
    
    # Save results
    os.makedirs("tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    # LaTeX tables
    suffix = "_backward" if args.backward else "_forward"
    
    latency_tex = format_latex_scaling_table(
        results,
        metric="latency_ms",
        caption=f"Attention latency ({mode}) vs sequence length (ms).",
        label=f"tab:latency-scaling{suffix}",
    )
    with open(f"tables/latency_scaling{suffix}.tex", "w") as f:
        f.write(latency_tex)
    print(f"\nWrote tables/latency_scaling{suffix}.tex")
    
    if cfg.device.startswith("cuda"):
        memory_tex = format_latex_scaling_table(
            results,
            metric="memory_mb",
            caption=f"Peak memory ({mode}) vs sequence length.",
            label=f"tab:memory-scaling{suffix}",
        )
        with open(f"tables/memory_scaling{suffix}.tex", "w") as f:
            f.write(memory_tex)
        print(f"Wrote tables/memory_scaling{suffix}.tex")
    
    # JSON dump for reproducibility
    json_path = f"artifacts/scaling_results{suffix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "results": results,
        }, f, indent=2, default=str)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
