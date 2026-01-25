#!/usr/bin/env python3
"""
JAX/Flax Kernel Approximation Quality Benchmark

Validates that linearized SLAY variants approximate the exact spherical kernel.

Metrics:
- Relative L2 error: ||y_approx - y_exact|| / ||y_exact||
- Cosine similarity: cos(y_approx, y_exact)
- MSE: mean((y_approx - y_exact)^2)

Run:
    python kaggle/benchmark_kernel_quality.py
    python kaggle/benchmark_kernel_quality.py --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from main import (
    StandardAttention,
    YatSphericalAttention,
    SLAYAttention,
    SLAYTensorAttention,
    SLAYLaplaceAttention,
    SLAYRMAttention,
    SLAYNystromAttention,
    SLAYAnchorAttention,
    precompute_freqs_cis,
    mesh,
)


@dataclass
class KernelConfig:
    """Configuration for kernel quality benchmarks."""
    embed_dim: int = 64
    num_heads: int = 4
    batch_size: int = 4
    
    # Test configurations
    seq_lengths: tuple = (64, 128, 256, 512)
    
    seed: int = 42
    
    # SLAY hyperparameters to sweep
    num_features_options: tuple = (16, 32, 64)
    num_quad_options: tuple = (1, 2, 4)
    poly_dim_options: tuple = (8, 16, 32)


# Reference (exact) and linearized variants
EXACT_ATTENTION = "yat-spherical"

LINEARIZED_VARIANTS = {
    "slay": SLAYAttention,
    "slay-laplace": SLAYLaplaceAttention,
    "slay-tensor": SLAYTensorAttention,
    "slay-rm": SLAYRMAttention,
    "slay-nystrom": SLAYNystromAttention,
    "slay-anchor": SLAYAnchorAttention,
}


def compute_metrics(y_approx: jnp.ndarray, y_exact: jnp.ndarray) -> Dict[str, float]:
    """Compute approximation quality metrics."""
    
    # Flatten for metrics
    y_approx_flat = y_approx.reshape(-1)
    y_exact_flat = y_exact.reshape(-1)
    
    # Relative L2 error
    l2_diff = jnp.linalg.norm(y_approx_flat - y_exact_flat)
    l2_exact = jnp.linalg.norm(y_exact_flat)
    rel_l2 = float(l2_diff / (l2_exact + 1e-10))
    
    # Cosine similarity
    dot_prod = jnp.dot(y_approx_flat, y_exact_flat)
    norm_prod = jnp.linalg.norm(y_approx_flat) * jnp.linalg.norm(y_exact_flat)
    cos_sim = float(dot_prod / (norm_prod + 1e-10))
    
    # MSE
    mse = float(jnp.mean((y_approx_flat - y_exact_flat) ** 2))
    
    return {
        "rel_l2": rel_l2,
        "cos_sim": cos_sim,
        "mse": mse,
    }


def benchmark_single_config(
    cfg: KernelConfig,
    seq_len: int,
    variant_name: str,
    variant_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Benchmark a single linearized variant against exact attention."""
    
    with mesh:
        rngs = nnx.Rngs(cfg.seed)
        key = jax.random.PRNGKey(cfg.seed)
    
        head_dim = cfg.embed_dim // cfg.num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
    
        # Create input
        x = jax.random.normal(key, (cfg.batch_size, seq_len, cfg.embed_dim))
    
        # Create exact attention
        exact_attn = YatSphericalAttention(cfg.embed_dim, cfg.num_heads, rngs=rngs, epsilon=1e-6)
    
        # Create linearized variant
        rngs_approx = nnx.Rngs(cfg.seed)
        variant_cls = LINEARIZED_VARIANTS[variant_name]
        approx_attn = variant_cls(cfg.embed_dim, cfg.num_heads, rngs=rngs_approx, **variant_kwargs)
    
    # Forward pass (JIT compiled)
        @jax.jit
        def exact_forward(attn, inp, fc, fs):
            return attn(inp, fc, fs)
    
        @jax.jit
        def approx_forward(attn, inp, fc, fs):
            return attn(inp, fc, fs)
    
        # Compute outputs
        y_exact = exact_forward(exact_attn, x, freqs_cos, freqs_sin)
        jax.block_until_ready(y_exact)
    
        start = time.perf_counter()
        y_approx = approx_forward(approx_attn, x, freqs_cos, freqs_sin)
        jax.block_until_ready(y_approx)
        latency_ms = (time.perf_counter() - start) * 1000
    
        # Compute metrics
        metrics = compute_metrics(y_approx, y_exact)
        metrics["latency_ms"] = latency_ms
    
    return metrics


def run_kernel_benchmark(cfg: KernelConfig, sweep: bool = False) -> List[Dict[str, Any]]:
    """Run kernel approximation quality benchmarks."""
    
    results = []
    
    if sweep:
        # Full sweep over hyperparameters
        configs_to_test = []
        
        for num_feat in cfg.num_features_options:
            for num_quad in cfg.num_quad_options:
                # SLAY and SLAY-Laplace
                configs_to_test.append({
                    "variant": "slay",
                    "kwargs": {"num_features": num_feat, "num_quadrature_nodes": num_quad},
                    "label": f"SLAY (F={num_feat}, Q={num_quad})",
                })
                configs_to_test.append({
                    "variant": "slay-laplace",
                    "kwargs": {"num_features": num_feat, "num_quadrature_nodes": num_quad},
                    "label": f"SLAY-Laplace (F={num_feat}, Q={num_quad})",
                })
        
        for poly_dim in cfg.poly_dim_options:
            for num_quad in cfg.num_quad_options:
                # Polynomial variants
                for variant in ["slay-rm", "slay-nystrom", "slay-anchor"]:
                    configs_to_test.append({
                        "variant": variant,
                        "kwargs": {
                            "num_prf_features": 8,
                            "num_quadrature_nodes": num_quad,
                            "poly_dim": poly_dim,
                        },
                        "label": f"{variant.upper()} (P={poly_dim}, Q={num_quad})",
                    })
    else:
        # Default configurations
        configs_to_test = [
            {"variant": "slay", "kwargs": {"num_features": 32, "num_quadrature_nodes": 2}, "label": "SLAY"},
            {"variant": "slay-laplace", "kwargs": {"num_features": 32, "num_quadrature_nodes": 2}, "label": "SLAY-Laplace"},
            {"variant": "slay-tensor", "kwargs": {"num_prf_features": 8, "num_quadrature_nodes": 1, "poly_sketch_dim": 16}, "label": "SLAY-Tensor"},
            {"variant": "slay-rm", "kwargs": {"num_prf_features": 8, "num_quadrature_nodes": 1, "poly_dim": 16}, "label": "SLAY-RM"},
            {"variant": "slay-nystrom", "kwargs": {"num_prf_features": 8, "num_quadrature_nodes": 1, "poly_dim": 16}, "label": "SLAY-Nystrom"},
            {"variant": "slay-anchor", "kwargs": {"num_prf_features": 8, "num_quadrature_nodes": 1, "poly_dim": 16}, "label": "SLAY-Anchor"},
        ]
    
    for test_cfg in configs_to_test:
        variant = test_cfg["variant"]
        kwargs = test_cfg["kwargs"]
        label = test_cfg["label"]
        
        print(f"\n[{label}]")
        
        for seq_len in cfg.seq_lengths:
            print(f"  L={seq_len}...", end=" ", flush=True)
            
            try:
                metrics = benchmark_single_config(cfg, seq_len, variant, kwargs)
                
                result = {
                    "method": label,
                    "variant": variant,
                    "seq_len": seq_len,
                    **kwargs,
                    **metrics,
                }
                results.append(result)
                
                print(f"rel_L2={metrics['rel_l2']:.4f}, cos_sim={metrics['cos_sim']:.4f}, {metrics['latency_ms']:.2f}ms")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "method": label,
                    "variant": variant,
                    "seq_len": seq_len,
                    **kwargs,
                    "error": str(e),
                })
    
    return results


def format_kernel_latex_table(results: List[Dict[str, Any]], seq_len: int = 256) -> str:
    """Format kernel quality results as LaTeX table."""
    
    # Filter to specific sequence length
    filtered = [r for r in results if r.get("seq_len") == seq_len and "error" not in r]
    
    # Sort by rel_l2
    filtered.sort(key=lambda r: r.get("rel_l2", float("inf")))
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Kernel approximation quality at $L={seq_len}$. Lower Rel.~$\\ell_2$ is better.}}")
    lines.append(r"\label{tab:kernel-quality}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Rel.~$\ell_2\downarrow$ & Cos Sim$\uparrow$ & Latency (ms) \\")
    lines.append(r"\midrule")
    
    for r in filtered:
        method = r.get("method", "Unknown")
        rel_l2 = r.get("rel_l2", float("nan"))
        cos_sim = r.get("cos_sim", float("nan"))
        latency = r.get("latency_ms", float("nan"))
        
        rel_str = f"{rel_l2:.4f}" if rel_l2 == rel_l2 else "--"
        cos_str = f"{cos_sim:.4f}" if cos_sim == cos_sim else "--"
        lat_str = f"{latency:.2f}" if latency == latency else "--"
        
        lines.append(f"{method} & {rel_str} & {cos_str} & {lat_str} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="JAX Kernel Approximation Quality Benchmark")
    parser.add_argument("--sweep", action="store_true", help="Full hyperparameter sweep")
    parser.add_argument("--output-dir", default="../artifacts", help="Output directory")
    args = parser.parse_args()
    
    cfg = KernelConfig()
    
    print("=" * 70)
    print("JAX/Flax Kernel Approximation Quality Benchmark")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Embed dim: {cfg.embed_dim}, Heads: {cfg.num_heads}")
    print(f"Sequence lengths: {cfg.seq_lengths}")
    print(f"Mode: {'sweep' if args.sweep else 'default'}")
    print("=" * 70)
    
    # Run benchmarks
    results = run_kernel_benchmark(cfg, sweep=args.sweep)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "kernel_quality.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "sweep": args.sweep,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate LaTeX table
    os.makedirs("../tables", exist_ok=True)
    latex_table = format_kernel_latex_table(results, seq_len=256)
    table_file = os.path.join("../tables", "kernel_quality.tex")
    with open(table_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("KERNEL QUALITY SUMMARY (L=256)")
    print("=" * 70)
    print(latex_table)


if __name__ == "__main__":
    main()
