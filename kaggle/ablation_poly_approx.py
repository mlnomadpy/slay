#!/usr/bin/env python3
"""
JAX/Flax Polynomial Approximation Ablation Study

Compares different polynomial approximation methods for the SLAY kernel:
- Anchor features (default)
- Random Maclaurin
- Nyström
- TensorSketch
- Laplace-only (no polynomial)

Run:
    python kaggle/ablation_poly_approx.py
    python kaggle/ablation_poly_approx.py --sweep
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
class AblationConfig:
    """Configuration for polynomial ablation study."""
    embed_dim: int = 64
    num_heads: int = 4
    batch_size: int = 8
    
    seed: int = 42
    warmup_iters: int = 3
    bench_iters: int = 10
    
    # Scales to test
    scales: tuple = ("Small", "Medium", "Large")
    
    # Scale configurations: (seq_len, num_features/poly_dim)
    scale_configs: dict = None
    
    def __post_init__(self):
        if self.scale_configs is None:
            self.scale_configs = {
                "Small": {"seq_len": 64, "num_features": 16, "poly_dim": 8, "num_quad": 1},
                "Medium": {"seq_len": 128, "num_features": 32, "poly_dim": 16, "num_quad": 2},
                "Large": {"seq_len": 256, "num_features": 64, "poly_dim": 32, "num_quad": 2},
            }


# Method configurations
METHODS = {
    "Exact (Yat-Spherical)": {
        "class": YatSphericalAttention,
        "kwargs": lambda cfg: {"epsilon": 1e-6},
    },
    "SLAY (Hadamard)": {
        "class": SLAYAttention,
        "kwargs": lambda cfg: {
            "num_features": cfg["num_features"],
            "num_quadrature_nodes": cfg["num_quad"],
        },
    },
    "SLAY-Laplace": {
        "class": SLAYLaplaceAttention,
        "kwargs": lambda cfg: {
            "num_features": cfg["num_features"],
            "num_quadrature_nodes": cfg["num_quad"],
        },
    },
    "SLAY-Anchor": {
        "class": SLAYAnchorAttention,
        "kwargs": lambda cfg: {
            "num_prf_features": 8,
            "num_quadrature_nodes": cfg["num_quad"],
            "poly_dim": cfg["poly_dim"],
        },
    },
    "SLAY-RM": {
        "class": SLAYRMAttention,
        "kwargs": lambda cfg: {
            "num_prf_features": 8,
            "num_quadrature_nodes": cfg["num_quad"],
            "poly_dim": cfg["poly_dim"],
        },
    },
    "SLAY-Nystrom": {
        "class": SLAYNystromAttention,
        "kwargs": lambda cfg: {
            "num_prf_features": 8,
            "num_quadrature_nodes": cfg["num_quad"],
            "poly_dim": cfg["poly_dim"],
        },
    },
    "SLAY-Tensor": {
        "class": SLAYTensorAttention,
        "kwargs": lambda cfg: {
            "num_prf_features": 8,
            "num_quadrature_nodes": cfg["num_quad"],
            "poly_sketch_dim": cfg["poly_dim"],
        },
    },
}


def compute_metrics(y_approx: jnp.ndarray, y_exact: jnp.ndarray) -> Dict[str, float]:
    """Compute approximation quality metrics."""
    y_approx_flat = y_approx.reshape(-1)
    y_exact_flat = y_exact.reshape(-1)
    
    l2_diff = jnp.linalg.norm(y_approx_flat - y_exact_flat)
    l2_exact = jnp.linalg.norm(y_exact_flat)
    rel_l2 = float(l2_diff / (l2_exact + 1e-10))
    
    dot_prod = jnp.dot(y_approx_flat, y_exact_flat)
    norm_prod = jnp.linalg.norm(y_approx_flat) * jnp.linalg.norm(y_exact_flat)
    cos_sim = float(dot_prod / (norm_prod + 1e-10))
    
    mse = float(jnp.mean((y_approx_flat - y_exact_flat) ** 2))
    
    return {"rel_l2": rel_l2, "cos_sim": cos_sim, "mse": mse}


def benchmark_method(
    method_name: str,
    method_cfg: dict,
    scale_cfg: dict,
    ablation_cfg: AblationConfig,
) -> Dict[str, Any]:
    """Benchmark a single method at a given scale."""
    
    rngs = nnx.Rngs(ablation_cfg.seed)
    key = jax.random.PRNGKey(ablation_cfg.seed)
    
    seq_len = scale_cfg["seq_len"]
    head_dim = ablation_cfg.embed_dim // ablation_cfg.num_heads
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
    
    x = jax.random.normal(key, (ablation_cfg.batch_size, seq_len, ablation_cfg.embed_dim))
    
    # Create attention module
    attn_cls = method_cfg["class"]
    kwargs = method_cfg["kwargs"](scale_cfg)
    attn = attn_cls(ablation_cfg.embed_dim, ablation_cfg.num_heads, rngs=rngs, **kwargs)
    
    @jax.jit
    def forward(attn, inp, fc, fs):
        return attn(inp, fc, fs)
    
    # Warmup
    for _ in range(ablation_cfg.warmup_iters):
        out = forward(attn, x, freqs_cos, freqs_sin)
        jax.block_until_ready(out)
    
    # Benchmark latency
    start = time.perf_counter()
    for _ in range(ablation_cfg.bench_iters):
        out = forward(attn, x, freqs_cos, freqs_sin)
        jax.block_until_ready(out)
    latency_ms = (time.perf_counter() - start) / ablation_cfg.bench_iters * 1000
    
    return out, latency_ms


def run_ablation(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Run the full ablation study."""
    
    results = []
    
    for scale in cfg.scales:
        scale_cfg = cfg.scale_configs[scale]
        print(f"\n[Scale: {scale}] seq_len={scale_cfg['seq_len']}")
        
        # Get exact output for comparison
        exact_rngs = nnx.Rngs(cfg.seed)
        key = jax.random.PRNGKey(cfg.seed)
        
        seq_len = scale_cfg["seq_len"]
        head_dim = cfg.embed_dim // cfg.num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
        x = jax.random.normal(key, (cfg.batch_size, seq_len, cfg.embed_dim))
        
        exact_attn = YatSphericalAttention(cfg.embed_dim, cfg.num_heads, rngs=exact_rngs, epsilon=1e-6)
        
        @jax.jit
        def exact_forward(attn, inp, fc, fs):
            return attn(inp, fc, fs)
        
        y_exact = exact_forward(exact_attn, x, freqs_cos, freqs_sin)
        jax.block_until_ready(y_exact)
        
        # Benchmark each method
        for method_name, method_cfg in METHODS.items():
            print(f"  {method_name}...", end=" ", flush=True)
            
            try:
                y_out, latency_ms = benchmark_method(method_name, method_cfg, scale_cfg, cfg)
                
                if method_name == "Exact (Yat-Spherical)":
                    metrics = {"rel_l2": 0.0, "cos_sim": 1.0, "mse": 0.0}
                else:
                    metrics = compute_metrics(y_out, y_exact)
                
                result = {
                    "method": method_name,
                    "scale": scale,
                    "seq_len": scale_cfg["seq_len"],
                    "ms": latency_ms,
                    **metrics,
                }
                results.append(result)
                
                print(f"rel_L2={metrics['rel_l2']:.4f}, {latency_ms:.2f}ms")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "method": method_name,
                    "scale": scale,
                    "error": str(e),
                })
    
    return results


def format_ablation_table(results: List[Dict[str, Any]], scale: str = "Large") -> str:
    """Format ablation results as LaTeX table."""
    
    filtered = [r for r in results if r.get("scale") == scale and "error" not in r]
    filtered.sort(key=lambda r: r.get("rel_l2", float("inf")))
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Polynomial approximation ablation ({scale} scale). Lower Rel.~$\\ell_2$ is better.}}")
    lines.append(r"\label{tab:poly-ablation}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Rel.~$\ell_2\downarrow$ & Cos Sim$\uparrow$ & Latency (ms) \\")
    lines.append(r"\midrule")
    
    for r in filtered:
        method = r.get("method", "Unknown")
        rel_l2 = r.get("rel_l2", float("nan"))
        cos_sim = r.get("cos_sim", float("nan"))
        latency = r.get("ms", float("nan"))
        
        # Highlight best approximation
        if rel_l2 == 0.0:
            rel_str = "0 (exact)"
        elif rel_l2 == rel_l2:
            rel_str = f"{rel_l2:.4f}"
        else:
            rel_str = "--"
        
        cos_str = f"{cos_sim:.4f}" if cos_sim == cos_sim else "--"
        lat_str = f"{latency:.2f}" if latency == latency else "--"
        
        lines.append(f"{method} & {rel_str} & {cos_str} & {lat_str} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def run_sweep(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Run a sweep over hyperparameters."""
    
    results = []
    
    # Sweep over feature counts
    feature_counts = [8, 16, 32, 64]
    quad_nodes = [1, 2, 4]
    
    seq_len = 256
    head_dim = cfg.embed_dim // cfg.num_heads
    
    for num_feat in feature_counts:
        for num_quad in quad_nodes:
            print(f"\n[Sweep: F={num_feat}, Q={num_quad}]")
            
            # Get exact output
            rngs = nnx.Rngs(cfg.seed)
            key = jax.random.PRNGKey(cfg.seed)
            freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
            x = jax.random.normal(key, (cfg.batch_size, seq_len, cfg.embed_dim))
            
            exact_attn = YatSphericalAttention(cfg.embed_dim, cfg.num_heads, rngs=rngs, epsilon=1e-6)
            
            @jax.jit
            def forward(attn, inp, fc, fs):
                return attn(inp, fc, fs)
            
            y_exact = forward(exact_attn, x, freqs_cos, freqs_sin)
            jax.block_until_ready(y_exact)
            
            # Test SLAY
            try:
                slay_rngs = nnx.Rngs(cfg.seed)
                slay = SLAYAttention(
                    cfg.embed_dim, cfg.num_heads, rngs=slay_rngs,
                    num_features=num_feat, num_quadrature_nodes=num_quad
                )
                y_slay = forward(slay, x, freqs_cos, freqs_sin)
                jax.block_until_ready(y_slay)
                
                metrics = compute_metrics(y_slay, y_exact)
                results.append({
                    "method": "SLAY",
                    "num_features": num_feat,
                    "num_quad": num_quad,
                    "seq_len": seq_len,
                    **metrics,
                })
                print(f"  SLAY: rel_L2={metrics['rel_l2']:.4f}")
            except Exception as e:
                print(f"  SLAY ERROR: {e}")
            
            # Test SLAY-Laplace
            try:
                laplace_rngs = nnx.Rngs(cfg.seed)
                laplace = SLAYLaplaceAttention(
                    cfg.embed_dim, cfg.num_heads, rngs=laplace_rngs,
                    num_features=num_feat, num_quadrature_nodes=num_quad
                )
                y_laplace = forward(laplace, x, freqs_cos, freqs_sin)
                jax.block_until_ready(y_laplace)
                
                metrics = compute_metrics(y_laplace, y_exact)
                results.append({
                    "method": "SLAY-Laplace",
                    "num_features": num_feat,
                    "num_quad": num_quad,
                    "seq_len": seq_len,
                    **metrics,
                })
                print(f"  SLAY-Laplace: rel_L2={metrics['rel_l2']:.4f}")
            except Exception as e:
                print(f"  SLAY-Laplace ERROR: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="JAX Polynomial Approximation Ablation")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--output-dir", default="../artifacts", help="Output directory")
    args = parser.parse_args()
    
    cfg = AblationConfig()
    
    print("=" * 70)
    print("JAX/Flax Polynomial Approximation Ablation")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Embed dim: {cfg.embed_dim}, Heads: {cfg.num_heads}")
    print(f"Mode: {'sweep' if args.sweep else 'ablation'}")
    print("=" * 70)
    
    if args.sweep:
        results = run_sweep(cfg)
        output_name = "poly_ablation_sweep"
    else:
        results = run_ablation(cfg)
        output_name = "poly_ablation"
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f"{output_name}.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate LaTeX table
    os.makedirs("../tables", exist_ok=True)
    if not args.sweep:
        latex_table = format_ablation_table(results, scale="Large")
        table_file = os.path.join("../tables", f"{output_name}.tex")
        with open(table_file, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {table_file}")
        
        print("\n" + "=" * 70)
        print("ABLATION SUMMARY (Large Scale)")
        print("=" * 70)
        print(latex_table)


if __name__ == "__main__":
    main()
