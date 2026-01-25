#!/usr/bin/env python3
"""
JAX/Flax Feature Budget Trade-off Benchmark

Understand accuracy vs. speed/memory trade-off for SLAY variants.

Variables:
- R (quadrature nodes): 1, 2, 4, 8
- M (PRF features): 8, 16, 32, 64
- P (polynomial dim): 8, 16, 32, 64

Metrics:
- Kernel approximation error
- Latency (ms)
- Peak memory (MB estimate)

Run:
    python kaggle/benchmark_feature_budget.py
    python kaggle/benchmark_feature_budget.py --full
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
    SLAYLaplaceAttention,
    SLAYAnchorAttention,
    precompute_freqs_cis,
    mesh,
)


@dataclass
class FeatureBudgetConfig:
    """Configuration for feature budget benchmarks."""
    embed_dim: int = 64
    num_heads: int = 4
    batch_size: int = 4
    seq_len: int = 256
    
    seed: int = 42
    warmup_iters: int = 3
    bench_iters: int = 10
    
    # Sweep parameters
    quad_nodes: tuple = (1, 2, 4)
    prf_features: tuple = (8, 16, 32, 64)
    poly_dims: tuple = (8, 16, 32)


def compute_metrics(y_approx: jnp.ndarray, y_exact: jnp.ndarray) -> Dict[str, float]:
    """Compute approximation quality metrics."""
    y_approx_flat = y_approx.reshape(-1)
    y_exact_flat = y_exact.reshape(-1)
    
    l2_diff = jnp.linalg.norm(y_approx_flat - y_exact_flat)
    l2_exact = jnp.linalg.norm(y_exact_flat)
    rel_l2 = float(l2_diff / (l2_exact + 1e-10))
    
    return {"rel_l2": rel_l2}


def benchmark_config(
    cfg: FeatureBudgetConfig,
    variant: str,
    kwargs: Dict[str, Any],
    x: jnp.ndarray,
    y_exact: jnp.ndarray,
    freqs_cos: jnp.ndarray,
    freqs_sin: jnp.ndarray,
) -> Dict[str, float]:
    """Benchmark a single configuration."""
    
    with mesh:
        rngs = nnx.Rngs(cfg.seed)
    
        if variant == "slay":
            attn = SLAYAttention(cfg.embed_dim, cfg.num_heads, rngs=rngs, **kwargs)
        elif variant == "slay-laplace":
            attn = SLAYLaplaceAttention(cfg.embed_dim, cfg.num_heads, rngs=rngs, **kwargs)
        elif variant == "slay-anchor":
            attn = SLAYAnchorAttention(cfg.embed_dim, cfg.num_heads, rngs=rngs, **kwargs)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    
        @jax.jit
        def forward(attn, inp, fc, fs):
            return attn(inp, fc, fs)
    
        # Warmup
        for _ in range(cfg.warmup_iters):
            out = forward(attn, x, freqs_cos, freqs_sin)
            jax.block_until_ready(out)
    
        # Benchmark
        start = time.perf_counter()
        for _ in range(cfg.bench_iters):
            out = forward(attn, x, freqs_cos, freqs_sin)
            jax.block_until_ready(out)
        latency_ms = (time.perf_counter() - start) / cfg.bench_iters * 1000
    
        # Compute error
        metrics = compute_metrics(out, y_exact)
        metrics["latency_ms"] = latency_ms
    
        # Estimate feature dimension (for Pareto analysis)
        num_quad = kwargs.get("num_quadrature_nodes", 1)
        num_feat = kwargs.get("num_features", kwargs.get("num_prf_features", 32))
        poly_dim = kwargs.get("poly_dim", 1)
    
        if variant == "slay-laplace":
            feature_dim = num_quad * num_feat
        else:
            feature_dim = num_quad * num_feat * poly_dim
    
        metrics["feature_dim"] = feature_dim
    
    return metrics


def run_feature_budget_sweep(cfg: FeatureBudgetConfig) -> List[Dict[str, Any]]:
    """Run the feature budget sweep."""
    
    results = []
    
    # Setup
    head_dim = cfg.embed_dim // cfg.num_heads
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, cfg.seq_len)
    
    key = jax.random.PRNGKey(cfg.seed)
    x = jax.random.normal(key, (cfg.batch_size, cfg.seq_len, cfg.embed_dim))
    
    # Get exact output
    with mesh:
        exact_rngs = nnx.Rngs(cfg.seed)
        exact_attn = YatSphericalAttention(cfg.embed_dim, cfg.num_heads, rngs=exact_rngs, epsilon=1e-6)
    
        @jax.jit
        def exact_forward(attn, inp, fc, fs):
            return attn(inp, fc, fs)
    
        y_exact = exact_forward(exact_attn, x, freqs_cos, freqs_sin)
        jax.block_until_ready(y_exact)
    
    # Sweep SLAY (Hadamard fusion)
    print("\n[SLAY - Hadamard Fusion]")
    for num_quad in cfg.quad_nodes:
        for num_feat in cfg.prf_features:
            kwargs = {
                "num_features": num_feat,
                "num_quadrature_nodes": num_quad,
            }
            
            print(f"  Q={num_quad}, F={num_feat}...", end=" ", flush=True)
            
            try:
                metrics = benchmark_config(cfg, "slay", kwargs, x, y_exact, freqs_cos, freqs_sin)
                result = {
                    "variant": "slay",
                    "num_quad": num_quad,
                    "num_features": num_feat,
                    "poly_dim": None,
                    **metrics,
                }
                results.append(result)
                print(f"rel_L2={metrics['rel_l2']:.4f}, {metrics['latency_ms']:.2f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Sweep SLAY-Laplace (no polynomial)
    print("\n[SLAY-Laplace]")
    for num_quad in cfg.quad_nodes:
        for num_feat in cfg.prf_features:
            kwargs = {
                "num_features": num_feat,
                "num_quadrature_nodes": num_quad,
            }
            
            print(f"  Q={num_quad}, F={num_feat}...", end=" ", flush=True)
            
            try:
                metrics = benchmark_config(cfg, "slay-laplace", kwargs, x, y_exact, freqs_cos, freqs_sin)
                result = {
                    "variant": "slay-laplace",
                    "num_quad": num_quad,
                    "num_features": num_feat,
                    "poly_dim": None,
                    **metrics,
                }
                results.append(result)
                print(f"rel_L2={metrics['rel_l2']:.4f}, {metrics['latency_ms']:.2f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Sweep SLAY-Anchor (with polynomial dim)
    print("\n[SLAY-Anchor]")
    for num_quad in cfg.quad_nodes:
        for poly_dim in cfg.poly_dims:
            kwargs = {
                "num_prf_features": 8,  # Fixed PRF features
                "num_quadrature_nodes": num_quad,
                "poly_dim": poly_dim,
            }
            
            print(f"  Q={num_quad}, P={poly_dim}...", end=" ", flush=True)
            
            try:
                metrics = benchmark_config(cfg, "slay-anchor", kwargs, x, y_exact, freqs_cos, freqs_sin)
                result = {
                    "variant": "slay-anchor",
                    "num_quad": num_quad,
                    "num_features": 8,
                    "poly_dim": poly_dim,
                    **metrics,
                }
                results.append(result)
                print(f"rel_L2={metrics['rel_l2']:.4f}, {metrics['latency_ms']:.2f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
    
    return results


def find_pareto_frontier(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find Pareto-optimal configurations (minimize error & latency)."""
    
    pareto = []
    
    for r in results:
        is_dominated = False
        error = r.get("rel_l2", float("inf"))
        latency = r.get("latency_ms", float("inf"))
        
        for other in results:
            other_error = other.get("rel_l2", float("inf"))
            other_latency = other.get("latency_ms", float("inf"))
            
            # Check if 'other' dominates 'r'
            if other_error <= error and other_latency <= latency:
                if other_error < error or other_latency < latency:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto.append(r)
    
    return pareto


def format_feature_budget_table(results: List[Dict[str, Any]]) -> str:
    """Format results as LaTeX table."""
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Feature budget trade-off. $\star$ indicates Pareto-optimal configurations.}")
    lines.append(r"\label{tab:feature-budget}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Config & Features & Rel.~$\ell_2$ & Latency (ms) & Pareto \\")
    lines.append(r"\midrule")
    
    # Sort by variant, then error
    sorted_results = sorted(results, key=lambda r: (r.get("variant", ""), r.get("rel_l2", float("inf"))))
    
    pareto = find_pareto_frontier(results)
    pareto_set = set(id(r) for r in pareto)
    
    current_variant = None
    for r in sorted_results:
        variant = r.get("variant", "")
        
        # Add separator between variants
        if variant != current_variant and current_variant is not None:
            lines.append(r"\midrule")
        current_variant = variant
        
        # Config string
        num_quad = r.get("num_quad", "")
        num_feat = r.get("num_features", "")
        poly_dim = r.get("poly_dim")
        
        if poly_dim:
            config = f"Q={num_quad}, P={poly_dim}"
        else:
            config = f"Q={num_quad}, F={num_feat}"
        
        feat_dim = r.get("feature_dim", "--")
        rel_l2 = r.get("rel_l2", float("nan"))
        latency = r.get("latency_ms", float("nan"))
        
        is_pareto = "$\\star$" if id(r) in pareto_set else ""
        
        rel_str = f"{rel_l2:.4f}" if rel_l2 == rel_l2 else "--"
        lat_str = f"{latency:.2f}" if latency == latency else "--"
        
        lines.append(f"{variant} & {config} & {feat_dim} & {rel_str} & {lat_str} & {is_pareto} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="JAX Feature Budget Trade-off Benchmark")
    parser.add_argument("--full", action="store_true", help="Full sweep with more configurations")
    parser.add_argument("--output-dir", default="../artifacts", help="Output directory")
    args = parser.parse_args()
    
    if args.full:
        cfg = FeatureBudgetConfig(
            quad_nodes=(1, 2, 4, 8),
            prf_features=(8, 16, 32, 64, 128),
            poly_dims=(8, 16, 32, 64),
        )
    else:
        cfg = FeatureBudgetConfig()
    
    print("=" * 70)
    print("JAX/Flax Feature Budget Trade-off Benchmark")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Embed dim: {cfg.embed_dim}, Heads: {cfg.num_heads}")
    print(f"Sequence length: {cfg.seq_len}")
    print(f"Quadrature nodes: {cfg.quad_nodes}")
    print(f"PRF features: {cfg.prf_features}")
    print(f"Polynomial dims: {cfg.poly_dims}")
    print("=" * 70)
    
    # Run sweep
    results = run_feature_budget_sweep(cfg)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "feature_budget.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate LaTeX table
    os.makedirs("../tables", exist_ok=True)
    latex_table = format_feature_budget_table(results)
    table_file = os.path.join("../tables", "feature_budget.tex")
    with open(table_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_file}")
    
    # Find and print Pareto frontier
    pareto = find_pareto_frontier(results)
    
    print("\n" + "=" * 70)
    print("PARETO-OPTIMAL CONFIGURATIONS")
    print("=" * 70)
    for r in sorted(pareto, key=lambda x: x.get("rel_l2", float("inf"))):
        print(f"  {r['variant']}: Q={r.get('num_quad')}, F={r.get('num_features')}, "
              f"P={r.get('poly_dim')} -> rel_L2={r['rel_l2']:.4f}, {r['latency_ms']:.2f}ms")
    
    print("\n" + "=" * 70)
    print("FULL TABLE")
    print("=" * 70)
    print(latex_table)


if __name__ == "__main__":
    main()
