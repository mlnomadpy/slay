#!/usr/bin/env python3
"""
JAX/Flax Maximum Context Length Benchmark

Demonstrates long-context capability under fixed memory constraints.

Protocol:
1. Fix memory budget (estimated)
2. Binary search for maximum sequence length
3. Inference mode (no gradients)

Run:
    python kaggle/benchmark_max_context.py
    python kaggle/benchmark_max_context.py --memory-budget 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from main import (
    StandardAttention,
    LinearAttention,
    FastAttention,
    CosformerAttention,
    YatSphericalAttention,
    SLAYAttention,
    SLAYLaplaceAttention,
    precompute_freqs_cis,
)


@dataclass
class MaxContextConfig:
    """Configuration for max context benchmarks."""
    embed_dim: int = 256
    num_heads: int = 8
    batch_size: int = 1
    
    # Memory budget in GB
    memory_budget_gb: float = 8.0
    
    # Search range
    min_seq_len: int = 256
    max_seq_len: int = 131072  # 128K
    
    seed: int = 42
    
    # SLAY parameters
    slay_num_features: int = 32
    slay_num_quad: int = 2


ATTENTION_REGISTRY = {
    "standard": (StandardAttention, {}),
    "linear": (LinearAttention, {}),
    "performer": (FastAttention, {"kernel_size": 64}),
    "cosformer": (CosformerAttention, {}),
    "yat-spherical": (YatSphericalAttention, {"epsilon": 1e-6}),
    "slay": (SLAYAttention, {"num_features": 32, "num_quadrature_nodes": 2}),
    "slay-laplace": (SLAYLaplaceAttention, {"num_features": 32, "num_quadrature_nodes": 2}),
}


def estimate_memory_gb(
    attention_type: str,
    embed_dim: int,
    num_heads: int,
    batch_size: int,
    seq_len: int,
) -> float:
    """Estimate memory usage in GB for a given configuration."""
    
    bytes_per_float = 4  # float32
    
    # Input tensor
    input_mem = batch_size * seq_len * embed_dim * bytes_per_float
    
    # QKV projections
    qkv_mem = input_mem * 3
    
    # Attention-specific memory
    if attention_type in ["standard", "yat-spherical"]:
        # O(L^2) attention matrix
        attn_matrix_mem = batch_size * num_heads * seq_len * seq_len * bytes_per_float
    else:
        # Linear attention: O(L * D) features
        # Rough estimate: 2x input for features
        attn_matrix_mem = input_mem * 2
    
    # Output
    output_mem = input_mem
    
    # Total (with overhead factor)
    total_bytes = (input_mem + qkv_mem + attn_matrix_mem + output_mem) * 1.5
    
    return total_bytes / (1024 ** 3)


def can_run_at_seq_len(
    cfg: MaxContextConfig,
    attention_type: str,
    seq_len: int,
) -> bool:
    """Check if we can run at a given sequence length within memory budget."""
    
    estimated_mem = estimate_memory_gb(
        attention_type,
        cfg.embed_dim,
        cfg.num_heads,
        cfg.batch_size,
        seq_len,
    )
    
    if estimated_mem > cfg.memory_budget_gb:
        return False
    
    # Try to actually run it
    try:
        rngs = nnx.Rngs(cfg.seed)
        attn_cls, kwargs = ATTENTION_REGISTRY[attention_type]
        attn = attn_cls(cfg.embed_dim, cfg.num_heads, rngs=rngs, **kwargs)
        
        key = jax.random.PRNGKey(cfg.seed)
        x = jax.random.normal(key, (cfg.batch_size, seq_len, cfg.embed_dim))
        
        head_dim = cfg.embed_dim // cfg.num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
        
        @jax.jit
        def forward(attn, inp, fc, fs):
            return attn(inp, fc, fs)
        
        out = forward(attn, x, freqs_cos, freqs_sin)
        jax.block_until_ready(out)
        
        return True
        
    except Exception as e:
        return False


def binary_search_max_context(
    cfg: MaxContextConfig,
    attention_type: str,
) -> int:
    """Binary search for maximum context length."""
    
    lo = cfg.min_seq_len
    hi = cfg.max_seq_len
    max_working = lo
    
    while lo <= hi:
        mid = (lo + hi) // 2
        # Round to power of 2 or nice number
        mid = max(256, (mid // 256) * 256)
        
        if can_run_at_seq_len(cfg, attention_type, mid):
            max_working = mid
            lo = mid + 256
        else:
            hi = mid - 256
    
    return max_working


def run_max_context_benchmark(cfg: MaxContextConfig) -> Dict[str, Any]:
    """Find maximum context length for each attention type."""
    
    results = {}
    
    for name in ATTENTION_REGISTRY:
        print(f"\n[{name}] Searching for max context...", flush=True)
        
        try:
            max_len = binary_search_max_context(cfg, name)
            estimated_mem = estimate_memory_gb(
                name, cfg.embed_dim, cfg.num_heads, cfg.batch_size, max_len
            )
            
            results[name] = {
                "max_context": max_len,
                "estimated_memory_gb": estimated_mem,
                "status": "ok",
            }
            
            print(f"  Max context: {max_len} tokens (~{estimated_mem:.2f} GB)")
            
        except Exception as e:
            results[name] = {
                "max_context": cfg.min_seq_len,
                "error": str(e),
                "status": "error",
            }
            print(f"  ERROR: {e}")
    
    return results


def format_max_context_table(results: Dict[str, Any], memory_budget: float) -> str:
    """Format max context results as LaTeX table."""
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Maximum context length under {memory_budget:.0f} GB memory budget (inference).}}")
    lines.append(r"\label{tab:max-context}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Max Tokens & Est. Memory (GB) \\")
    lines.append(r"\midrule")
    
    # Sort by max context (descending)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("max_context", 0),
        reverse=True
    )
    
    for name, data in sorted_results:
        max_ctx = data.get("max_context", 0)
        est_mem = data.get("estimated_memory_gb", 0)
        
        if max_ctx >= 1024:
            ctx_str = f"{max_ctx // 1024}K"
        else:
            ctx_str = str(max_ctx)
        
        mem_str = f"{est_mem:.2f}" if est_mem else "--"
        
        lines.append(f"{name} & {ctx_str} & {mem_str} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="JAX Maximum Context Length Benchmark")
    parser.add_argument("--memory-budget", type=float, default=8.0, help="Memory budget in GB")
    parser.add_argument("--output-dir", default="../artifacts", help="Output directory")
    args = parser.parse_args()
    
    cfg = MaxContextConfig(memory_budget_gb=args.memory_budget)
    
    print("=" * 70)
    print("JAX/Flax Maximum Context Length Benchmark")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Memory budget: {cfg.memory_budget_gb} GB")
    print(f"Embed dim: {cfg.embed_dim}, Heads: {cfg.num_heads}")
    print(f"Search range: {cfg.min_seq_len} - {cfg.max_seq_len}")
    print("=" * 70)
    
    # Run benchmark
    results = run_max_context_benchmark(cfg)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "max_context.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate LaTeX table
    os.makedirs("../tables", exist_ok=True)
    latex_table = format_max_context_table(results, cfg.memory_budget_gb)
    table_file = os.path.join("../tables", "max_context.tex")
    with open(table_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("MAX CONTEXT SUMMARY")
    print("=" * 70)
    print(latex_table)


if __name__ == "__main__":
    main()
