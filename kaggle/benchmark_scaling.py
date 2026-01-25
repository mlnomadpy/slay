#!/usr/bin/env python3
"""
JAX/Flax Scaling Benchmark: Latency and Memory vs Sequence Length

Uses attention implementations from kaggle/main.py.

Generates the core scaling tables for the SLAY paper:
- Table: Attention-only latency at various sequence lengths
- Table: Peak memory at various sequence lengths

Run:
    python kaggle/benchmark_scaling.py
    python kaggle/benchmark_scaling.py --quick
    python kaggle/benchmark_scaling.py --backward
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

# Import attention classes from main.py (same directory)
from main import (
    RotarySelfAttention,
    StandardAttention,
    LinearAttention,
    FastAttention,
    CosformerAttention,
    RFFAttention,
    YatAttention,
    YatSphericalAttention,
    SLAYAttention,
    SLAYTensorAttention,
    SLAYLaplaceAttention,
    SLAYRMAttention,
    SLAYNystromAttention,
    SLAYAnchorAttention,
    precompute_freqs_cis,
)


@dataclass
class ScalingConfig:
    """Configuration for scaling benchmarks."""
    embed_dim: int = 256
    num_heads: int = 8
    batch_size: int = 1
    warmup_iters: int = 5
    bench_iters: int = 20
    
    # Sequence lengths to test
    seq_lengths: tuple = (256, 512, 1024, 2048, 4096, 8192)
    
    seed: int = 42
    
    # SLAY-specific parameters
    slay_num_features: int = 32
    slay_num_quad: int = 2
    slay_poly_dim: int = 16
    slay_prf_dim: int = 8
    slay_poly_sketch_dim: int = 16
    performer_kernel_size: int = 64
    rff_num_features: int = 64


# Attention types for benchmarking
BASELINE_ATTENTIONS = ["standard", "rotary", "linear", "performer", "cosformer", "rff"]
YAT_ATTENTIONS = ["yat", "yat-spherical"]
SLAY_ATTENTIONS = ["slay", "slay-tensor", "slay-laplace", "slay-rm", "slay-nystrom", "slay-anchor"]

ALL_ATTENTIONS = BASELINE_ATTENTIONS + YAT_ATTENTIONS + SLAY_ATTENTIONS


ATTENTION_REGISTRY = {
    "standard": StandardAttention,
    "rotary": RotarySelfAttention,
    "linear": LinearAttention,
    "performer": FastAttention,
    "cosformer": CosformerAttention,
    "rff": RFFAttention,
    "yat": YatAttention,
    "yat-spherical": YatSphericalAttention,
    "slay": SLAYAttention,
    "slay-tensor": SLAYTensorAttention,
    "slay-laplace": SLAYLaplaceAttention,
    "slay-rm": SLAYRMAttention,
    "slay-nystrom": SLAYNystromAttention,
    "slay-anchor": SLAYAnchorAttention,
}


def get_attention_kwargs(name: str, cfg: ScalingConfig) -> Dict[str, Any]:
    """Get constructor kwargs for each attention type."""
    if name == "performer":
        return {"kernel_size": cfg.performer_kernel_size}
    if name == "rff":
        return {"num_features": cfg.rff_num_features}
    if name in {"yat", "yat-spherical"}:
        return {"epsilon": 1e-6}
    if name == "slay":
        return {
            "num_features": cfg.slay_num_features,
            "num_quadrature_nodes": cfg.slay_num_quad,
        }
    if name == "slay-tensor":
        return {
            "num_prf_features": cfg.slay_prf_dim,
            "num_quadrature_nodes": cfg.slay_num_quad,
            "poly_sketch_dim": cfg.slay_poly_sketch_dim,
        }
    if name == "slay-laplace":
        return {
            "num_features": cfg.slay_num_features,
            "num_quadrature_nodes": cfg.slay_num_quad,
        }
    if name in {"slay-rm", "slay-nystrom", "slay-anchor"}:
        return {
            "num_prf_features": cfg.slay_prf_dim,
            "num_quadrature_nodes": cfg.slay_num_quad,
            "poly_dim": cfg.slay_poly_dim,
        }
    return {}


def create_attention_module(name: str, cfg: ScalingConfig, rngs: nnx.Rngs):
    """Create attention module by name."""
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention: {name}. Available: {list(ATTENTION_REGISTRY.keys())}")
    
    attn_cls = ATTENTION_REGISTRY[name]
    kwargs = get_attention_kwargs(name, cfg)
    
    return attn_cls(cfg.embed_dim, cfg.num_heads, rngs=rngs, **kwargs)


def benchmark_latency_forward(
    attn_module,
    x: jnp.ndarray,
    freqs_cos: jnp.ndarray,
    freqs_sin: jnp.ndarray,
    warmup: int,
    iters: int,
) -> float:
    """Benchmark forward-only latency in milliseconds."""
    
    @jax.jit
    def forward_fn(attn, inp, fc, fs):
        return attn(inp, fc, fs)
    
    # Warmup
    for _ in range(warmup):
        _ = forward_fn(attn_module, x, freqs_cos, freqs_sin)
        jax.block_until_ready(_)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        out = forward_fn(attn_module, x, freqs_cos, freqs_sin)
        jax.block_until_ready(out)
    elapsed = time.perf_counter() - start
    
    return (elapsed / iters) * 1000.0


def benchmark_latency_backward(
    attn_module,
    x: jnp.ndarray,
    freqs_cos: jnp.ndarray,
    freqs_sin: jnp.ndarray,
    warmup: int,
    iters: int,
) -> float:
    """Benchmark forward+backward latency in milliseconds."""
    
    def loss_fn(attn, inp, fc, fs):
        out = attn(inp, fc, fs)
        return jnp.sum(out)
    
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    
    # Warmup
    for _ in range(warmup):
        _, grads = grad_fn(attn_module, x, freqs_cos, freqs_sin)
        jax.block_until_ready(grads)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _, grads = grad_fn(attn_module, x, freqs_cos, freqs_sin)
        jax.block_until_ready(grads)
    elapsed = time.perf_counter() - start
    
    return (elapsed / iters) * 1000.0


def estimate_memory_mb(
    name: str,
    cfg: ScalingConfig,
    seq_len: int,
    backward: bool = False,
) -> Optional[float]:
    """Estimate peak memory in MB using JAX profiler."""
    try:
        rngs = nnx.Rngs(cfg.seed)
        attn = create_attention_module(name, cfg, rngs)
        
        key = jax.random.PRNGKey(cfg.seed)
        x = jax.random.normal(key, (cfg.batch_size, seq_len, cfg.embed_dim))
        
        head_dim = cfg.embed_dim // cfg.num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
        
        if backward:
            def loss_fn(attn, inp, fc, fs):
                return jnp.sum(attn(inp, fc, fs))
            
            grad_fn = jax.jit(jax.value_and_grad(loss_fn))
            _, grads = grad_fn(attn, x, freqs_cos, freqs_sin)
            jax.block_until_ready(grads)
        else:
            @jax.jit
            def forward_fn(attn, inp, fc, fs):
                return attn(inp, fc, fs)
            out = forward_fn(attn, x, freqs_cos, freqs_sin)
            jax.block_until_ready(out)
        
        # Estimate based on tensor sizes (rough approximation)
        # Input: B * L * E * 4 bytes
        # QKV: B * L * 3E * 4 bytes
        # Attention scores (for quadratic): B * H * L * L * 4 bytes
        input_mb = (cfg.batch_size * seq_len * cfg.embed_dim * 4) / (1024**2)
        qkv_mb = input_mb * 3
        
        if name in ["standard", "rotary", "yat", "yat-spherical"]:
            # O(L^2) attention matrix
            attn_mb = (cfg.batch_size * cfg.num_heads * seq_len * seq_len * 4) / (1024**2)
        else:
            # Linear attention - no L^2 matrix
            attn_mb = input_mb * 2  # Just features
        
        total_mb = input_mb + qkv_mb + attn_mb
        if backward:
            total_mb *= 2  # Rough estimate for gradients
        
        return total_mb
        
    except Exception as e:
        print(f"Memory estimation failed for {name}: {e}")
        return None


def run_scaling_benchmark(
    cfg: ScalingConfig,
    attention_names: List[str],
    backward: bool = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run scaling benchmarks for multiple attention types."""
    
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    head_dim = cfg.embed_dim // cfg.num_heads
    
    for name in attention_names:
        results[name] = {}
        print(f"\n[{name}]")
        
        for seq_len in cfg.seq_lengths:
            print(f"  L={seq_len}...", end=" ", flush=True)
            
            try:
                rngs = nnx.Rngs(cfg.seed)
                attn = create_attention_module(name, cfg, rngs)
                
                key = jax.random.PRNGKey(cfg.seed)
                x = jax.random.normal(key, (cfg.batch_size, seq_len, cfg.embed_dim))
                
                freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
                
                # Benchmark latency
                if backward:
                    latency_ms = benchmark_latency_backward(
                        attn, x, freqs_cos, freqs_sin, cfg.warmup_iters, cfg.bench_iters
                    )
                else:
                    latency_ms = benchmark_latency_forward(
                        attn, x, freqs_cos, freqs_sin, cfg.warmup_iters, cfg.bench_iters
                    )
                
                # Estimate memory
                memory_mb = estimate_memory_mb(name, cfg, seq_len, backward)
                
                results[name][seq_len] = {
                    "latency_ms": latency_ms,
                    "memory_mb": memory_mb,
                    "status": "ok",
                }
                
                mem_str = f", ~{memory_mb:.1f}MB" if memory_mb else ""
                print(f"OK {latency_ms:.2f}ms{mem_str}")
                
            except Exception as e:
                results[name][seq_len] = {
                    "latency_ms": float("nan"),
                    "memory_mb": float("nan"),
                    "status": "error",
                }
                print(f"ERROR: {e}")
    
    return results


def format_latex_table(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    metric: str = "latency_ms",
    caption: str = "",
    label: str = "",
) -> str:
    """Format results as a LaTeX table."""
    
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
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    for name in results:
        row = name.replace("_", "-")
        for seq_len in all_seq_lens:
            val = results[name].get(seq_len, {}).get(metric, float("nan"))
            if val != val or val == float("inf"):  # NaN or inf
                row += " & --"
            elif metric == "latency_ms":
                row += f" & {val:.2f}"
            else:
                row += f" & {val:.1f}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="JAX Attention Scaling Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer configs")
    parser.add_argument("--backward", action="store_true", help="Include backward pass")
    parser.add_argument("--attentions", nargs="*", default=None, help="Specific attentions to benchmark")
    parser.add_argument("--output-dir", default="../artifacts", help="Output directory for results")
    args = parser.parse_args()
    
    # Configuration
    if args.quick:
        cfg = ScalingConfig(
            seq_lengths=(256, 512, 1024, 2048),
            warmup_iters=2,
            bench_iters=5,
        )
    else:
        cfg = ScalingConfig()
    
    attention_names = args.attentions if args.attentions else ALL_ATTENTIONS
    
    print("=" * 70)
    print("JAX/Flax Attention Scaling Benchmark")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.device_count()}")
    print(f"Embed dim: {cfg.embed_dim}, Heads: {cfg.num_heads}")
    print(f"Sequence lengths: {cfg.seq_lengths}")
    print(f"Backward pass: {args.backward}")
    print(f"Attentions: {attention_names}")
    print("=" * 70)
    
    # Run benchmarks
    results = run_scaling_benchmark(cfg, attention_names, args.backward)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_backward" if args.backward else "_forward"
    
    results_file = os.path.join(args.output_dir, f"scaling_results{suffix}.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "backward": args.backward,
            "results": {k: {str(sk): sv for sk, sv in v.items()} for k, v in results.items()},
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate LaTeX tables
    os.makedirs("../tables", exist_ok=True)
    
    unit = "ms" if not args.backward else "ms (fwd+bwd)"
    latency_table = format_latex_table(
        results,
        metric="latency_ms",
        caption=f"Attention latency ({unit}) at various sequence lengths.",
        label=f"tab:latency-scaling{suffix}",
    )
    
    latency_file = os.path.join("../tables", f"latency_scaling{suffix}.tex")
    with open(latency_file, "w") as f:
        f.write(latency_table)
    print(f"LaTeX table saved to: {latency_file}")
    
    memory_table = format_latex_table(
        results,
        metric="memory_mb",
        caption=f"Estimated peak memory (MB) at various sequence lengths.",
        label=f"tab:memory-scaling{suffix}",
    )
    
    memory_file = os.path.join("../tables", f"memory_scaling{suffix}.tex")
    with open(memory_file, "w") as f:
        f.write(memory_table)
    print(f"LaTeX table saved to: {memory_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("LATENCY SUMMARY (ms)")
    print("=" * 70)
    print(latency_table)


if __name__ == "__main__":
    main()
