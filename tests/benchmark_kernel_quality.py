"""
Kernel Approximation Quality Benchmark

Measures how well linearized attention approximates the exact kernel.

Key metrics:
1. Relative L2 error: ||approx - exact|| / ||exact||
2. Cosine similarity: cos(approx, exact)
3. Attention weight correlation: correlation of attention patterns
4. Top-k recall: Do the highest attention positions match?

Run:
    python tests/benchmark_kernel_quality.py --device cuda
    python tests/benchmark_kernel_quality.py --device cuda --sweep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import get_attention_class


@dataclass
class QualityConfig:
    """Configuration for kernel quality benchmarks."""
    embed_dim: int = 64
    n_heads: int = 4
    batch_size: int = 8
    seq_len: int = 256
    
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


class ExactSphericalYatAttention(nn.Module):
    """
    Exact spherical Yat attention with kernel normalization (not softmax).
    
    This matches the normalization used by linearized versions:
        out_i = (Σ_j K(q_i, k_j) v_j) / (Σ_j K(q_i, k_j))
    
    Used as ground truth for approximation quality.
    """
    
    def __init__(self, embed_dim: int, n_heads: int, epsilon: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Upcast for precision
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # Normalize to sphere
        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)
        
        # Compute kernel: K(q,k) = x² / (C - 2x) where x = qn·kn
        x_dot = torch.matmul(qn, kn.transpose(-2, -1))  # (B, H, T, T)
        denom = torch.clamp(self.C - 2.0 * x_dot, min=1e-6)
        K = (x_dot ** 2) / denom
        
        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        K = K.masked_fill(causal.unsqueeze(0).unsqueeze(0), 0.0)
        
        # Kernel-normalized attention (not softmax)
        numer = torch.matmul(K, v)  # (B, H, T, D)
        z = K.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, H, T, 1)
        out = (numer / z).to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (kernel-normalized) attention weights."""
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).float()
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).float()
        
        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)
        
        x_dot = torch.matmul(qn, kn.transpose(-2, -1))
        denom = torch.clamp(self.C - 2.0 * x_dot, min=1e-6)
        K = (x_dot ** 2) / denom
        
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        K = K.masked_fill(causal.unsqueeze(0).unsqueeze(0), 0.0)
        
        # Normalize
        z = K.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return K / z


def get_attention_kwargs(name: str, cfg: QualityConfig) -> Dict[str, Any]:
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
    return {}


def _torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping.get(dtype_str, torch.float32)


def relative_l2(approx: torch.Tensor, exact: torch.Tensor, eps: float = 1e-12) -> float:
    """Relative L2 error: ||approx - exact|| / ||exact||"""
    diff = (approx - exact).norm()
    base = exact.norm().clamp_min(eps)
    return float(diff / base)


def cosine_similarity(approx: torch.Tensor, exact: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between flattened tensors."""
    a = approx.flatten()
    b = exact.flatten()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)


def mse(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Mean squared error."""
    return float(torch.mean((approx - exact) ** 2))


def max_abs_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Maximum absolute error."""
    return float(torch.max(torch.abs(approx - exact)))


def tie_weights(source: nn.Module, target: nn.Module) -> None:
    """Copy QKV and output projection weights from source to target."""
    target.qkv.load_state_dict(source.qkv.state_dict())
    target.out.load_state_dict(source.out.state_dict())


def benchmark_kernel_quality(
    cfg: QualityConfig,
    attention_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark approximation quality against exact kernel attention.
    
    Returns:
        results[attention_name] = {
            "rel_l2": float,
            "cosine": float,
            "mse": float,
            "max_abs": float,
        }
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dtype = _torch_dtype(cfg.dtype)
    
    # Create exact reference
    exact_attn = ExactSphericalYatAttention(
        cfg.embed_dim, cfg.n_heads, epsilon=cfg.yat_epsilon
    ).to(device)
    
    # Create input
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.embed_dim, device=device, dtype=dtype)
    
    # Get exact output
    with torch.no_grad():
        y_exact = exact_attn(x).float()
    
    results = {}
    
    for name in attention_names:
        print(f"  {name}...", end=" ", flush=True)
        
        try:
            attn_cls = get_attention_class(name)
            kwargs = get_attention_kwargs(name, cfg)
            attn = attn_cls(cfg.embed_dim, cfg.n_heads, **kwargs).to(device)
            
            # Tie weights to ensure fair comparison
            tie_weights(exact_attn, attn)
            
            with torch.no_grad():
                y_approx = attn(x).float()
            
            results[name] = {
                "rel_l2": relative_l2(y_approx, y_exact),
                "cosine": cosine_similarity(y_approx, y_exact),
                "mse": mse(y_approx, y_exact),
                "max_abs": max_abs_error(y_approx, y_exact),
            }
            
            print(f"OK rel_l2={results[name]['rel_l2']:.4f}, cos={results[name]['cosine']:.4f}")
            
            del attn
            
        except Exception as e:
            results[name] = {
                "rel_l2": float("nan"),
                "cosine": float("nan"),
                "mse": float("nan"),
                "max_abs": float("nan"),
                "error": str(e),
            }
            print(f"ERROR: {e}")
    
    return results


def run_feature_budget_sweep(
    cfg: QualityConfig,
    attention_name: str = "yat-performer-anchor",
) -> List[Dict[str, Any]]:
    """
    Sweep over feature budgets to understand quality vs. cost trade-off.
    
    Returns list of {R, M, P, rel_l2, cosine, latency_ms}
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dtype = _torch_dtype(cfg.dtype)
    
    # Create exact reference
    exact_attn = ExactSphericalYatAttention(
        cfg.embed_dim, cfg.n_heads, epsilon=cfg.yat_epsilon
    ).to(device)
    
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.embed_dim, device=device, dtype=dtype)
    
    with torch.no_grad():
        y_exact = exact_attn(x).float()
    
    results = []
    
    # Sweep configurations
    Rs = [1, 2, 4]
    Ms = [8, 16, 32]
    Ps = [8, 16, 32, 64]
    
    for R in Rs:
        for M in Ms:
            for P in Ps:
                print(f"  R={R}, M={M}, P={P}...", end=" ", flush=True)
                
                try:
                    if attention_name == "yat-performer-anchor":
                        attn = get_attention_class(attention_name)(
                            cfg.embed_dim, cfg.n_heads,
                            num_prf_features=M,
                            num_quadrature_nodes=R,
                            poly_dim=P,
                            epsilon=cfg.yat_epsilon,
                        ).to(device)
                    else:
                        # For other types, adjust as needed
                        attn = get_attention_class(attention_name)(
                            cfg.embed_dim, cfg.n_heads,
                            num_features=M * P,
                            num_quadrature_nodes=R,
                            epsilon=cfg.yat_epsilon,
                        ).to(device)
                    
                    tie_weights(exact_attn, attn)
                    
                    # Quality
                    with torch.no_grad():
                        y_approx = attn(x).float()
                    
                    rel = relative_l2(y_approx, y_exact)
                    cos = cosine_similarity(y_approx, y_exact)
                    
                    # Latency
                    attn.eval()
                    for _ in range(5):
                        with torch.no_grad():
                            _ = attn(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    for _ in range(20):
                        with torch.no_grad():
                            _ = attn(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latency_ms = (time.perf_counter() - start) / 20 * 1000
                    
                    results.append({
                        "R": R, "M": M, "P": P,
                        "rel_l2": rel,
                        "cosine": cos,
                        "latency_ms": latency_ms,
                        "total_features": R * M * P,
                    })
                    
                    print(f"OK rel_l2={rel:.4f}, latency={latency_ms:.2f}ms")
                    
                    del attn
                    
                except Exception as e:
                    results.append({
                        "R": R, "M": M, "P": P,
                        "rel_l2": float("nan"),
                        "cosine": float("nan"),
                        "latency_ms": float("nan"),
                        "error": str(e),
                    })
                    print(f"ERROR: {e}")
    
    return results


def format_latex_quality_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format quality results as LaTeX table."""
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Kernel approximation quality. Rel.~$\ell_2$ is relative L2 error (lower is better), Cos is cosine similarity (higher is better).}")
    lines.append(r"\label{tab:kernel-quality}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Rel.~$\ell_2\downarrow$ & Cos$\uparrow$ & MSE$\downarrow$\\")
    lines.append(r"\midrule")
    
    # Sort by rel_l2
    sorted_names = sorted(results.keys(), key=lambda n: results[n].get("rel_l2", float("inf")))
    
    for name in sorted_names:
        r = results[name]
        if math.isnan(r.get("rel_l2", float("nan"))):
            lines.append(f"{name.replace('_', '\\_')} & -- & -- & --\\\\")
        else:
            lines.append(f"{name.replace('_', '\\_')} & {r['rel_l2']:.4f} & {r['cosine']:.4f} & {r['mse']:.2e}\\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def format_markdown_quality_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format quality results as Markdown table."""
    
    lines = []
    lines.append("| Method | Rel L2 (v) | Cosine (^) | MSE (v) | Max Abs |")
    lines.append("|---|---:|---:|---:|---:|")
    
    sorted_names = sorted(results.keys(), key=lambda n: results[n].get("rel_l2", float("inf")))
    
    for name in sorted_names:
        r = results[name]
        if math.isnan(r.get("rel_l2", float("nan"))):
            lines.append(f"| {name} | -- | -- | -- | -- |")
        else:
            lines.append(f"| {name} | {r['rel_l2']:.4f} | {r['cosine']:.4f} | {r['mse']:.2e} | {r['max_abs']:.2e} |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Kernel approximation quality benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--sweep", action="store_true", help="Run feature budget sweep")
    parser.add_argument(
        "--attentions",
        nargs="*",
        default=None,
        help="Attention types to benchmark",
    )
    args = parser.parse_args()
    
    cfg = QualityConfig(
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
    )
    
    # Default attention types to benchmark
    default_attentions = [
        "yat-performer-anchor",
        "yat-performer-laplace",
        "yat-performer",
        "yat-performer-tensor",
        "yat-performer-rm",
        "yat-performer-nystrom",
    ]
    
    attention_names = args.attentions if args.attentions else default_attentions
    
    print("=" * 70)
    print("SLAY Kernel Approximation Quality Benchmark")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"Dtype: {cfg.dtype}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Sequence length: {cfg.seq_len}")
    print(f"Embed dim: {cfg.embed_dim}")
    print(f"Heads: {cfg.n_heads}")
    print("=" * 70)
    
    # Main quality benchmark
    print("\n--- Kernel Approximation Quality ---")
    results = benchmark_kernel_quality(cfg, attention_names)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(format_markdown_quality_table(results))
    
    # Save results
    os.makedirs("tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    # LaTeX
    latex = format_latex_quality_table(results)
    with open("tables/kernel_quality.tex", "w") as f:
        f.write(latex)
    print(f"\nWrote tables/kernel_quality.tex")
    
    # JSON
    with open("artifacts/kernel_quality.json", "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2)
    print(f"Wrote artifacts/kernel_quality.json")
    
    # Feature budget sweep
    if args.sweep:
        print("\n" + "=" * 70)
        print("FEATURE BUDGET SWEEP")
        print("=" * 70)
        
        sweep_results = run_feature_budget_sweep(cfg)
        
        # Save sweep results
        with open("artifacts/feature_budget_sweep.json", "w") as f:
            json.dump({"config": asdict(cfg), "sweep": sweep_results}, f, indent=2)
        print(f"Wrote artifacts/feature_budget_sweep.json")
        
        # Print pareto frontier
        print("\n--- Pareto Frontier (best quality for each latency tier) ---")
        print("| R | M | P | Total | Rel L2 | Latency (ms) |")
        print("|---|---|---|---:|---:|---:|")
        
        # Sort by latency and find pareto points
        sorted_sweep = sorted(sweep_results, key=lambda r: r.get("latency_ms", float("inf")))
        best_rel_l2 = float("inf")
        for r in sorted_sweep:
            if r.get("rel_l2", float("inf")) < best_rel_l2:
                best_rel_l2 = r["rel_l2"]
                print(f"| {r['R']} | {r['M']} | {r['P']} | {r['total_features']} | {r['rel_l2']:.4f} | {r['latency_ms']:.2f} |")


if __name__ == "__main__":
    main()
