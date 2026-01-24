"""Polynomial-approximation ablation for Spherical Yat attention.

This script isolates the effect of the polynomial-kernel approximation by:
- Using the exact spherical Yat attention as a reference.
- Tying QKV and output projection weights across all variants.
- Comparing output error (MSE / relative L2 / cosine similarity) and latency.

It evaluates the following implementations (see src.attention registry):
- Exact: YatSphericalCausalAttention
- Approx (poly+PRF): TensorSketch / Random Maclaurin / Nyström / Anchor
- Baselines: Hadamard (yat-performer) and Laplace-only (yat-performer-laplace)

Run (CPU):
  python tests/ablation_poly_approx.py --quick

Run (CUDA):
  python tests/ablation_poly_approx.py --device cuda --dtype float16 --T 1024 --iters 50

Outputs:
- tables/poly_ablation_results.tex  (LaTeX table for main.tex)
- artifacts/poly_ablation.md        (Markdown copy for inspection)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running directly from repo root
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import (
    YatPerformerTensorCausalAttention,
    YatPerformerRMCausalAttention,
    YatPerformerNystromCausalAttention,
    YatPerformerAnchorCausalAttention,
    YatPerformerCausalAttention,
    YatPerformerLaplaceCausalAttention,
)


class ExactSphericalKernelCausalAttention(nn.Module):
    r"""Exact spherical-YAT *kernel-normalized* causal attention.

    This matches the normalization used by the linearized implementations:
        out_i = (\sum_{j\le i} K(q_i,k_j) v_j) / (\sum_{j\le i} K(q_i,k_j)).

    Note: this is O(T^2) and intended only as an ablation reference.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = float(epsilon)
        self.C = 2.0 + self.epsilon

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()

        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)

        x_dot = torch.matmul(qn, kn.transpose(-2, -1))  # (B,H,T,T)
        denom = torch.clamp(self.C - 2.0 * x_dot, min=1e-6)
        K = (x_dot**2) / denom  # nonnegative on sphere

        # Causal masking: future weights are zero (kernel attention, not softmax)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        K = K.masked_fill(causal.unsqueeze(0).unsqueeze(0), 0.0)

        numer = torch.matmul(K, v)  # (B,H,T,D)
        z = K.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (B,H,T,1)
        out = (numer / z).to(input_dtype)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


@dataclass
class AblationConfig:
    embed_dim: int = 64
    n_heads: int = 4
    batch_size: int = 8
    T: int = 256
    epsilon: float = 1e-6

    # Feature budgets (kept identical across poly-approx variants)
    num_quadrature_nodes: int = 2  # R
    num_prf_features: int = 16  # M
    poly_dim: int = 16  # P (or TensorSketch dim)

    # Benchmarking
    warmup: int = 10
    iters: int = 50
    seed: int = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"  # float32 | float16 | bfloat16


@dataclass(frozen=True)
class SweepPoint:
    name: str
    T: int
    R: int
    M: int
    P: int
    warmup: int
    iters: int


def _torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_forward(module: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int, device: str) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
        _sync_if_cuda(device)
        start = time.perf_counter()
        for _ in range(iters):
            _ = module(x)
        _sync_if_cuda(device)
    return (time.perf_counter() - start) / iters * 1000.0


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    denom = (a_f.norm() * b_f.norm()).clamp_min(eps)
    return float((a_f @ b_f) / denom)


def _relative_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    diff = (a - b).norm()
    base = b.norm().clamp_min(eps)
    return float(diff / base)


def _tie_linear_weights(reference: torch.nn.Module, other: torch.nn.Module) -> None:
    # All modules in this repo expose qkv/out with same shapes.
    other.qkv.load_state_dict(reference.qkv.state_dict())
    other.out.load_state_dict(reference.out.state_dict())


def build_modules(cfg: AblationConfig) -> Dict[str, torch.nn.Module]:
    # Reference (exact) attention
    exact = ExactSphericalKernelCausalAttention(
        cfg.embed_dim,
        cfg.n_heads,
        epsilon=cfg.epsilon,
    )

    # Polynomial-approx variants (match R/M/P)
    ts = YatPerformerTensorCausalAttention(
        cfg.embed_dim,
        cfg.n_heads,
        num_prf_features=cfg.num_prf_features,
        num_quadrature_nodes=cfg.num_quadrature_nodes,
        poly_sketch_dim=cfg.poly_dim,
        epsilon=cfg.epsilon,
    )
    rm = YatPerformerRMCausalAttention(
        cfg.embed_dim,
        cfg.n_heads,
        num_prf_features=cfg.num_prf_features,
        num_quadrature_nodes=cfg.num_quadrature_nodes,
        poly_dim=cfg.poly_dim,
        epsilon=cfg.epsilon,
    )
    nys = YatPerformerNystromCausalAttention(
        cfg.embed_dim,
        cfg.n_heads,
        num_prf_features=cfg.num_prf_features,
        num_quadrature_nodes=cfg.num_quadrature_nodes,
        poly_dim=cfg.poly_dim,
        epsilon=cfg.epsilon,
    )
    anc = YatPerformerAnchorCausalAttention(
        cfg.embed_dim,
        cfg.n_heads,
        num_prf_features=cfg.num_prf_features,
        num_quadrature_nodes=cfg.num_quadrature_nodes,
        poly_dim=cfg.poly_dim,
        epsilon=cfg.epsilon,
    )

    # Baselines
    hadamard = YatPerformerCausalAttention(
        cfg.embed_dim,
        cfg.n_heads,
        num_features=max(1, (cfg.num_prf_features * cfg.poly_dim)),
        num_quadrature_nodes=cfg.num_quadrature_nodes,
        epsilon=cfg.epsilon,
    )
    laplace_only = YatPerformerLaplaceCausalAttention(
        cfg.embed_dim,
        cfg.n_heads,
        num_features=max(1, (cfg.num_prf_features * cfg.poly_dim)),
        num_quadrature_nodes=cfg.num_quadrature_nodes,
        epsilon=cfg.epsilon,
    )

    modules: Dict[str, torch.nn.Module] = {
        "Exact (Spherical)": exact,
        "TensorSketch": ts,
        "Random Maclaurin": rm,
        "Nyström": nys,
        "Anchor": anc,
        "Hadamard (shared ω)": hadamard,
        "Laplace-only": laplace_only,
    }

    # Tie QKV/out weights to isolate kernel approximation differences
    for name, m in modules.items():
        if name == "Exact (Spherical)":
            continue
        _tie_linear_weights(exact, m)

    return modules


def compute_rows(cfg: AblationConfig) -> List[Dict[str, object]]:
    """Compute per-method metrics for a single configuration.

    Returns rows sorted by (rel_l2, latency).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    dtype = _torch_dtype(cfg.dtype)

    modules = build_modules(cfg)
    for m in modules.values():
        m.to(device)

    x = torch.randn(cfg.batch_size, cfg.T, cfg.embed_dim, device=device, dtype=dtype)

    with torch.no_grad():
        y_ref = modules["Exact (Spherical)"](x).float()

    rows: List[Dict[str, object]] = []

    for name, m in modules.items():
        # Runtime: exact reference is O(T^2), so time only one pass.
        if name == "Exact (Spherical)":
            with torch.no_grad():
                _sync_if_cuda(device)
                start = time.perf_counter()
                _ = m(x)
                _sync_if_cuda(device)
            ms = (time.perf_counter() - start) * 1000.0
        else:
            ms = _bench_forward(m, x, cfg.warmup, cfg.iters, device)

        with torch.no_grad():
            y = m(x).float()

        mse = float(torch.mean((y - y_ref) ** 2).item())
        rel = _relative_l2(y, y_ref)
        cos = _cosine_similarity(y, y_ref)

        peak_mb: Optional[float] = None
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = m(x)
            _sync_if_cuda(device)
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

        rows.append(
            {
                "method": name,
                "ms": ms,
                "mse": mse,
                "rel_l2": rel,
                "cos": cos,
                "peak_mb": peak_mb,
            }
        )

    return sorted(rows, key=lambda r: (r["rel_l2"], r["ms"]))


def run_ablation(cfg: AblationConfig) -> Tuple[str, str]:
    rows_sorted = compute_rows(cfg)

    # Markdown table
    md_lines = []
    md_lines.append("# Polynomial Approximation Ablation (Spherical Yat)\n")
    md_lines.append(
        f"Config: B={cfg.batch_size}, T={cfg.T}, C={cfg.embed_dim}, H={cfg.n_heads}, ε={cfg.epsilon}, "
        f"R={cfg.num_quadrature_nodes}, M={cfg.num_prf_features}, P={cfg.poly_dim}, device={cfg.device}, dtype={cfg.dtype}\n"
    )
    md_lines.append("| Method | Rel L2 ↓ | MSE ↓ | Cos ↑ | Latency (ms) ↓ | Peak MB (CUDA) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows_sorted:
        peak = "-" if r["peak_mb"] is None else f"{r['peak_mb']:.1f}"
        md_lines.append(
            f"| {r['method']} | {r['rel_l2']:.4f} | {r['mse']:.3e} | {r['cos']:.4f} | {r['ms']:.2f} | {peak} |"
        )
    md = "\n".join(md_lines) + "\n"

    # LaTeX table (double column) – keep it self-contained for \input{}
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(
        r"\caption{Ablation of polynomial-kernel approximations for the spherical YAT kernel. "
        r"We compare attention outputs against \emph{exact kernel-normalized} spherical YAT with tied QKV/out projections. "
        r"Lower Rel.~$\ell_2$ / MSE is better; higher cosine similarity is better.}")
    latex_lines.append(r"\label{tab:poly-ablation}")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Method & Rel.~$\ell_2\downarrow$ & MSE$\downarrow$ & Cos$\uparrow$ & Latency (ms)$\downarrow$\\")
    latex_lines.append(r"\midrule")

    for r in rows_sorted:
        method = str(r["method"]).replace("&", r"\&")
        latex_lines.append(
            f"{method} & {r['rel_l2']:.4f} & {r['mse']:.2e} & {r['cos']:.4f} & {r['ms']:.2f}\\\\"
        )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")
    latex = "\n".join(latex_lines) + "\n"

    return md, latex


def _format_sweep_table(rows: List[Dict[str, object]]) -> str:
    """Grouped LaTeX table for multiple sweep points.

    Expected keys: scale, T, R, M, P, method, rel_l2, ms
    """
    scale_order: List[str] = []
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        s = str(r["scale"])
        if s not in grouped:
            grouped[s] = []
            scale_order.append(s)
        grouped[s].append(r)

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Multi-scale ablation over feature budgets for polynomial-kernel approximations. "
        r"We compare attention outputs against \emph{exact kernel-normalized} spherical YAT with tied QKV/out projections. "
        r"Lower Rel.~$\ell_2$ is better; latency is forward-pass time.}")
    lines.append(r"\label{tab:poly-sweep}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Scale & Method & $T$ & $R$ & $M$ & $P$ & Rel.~$\ell_2\downarrow$ & Latency (ms)$\downarrow$\\")
    lines.append(r"\midrule")

    for s in scale_order:
        rs = grouped[s]
        # Sort methods by quality then speed
        rs_sorted = sorted(rs, key=lambda r: (float(r["rel_l2"]), float(r["ms"])))
        T = int(rs_sorted[0]["T"])
        R = int(rs_sorted[0]["R"])
        M = int(rs_sorted[0]["M"])
        P = int(rs_sorted[0]["P"])

        first = True
        for r in rs_sorted:
            method = str(r["method"]).replace("&", r"\&")
            rel = float(r["rel_l2"])
            ms = float(r["ms"])
            if first:
                lines.append(f"{s} & {method} & {T} & {R} & {M} & {P} & {rel:.4f} & {ms:.2f}\\\\")
                first = False
            else:
                lines.append(f" & {method} &  &  &  &  & {rel:.4f} & {ms:.2f}\\\\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def run_sweep(base_cfg: AblationConfig, points: List[SweepPoint]) -> Tuple[str, str]:
    all_rows: List[Dict[str, object]] = []
    md_lines: List[str] = []

    md_lines.append("# Polynomial Approximation Sweep (Spherical Yat)\n")
    md_lines.append(
        f"Base: device={base_cfg.device}, dtype={base_cfg.dtype}, embed_dim={base_cfg.embed_dim}, heads={base_cfg.n_heads}, "
        f"batch_size={base_cfg.batch_size}, eps={base_cfg.epsilon}\n"
    )

    for p in points:
        cfg = AblationConfig(**vars(base_cfg))
        cfg.T = p.T
        cfg.num_quadrature_nodes = p.R
        cfg.num_prf_features = p.M
        cfg.poly_dim = p.P
        cfg.warmup = p.warmup
        cfg.iters = p.iters

        rows_sorted = compute_rows(cfg)
        md_lines.append(f"## {p.name} (T={p.T}, R={p.R}, M={p.M}, P={p.P})\n")
        md_lines.append("| Method | Rel L2 ↓ | Latency (ms) ↓ | Cos ↑ |")
        md_lines.append("|---|---:|---:|---:|")
        for r in rows_sorted:
            md_lines.append(f"| {r['method']} | {float(r['rel_l2']):.4f} | {float(r['ms']):.2f} | {float(r['cos']):.4f} |")
        md_lines.append("")

        for r in rows_sorted:
            all_rows.append(
                {
                    "scale": p.name,
                    "T": cfg.T,
                    "R": cfg.num_quadrature_nodes,
                    "M": cfg.num_prf_features,
                    "P": cfg.poly_dim,
                    "method": r["method"],
                    "rel_l2": r["rel_l2"],
                    "ms": r["ms"],
                }
            )

    latex = _format_sweep_table(all_rows)
    return "\n".join(md_lines) + "\n", latex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 | ...")
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)

    parser.add_argument("--R", type=int, default=None, help="Quadrature nodes")
    parser.add_argument("--M", type=int, default=None, help="PRF feature count")
    parser.add_argument("--P", type=int, default=None, help="Polynomial feature dim/sketch dim")

    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--quick", action="store_true", help="Small/fast config")
    parser.add_argument("--sweep", action="store_true", help="Run a multi-scale sweep and write tables/poly_ablation_sweep.tex")

    args = parser.parse_args()

    cfg = AblationConfig()
    if args.quick:
        cfg.batch_size = 4
        cfg.T = 128
        cfg.warmup = 5
        cfg.iters = 20
        cfg.num_quadrature_nodes = 2
        cfg.num_prf_features = 8
        cfg.poly_dim = 8

    if args.device is not None:
        cfg.device = args.device
    if args.dtype is not None:
        cfg.dtype = args.dtype
    if args.B is not None:
        cfg.batch_size = args.B
    if args.T is not None:
        cfg.T = args.T
    if args.embed_dim is not None:
        cfg.embed_dim = args.embed_dim
    if args.heads is not None:
        cfg.n_heads = args.heads
    if args.epsilon is not None:
        cfg.epsilon = args.epsilon

    if args.R is not None:
        cfg.num_quadrature_nodes = args.R
    if args.M is not None:
        cfg.num_prf_features = args.M
    if args.P is not None:
        cfg.poly_dim = args.P

    if args.warmup is not None:
        cfg.warmup = args.warmup
    if args.iters is not None:
        cfg.iters = args.iters
    if args.seed is not None:
        cfg.seed = args.seed

    if args.sweep:
        points = [
            SweepPoint(name="Small", T=128, R=2, M=8, P=8, warmup=5, iters=50),
            SweepPoint(name="Medium", T=256, R=2, M=16, P=16, warmup=5, iters=30),
            SweepPoint(name="Large", T=512, R=2, M=32, P=32, warmup=3, iters=15),
        ]
        md, latex = run_sweep(cfg, points)
    else:
        md, latex = run_ablation(cfg)

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    if args.sweep:
        md_path = os.path.join("artifacts", "poly_ablation_sweep.md")
        tex_path = os.path.join("tables", "poly_ablation_sweep.tex")
    else:
        md_path = os.path.join("artifacts", "poly_ablation.md")
        tex_path = os.path.join("tables", "poly_ablation_results.tex")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(md)
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
