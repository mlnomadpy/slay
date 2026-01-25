#!/usr/bin/env python3
"""
Generate Paper-Ready LaTeX Tables from Benchmark Results

Consolidates benchmark results from JSON artifacts into LaTeX tables.

Usage:
    python kaggle/generate_paper_tables.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def format_number(val: float, fmt: str = ".2f") -> str:
    """Format a number, handling NaN and infinity."""
    if val != val or math.isinf(val):  # NaN or inf
        return "--"
    return f"{val:{fmt}}"


def generate_scaling_latency_table(data: Dict[str, Any], backward: bool = False) -> str:
    """Generate latency scaling table."""
    if not data:
        return "% No scaling data found\n"
    
    results = data.get("results", {})
    
    # Get all sequence lengths
    all_seq_lens = set()
    for attn_results in results.values():
        all_seq_lens.update(int(k) for k in attn_results.keys())
    all_seq_lens = sorted(all_seq_lens)
    
    suffix = " (fwd+bwd)" if backward else " (fwd)"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Attention latency (ms){suffix} at various sequence lengths.}}")
    lines.append(f"\\label{{tab:latency-scaling{'-backward' if backward else ''}}}")
    
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
    
    # Sort methods: baselines first, then SLAY variants
    method_order = ["standard", "rotary", "linear", "performer", "cosformer", "rff",
                   "yat", "yat-spherical", "slay", "slay-laplace", "slay-tensor",
                   "slay-rm", "slay-nystrom", "slay-anchor"]
    
    sorted_methods = sorted(results.keys(), key=lambda x: method_order.index(x) if x in method_order else 100)
    
    for method in sorted_methods:
        method_results = results[method]
        row = method.replace("_", "-")
        
        for seq_len in all_seq_lens:
            val = method_results.get(str(seq_len), {}).get("latency_ms", float("nan"))
            row += f" & {format_number(val, '.2f')}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_scaling_memory_table(data: Dict[str, Any], backward: bool = False) -> str:
    """Generate memory scaling table."""
    if not data:
        return "% No scaling data found\n"
    
    results = data.get("results", {})
    
    all_seq_lens = set()
    for attn_results in results.values():
        all_seq_lens.update(int(k) for k in attn_results.keys())
    all_seq_lens = sorted(all_seq_lens)
    
    suffix = " (fwd+bwd)" if backward else " (fwd)"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Peak memory (MB){suffix} at various sequence lengths.}}")
    lines.append(f"\\label{{tab:memory-scaling{'-backward' if backward else ''}}}")
    
    col_spec = "l" + "r" * len(all_seq_lens)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    header = "Method"
    for seq_len in all_seq_lens:
        if seq_len >= 1024:
            header += f" & {seq_len // 1024}K"
        else:
            header += f" & {seq_len}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    method_order = ["standard", "rotary", "linear", "performer", "cosformer", "rff",
                   "yat", "yat-spherical", "slay", "slay-laplace", "slay-tensor",
                   "slay-rm", "slay-nystrom", "slay-anchor"]
    
    sorted_methods = sorted(results.keys(), key=lambda x: method_order.index(x) if x in method_order else 100)
    
    for method in sorted_methods:
        method_results = results[method]
        row = method.replace("_", "-")
        
        for seq_len in all_seq_lens:
            val = method_results.get(str(seq_len), {}).get("memory_mb", float("nan"))
            if val and val == val:
                row += f" & {val:.1f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_poly_ablation_table(data: Dict[str, Any]) -> str:
    """Generate polynomial ablation table."""
    if not data:
        return "% No ablation data found\n"
    
    results = data.get("results", [])
    
    # Filter to Large scale
    large_results = [r for r in results if r.get("scale") == "Large" and "error" not in r]
    large_results.sort(key=lambda r: r.get("rel_l2", float("inf")))
    
    if not large_results:
        return "% No Large scale results found\n"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Polynomial approximation ablation (Large scale). Lower Rel.~$\ell_2$ is better.}")
    lines.append(r"\label{tab:poly-ablation-results}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Rel.~$\ell_2\downarrow$ & Cos Sim$\uparrow$ & Latency (ms) \\")
    lines.append(r"\midrule")
    
    for r in large_results:
        method = r.get("method", "Unknown")
        rel_l2 = r.get("rel_l2", float("nan"))
        cos_sim = r.get("cos_sim", float("nan"))
        latency = r.get("ms", float("nan"))
        
        if rel_l2 == 0.0:
            rel_str = "0 (exact)"
        else:
            rel_str = format_number(rel_l2, ".4f")
        cos_str = format_number(cos_sim, ".4f")
        lat_str = format_number(latency, ".2f")
        
        lines.append(f"{method} & {rel_str} & {cos_str} & {lat_str} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_task_performance_table(data: Dict[str, Any]) -> str:
    """Generate task performance table."""
    if not data:
        return "% No task data found\n"
    
    results = data.get("results", [])
    
    # Get unique tasks and attentions
    tasks = sorted(set(r["task"] for r in results))
    attentions = sorted(set(r["attention"] for r in results))
    
    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r["task"], r["attention"])] = r
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Synthetic task accuracy by attention mechanism.}")
    lines.append(r"\label{tab:task-performance}")
    
    col_spec = "l" + "c" * len(attentions)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header - truncate long attention names
    header = "Task"
    for attn in attentions:
        short_name = attn[:8] if len(attn) > 8 else attn
        header += f" & {short_name}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    for task in tasks:
        row = task.replace("_", " ").title()
        for attn in attentions:
            r = lookup.get((task, attn), {})
            acc = r.get("final_accuracy", float("nan"))
            row += f" & {format_number(acc, '.3f')}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_max_context_table(data: Dict[str, Any]) -> str:
    """Generate max context table."""
    if not data:
        return "% No max context data found\n"
    
    config = data.get("config", {})
    results = data.get("results", {})
    memory_budget = config.get("memory_budget_gb", 8.0)
    
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
    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1].get("max_context", 0),
        reverse=True
    )
    
    for name, data in sorted_items:
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
    artifacts_dir = "../artifacts"
    tables_dir = "../tables"
    
    os.makedirs(tables_dir, exist_ok=True)
    
    print("=" * 70)
    print("Generating Paper Tables from Benchmark Results")
    print("=" * 70)
    
    # 1. Scaling tables (forward)
    scaling_fwd = load_json(os.path.join(artifacts_dir, "scaling_results_forward.json"))
    if scaling_fwd:
        latency_table = generate_scaling_latency_table(scaling_fwd, backward=False)
        with open(os.path.join(tables_dir, "latency_scaling_forward.tex"), "w") as f:
            f.write(latency_table)
        print("[OK] Generated latency_scaling_forward.tex")
        
        memory_table = generate_scaling_memory_table(scaling_fwd, backward=False)
        with open(os.path.join(tables_dir, "memory_scaling_forward.tex"), "w") as f:
            f.write(memory_table)
        print("[OK] Generated memory_scaling_forward.tex")
    else:
        print("[SKIP] No forward scaling results found")
    
    # 2. Scaling tables (backward)
    scaling_bwd = load_json(os.path.join(artifacts_dir, "scaling_results_backward.json"))
    if scaling_bwd:
        latency_table = generate_scaling_latency_table(scaling_bwd, backward=True)
        with open(os.path.join(tables_dir, "latency_scaling_backward.tex"), "w") as f:
            f.write(latency_table)
        print("[OK] Generated latency_scaling_backward.tex")
        
        memory_table = generate_scaling_memory_table(scaling_bwd, backward=True)
        with open(os.path.join(tables_dir, "memory_scaling_backward.tex"), "w") as f:
            f.write(memory_table)
        print("[OK] Generated memory_scaling_backward.tex")
    else:
        print("[SKIP] No backward scaling results found")
    
    # 3. Polynomial ablation
    poly_ablation = load_json(os.path.join(artifacts_dir, "poly_ablation.json"))
    if poly_ablation:
        table = generate_poly_ablation_table(poly_ablation)
        with open(os.path.join(tables_dir, "poly_ablation_results.tex"), "w") as f:
            f.write(table)
        print("[OK] Generated poly_ablation_results.tex")
    else:
        print("[SKIP] No polynomial ablation results found")
    
    # 4. Sweep results
    poly_sweep = load_json(os.path.join(artifacts_dir, "poly_ablation_sweep.json"))
    if poly_sweep:
        # Generate sweep summary
        results = poly_sweep.get("results", [])
        
        lines = ["# Polynomial Ablation Sweep Results\n"]
        lines.append("| Method | Features | Quad | Rel L2 | Cos Sim |")
        lines.append("|--------|----------|------|--------|---------|")
        
        for r in sorted(results, key=lambda x: x.get("rel_l2", float("inf"))):
            method = r.get("method", "")
            num_feat = r.get("num_features", "")
            num_quad = r.get("num_quad", "")
            rel_l2 = r.get("rel_l2", float("nan"))
            cos_sim = r.get("cos_sim", float("nan"))
            lines.append(f"| {method} | {num_feat} | {num_quad} | {rel_l2:.4f} | {cos_sim:.4f} |")
        
        with open(os.path.join(tables_dir, "poly_ablation_sweep.tex"), "w") as f:
            f.write("\n".join(lines))
        print("[OK] Generated poly_ablation_sweep.tex (markdown format)")
    else:
        print("[SKIP] No sweep results found")
    
    # 5. Task performance
    task_results = load_json(os.path.join(artifacts_dir, "task_results.json"))
    if task_results:
        table = generate_task_performance_table(task_results)
        with open(os.path.join(tables_dir, "task_performance.tex"), "w") as f:
            f.write(table)
        print("[OK] Generated task_performance.tex")
    else:
        print("[SKIP] No task results found")
    
    # 6. Max context
    max_context = load_json(os.path.join(artifacts_dir, "max_context.json"))
    if max_context:
        table = generate_max_context_table(max_context)
        with open(os.path.join(tables_dir, "max_context.tex"), "w") as f:
            f.write(table)
        print("[OK] Generated max_context.tex")
    else:
        print("[SKIP] No max context results found")
    
    print("\n" + "=" * 70)
    print("Done! Tables saved to:", tables_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
