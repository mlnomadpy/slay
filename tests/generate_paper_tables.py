#!/usr/bin/env python3
"""
Generate Paper-Ready LaTeX Tables

Consolidates benchmark results from JSON artifacts into LaTeX tables
matching the format used in main.tex.

Usage:
    python tests/generate_paper_tables.py
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


def generate_poly_ablation_table(data: Dict[str, Any]) -> str:
    """
    Generate Table 1 (poly-ablation-snapshot) from ablation results.
    """
    if not data:
        return "% No ablation data found\n"
    
    # Find the "Large" scale results
    results = data.get("results", [])
    large_results = [r for r in results if r.get("scale") == "Large"]
    
    if not large_results:
        return "% No Large scale results found\n"
    
    # Sort by rel_l2
    large_results.sort(key=lambda r: r.get("rel_l2", float("inf")))
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Polynomial-approximation ablation (large-scale snapshot). Lower Rel.~$\ell_2$ is better; latency is forward-pass time.}")
    lines.append(r"\label{tab:poly-ablation-snapshot}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Rel.~$\ell_2\downarrow$ & Latency (ms)$\downarrow$ \\")
    lines.append(r"\midrule")
    
    for r in large_results:
        method = r.get("method", "Unknown").replace("&", r"\&")
        rel_l2 = r.get("rel_l2", float("nan"))
        latency = r.get("ms", float("nan"))
        
        rel_str = format_number(rel_l2, ".4f")
        lat_str = format_number(latency, ".2f")
        
        lines.append(f"{method} & {rel_str} & {lat_str}\\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_scaling_tables(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate scaling tables (latency and memory) for specific sequence lengths.
    """
    tables = {}
    
    if not data:
        return {"latency": "% No scaling data\n", "memory": "% No scaling data\n"}
    
    results = data.get("results", {})
    
    # Table for L=32768
    target_l = 32768
    
    # Latency table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Attention-only latency at $L={target_l}$ (ms).}}")
    lines.append(r"\label{tab:attn-latency-32768}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Forward-only & Forward+backward \\")
    lines.append(r"\midrule")
    
    # Group methods
    key_methods = ["standard", "yat-performer-anchor"]
    for method in key_methods:
        if method in results:
            if target_l in results[method]:
                lat = results[method][target_l].get("latency_ms", float("nan"))
                lat_str = format_number(lat, ".1f")
            else:
                lat_str = "--"
            
            method_display = "Standard attention" if method == "standard" else "Spherical $\\E$ attention"
            lines.append(f"{method_display} & {lat_str} & -- \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    tables["latency"] = "\n".join(lines)
    
    # Memory table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Peak memory at $L={target_l}$ (forward+backward).}}")
    lines.append(r"\label{tab:attn-mem-32768}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Peak memory (GB) \\")
    lines.append(r"\midrule")
    
    for method in key_methods:
        if method in results:
            if target_l in results[method]:
                mem = results[method][target_l].get("memory_mb", float("nan"))
                if mem and mem == mem:  # Not NaN
                    mem_str = format_number(mem / 1024, ".1f")
                else:
                    mem_str = "--"
            else:
                mem_str = "--"
            
            method_display = "Standard attention" if method == "standard" else "Spherical $\\E$ attention"
            lines.append(f"{method_display} & {mem_str} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    tables["memory"] = "\n".join(lines)
    
    return tables


def generate_task_table(data: Dict[str, Any]) -> str:
    """Generate synthetic task performance table."""
    if not data:
        return "% No task data\n"
    
    results = data.get("results", {})
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Synthetic task performance (accuracy).}")
    lines.append(r"\label{tab:task-performance}")
    
    tasks = list(results.keys())
    attentions = list(results.get(tasks[0], {}).keys()) if tasks else []
    
    col_spec = "l" + "c" * len(tasks)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header
    header = "Method"
    for task in tasks:
        header += f" & {task.replace('_', ' ').title()}"
    header += r"\\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data
    for attn in attentions:
        row = attn.replace("_", r"\_")
        for task in tasks:
            acc = results.get(task, {}).get(attn, {}).get("final_acc", float("nan"))
            row += f" & {format_number(acc, '.3f')}"
        row += r"\\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_kernel_quality_table(data: Dict[str, Any]) -> str:
    """Generate kernel approximation quality table."""
    if not data:
        return "% No kernel quality data\n"
    
    results = data.get("results", {})
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Kernel approximation quality against exact spherical Yat attention.}")
    lines.append(r"\label{tab:kernel-quality}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Rel.~$\ell_2\downarrow$ & Cos$\uparrow$ & MSE$\downarrow$\\")
    lines.append(r"\midrule")
    
    # Sort by rel_l2
    sorted_names = sorted(results.keys(), key=lambda n: results[n].get("rel_l2", float("inf")))
    
    for name in sorted_names:
        r = results[name]
        rel_l2 = r.get("rel_l2", float("nan"))
        cos = r.get("cosine", float("nan"))
        mse = r.get("mse", float("nan"))
        
        lines.append(f"{name.replace('_', '\\_')} & {format_number(rel_l2, '.4f')} & {format_number(cos, '.4f')} & {format_number(mse, '.2e')}\\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Generating Paper Tables")
    print("=" * 70)
    
    artifacts_dir = Path("artifacts")
    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)
    
    # Load all available data
    ablation_data = load_json(artifacts_dir / "poly_ablation_sweep.json")
    if not ablation_data:
        # Try loading from the sweep markdown (parse it)
        pass
    
    scaling_forward = load_json(artifacts_dir / "scaling_results_forward.json")
    scaling_backward = load_json(artifacts_dir / "scaling_results_backward.json")
    task_data = load_json(artifacts_dir / "task_results.json")
    kernel_data = load_json(artifacts_dir / "kernel_quality.json")
    
    generated = []
    
    # 1. Poly ablation table
    if ablation_data:
        table = generate_poly_ablation_table(ablation_data)
        with open(tables_dir / "poly_ablation_snapshot.tex", "w") as f:
            f.write(table)
        generated.append("poly_ablation_snapshot.tex")
        print("[OK] Generated poly_ablation_snapshot.tex")
    else:
        print("[!] No ablation data found")
    
    # 2. Scaling tables
    if scaling_forward:
        tables = generate_scaling_tables(scaling_forward)
        for name, content in tables.items():
            filename = f"scaling_{name}.tex"
            with open(tables_dir / filename, "w") as f:
                f.write(content)
            generated.append(filename)
            print(f"[OK] Generated {filename}")
    else:
        print("[!] No scaling data found")
    
    # 3. Task table
    if task_data:
        table = generate_task_table(task_data)
        with open(tables_dir / "task_performance_paper.tex", "w") as f:
            f.write(table)
        generated.append("task_performance_paper.tex")
        print("[OK] Generated task_performance_paper.tex")
    else:
        print("[!] No task data found")
    
    # 4. Kernel quality table
    if kernel_data:
        table = generate_kernel_quality_table(kernel_data)
        with open(tables_dir / "kernel_quality_paper.tex", "w") as f:
            f.write(table)
        generated.append("kernel_quality_paper.tex")
        print("[OK] Generated kernel_quality_paper.tex")
    else:
        print("[!] No kernel quality data found")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Generated {len(generated)} table(s):")
    for f in generated:
        print(f"  - tables/{f}")
    
    print("\nTo include in your paper:")
    print("  \\input{tables/poly_ablation_snapshot.tex}")
    print("  \\input{tables/kernel_quality_paper.tex}")
    print("  \\input{tables/task_performance_paper.tex}")


if __name__ == "__main__":
    main()
