"""
Synthetic Task Benchmarks for Attention Mechanisms

Tests attention quality on controlled tasks:
1. Copy: Reproduce input sequence (tests basic information flow)
2. Sorting: Sort input sequence (tests global dependencies)
3. Associative Recall: Key-value retrieval (tests memory)
4. Induction Heads: Pattern completion (tests in-context learning)
5. Needle in Haystack: Long-range retrieval

Run:
    python tests/benchmark_tasks.py --device cuda
    python tests/benchmark_tasks.py --device cuda --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import get_attention_class, list_attention_types


@dataclass
class TaskConfig:
    """Configuration for task benchmarks."""
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    batch_size: int = 32
    
    # Training
    num_steps: int = 500
    lr: float = 1e-3
    eval_every: int = 50
    
    device: str = "cuda"
    seed: int = 42
    
    # Yat-specific parameters
    yat_num_features: int = 32
    yat_num_quadrature_nodes: int = 2
    yat_poly_dim: int = 32
    yat_prf_dim: int = 16
    yat_epsilon: float = 1e-6


# Attention types for task benchmarks
TASK_ATTENTIONS = [
    "standard",
    "performer", 
    "linear",
    "yat-spherical",
    "yat-performer-anchor",
    "yat-performer-laplace",
    "yat-performer",
]


def get_attention_kwargs(name: str, cfg: TaskConfig) -> Dict[str, Any]:
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
    if name in {"yat-spherical", "yat-exact-spherical"}:
        return {"epsilon": cfg.yat_epsilon}
    if name == "performer":
        return {"kernel_size": 64}
    return {}


class TaskModel(nn.Module):
    """Simple Transformer model for synthetic tasks."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        attn_class,
        attn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(2048, embed_dim)  # Max seq len
        
        self.layers = nn.ModuleList([
            attn_class(embed_dim, n_heads, **attn_kwargs)
            for _ in range(n_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        
        h = self.embed(x) + self.pos_embed(positions)
        
        for layer in self.layers:
            h = h + layer(h)  # Residual
        
        h = self.ln(h)
        return self.head(h)


# ============================================================================
# Task Definitions
# ============================================================================

def make_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy task: input = output."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, x


def make_sort_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort task: output = sorted(input)."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y, _ = torch.sort(x, dim=1)
    return x, y


def make_associative_recall_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Associative recall: Given key-value pairs and a query key, output the value.
    
    Format: [k1, v1, k2, v2, ..., SEP, query_key] -> [*, *, *, *, ..., *, query_value]
    Only the last position matters for loss.
    """
    SEP = vocab_size - 1
    num_pairs = (seq_len - 2) // 2
    
    batch_x = []
    batch_y = []
    
    for _ in range(batch_size):
        # Random key-value pairs (keys must be unique)
        keys = torch.randperm(vocab_size - 1)[:num_pairs]
        vals = torch.randint(0, vocab_size - 1, (num_pairs,))
        
        # Build sequence
        seq = torch.zeros(seq_len, dtype=torch.long)
        for i, (k, v) in enumerate(zip(keys, vals)):
            seq[2 * i] = k
            seq[2 * i + 1] = v
        
        # Query
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        seq[-2] = SEP
        seq[-1] = keys[query_idx]
        
        # Target (only care about last position)
        target = torch.zeros(seq_len, dtype=torch.long)
        target[-1] = vals[query_idx]
        
        batch_x.append(seq)
        batch_y.append(target)
    
    return torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)


def make_induction_head_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Induction head task: [A, B, ..., A] -> [*, *, ..., B]
    
    Tests the model's ability to complete patterns seen earlier in context.
    """
    # Create random sequences
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x.clone()
    
    # For each batch, copy a token from earlier in the sequence
    for b in range(batch_size):
        # Pick a random position to be the "trigger"
        trigger_pos = torch.randint(1, seq_len // 2, (1,)).item()
        trigger_token = x[b, trigger_pos].item()
        following_token = x[b, trigger_pos + 1].item()
        
        # Place trigger token near the end
        query_pos = seq_len - 2
        x[b, query_pos] = trigger_token
        
        # Target is the token that followed the trigger
        y[b, query_pos + 1] = following_token
    
    return x, y


def make_needle_haystack_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Needle in haystack: Find a special token hidden in random noise.
    
    Model must output the position (encoded) or the needle value at the end.
    """
    NEEDLE = vocab_size - 1
    QUERY = vocab_size - 2
    
    x = torch.randint(0, vocab_size - 2, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        # Insert needle at random position
        needle_pos = torch.randint(0, seq_len - 2, (1,)).item()
        x[b, needle_pos] = NEEDLE
        
        # Last token is query
        x[b, -1] = QUERY
        
        # Target: output NEEDLE at last position
        y[b, -1] = NEEDLE
    
    return x, y


# Task registry
TASKS = {
    "copy": {
        "make_batch": make_copy_batch,
        "vocab_size": 32,
        "seq_len": 32,
        "loss_type": "full",  # Compute loss on all positions
        "description": "Copy input sequence",
    },
    "sort": {
        "make_batch": make_sort_batch,
        "vocab_size": 32,
        "seq_len": 16,
        "loss_type": "full",
        "description": "Sort input sequence",
    },
    "associative_recall": {
        "make_batch": make_associative_recall_batch,
        "vocab_size": 64,
        "seq_len": 22,  # 10 pairs + SEP + query
        "loss_type": "last",  # Only last position matters
        "description": "Key-value retrieval",
    },
    "induction": {
        "make_batch": make_induction_head_batch,
        "vocab_size": 64,
        "seq_len": 64,
        "loss_type": "last",
        "description": "Induction head (pattern completion)",
    },
    "needle": {
        "make_batch": make_needle_haystack_batch,
        "vocab_size": 64,
        "seq_len": 128,
        "loss_type": "last",
        "description": "Needle in haystack",
    },
}


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Compute loss based on task type."""
    if loss_type == "full":
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    elif loss_type == "last":
        # Only last position
        return F.cross_entropy(logits[:, -1, :], targets[:, -1])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str,
) -> float:
    """Compute accuracy based on task type."""
    preds = logits.argmax(dim=-1)
    
    if loss_type == "full":
        return (preds == targets).float().mean().item()
    elif loss_type == "last":
        return (preds[:, -1] == targets[:, -1]).float().mean().item()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_task(
    cfg: TaskConfig,
    attention_name: str,
    task_name: str,
) -> Dict[str, Any]:
    """Train a model on a single task and return metrics."""
    
    task = TASKS[task_name]
    make_batch = task["make_batch"]
    vocab_size = task["vocab_size"]
    seq_len = task["seq_len"]
    loss_type = task["loss_type"]
    
    # Create model
    attn_cls = get_attention_class(attention_name)
    attn_kwargs = get_attention_kwargs(attention_name, cfg)
    
    model = TaskModel(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        attn_class=attn_cls,
        attn_kwargs=attn_kwargs,
    ).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # Training
    losses = []
    accs = []
    
    start_time = time.time()
    
    for step in range(cfg.num_steps):
        model.train()
        x, y = make_batch(cfg.batch_size, seq_len, vocab_size, cfg.device)
        
        logits = model(x)
        loss = compute_loss(logits, y, loss_type)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Eval
        if (step + 1) % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_eval, y_eval = make_batch(cfg.batch_size * 2, seq_len, vocab_size, cfg.device)
                logits_eval = model(x_eval)
                acc = compute_accuracy(logits_eval, y_eval, loss_type)
                accs.append(acc)
    
    train_time = time.time() - start_time
    
    # Final eval
    model.eval()
    with torch.no_grad():
        x_eval, y_eval = make_batch(cfg.batch_size * 4, seq_len, vocab_size, cfg.device)
        logits_eval = model(x_eval)
        final_acc = compute_accuracy(logits_eval, y_eval, loss_type)
        final_loss = compute_loss(logits_eval, y_eval, loss_type).item()
    
    return {
        "final_loss": final_loss,
        "final_acc": final_acc,
        "train_time": train_time,
        "loss_curve": losses[::10],  # Subsample
        "acc_curve": accs,
    }


def run_task_benchmarks(
    cfg: TaskConfig,
    attention_names: List[str],
    task_names: List[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run all task benchmarks.
    
    Returns:
        results[task_name][attention_name] = {metrics}
    """
    results = {}
    
    for task_name in task_names:
        results[task_name] = {}
        print(f"\n{'='*60}")
        print(f"Task: {task_name} - {TASKS[task_name]['description']}")
        print(f"{'='*60}")
        
        for attn_name in attention_names:
            print(f"  {attn_name}...", end=" ", flush=True)
            
            try:
                torch.manual_seed(cfg.seed)
                metrics = train_task(cfg, attn_name, task_name)
                results[task_name][attn_name] = metrics
                print(f"OK Loss: {metrics['final_loss']:.4f}, Acc: {metrics['final_acc']:.3f}, Time: {metrics['train_time']:.1f}s")
                
            except Exception as e:
                results[task_name][attn_name] = {
                    "final_loss": float("nan"),
                    "final_acc": 0.0,
                    "train_time": float("nan"),
                    "error": str(e),
                }
                print(f"ERROR: {e}")
    
    return results


def format_latex_task_table(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format task results as LaTeX table."""
    
    task_names = list(results.keys())
    attn_names = list(next(iter(results.values())).keys())
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Synthetic task performance. We report final accuracy (higher is better).}")
    lines.append(r"\label{tab:task-performance}")
    
    # Column spec
    col_spec = "l" + "c" * len(task_names)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header
    header = "Method"
    for task in task_names:
        header += f" & {task.replace('_', ' ').title()}"
    header += r"\\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    for attn in attn_names:
        row = attn.replace("_", "\\_")
        for task in task_names:
            acc = results[task].get(attn, {}).get("final_acc", float("nan"))
            if acc != acc:  # NaN check
                row += " & --"
            else:
                row += f" & {acc:.3f}"
        row += r"\\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def format_markdown_task_table(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format task results as Markdown table."""
    
    task_names = list(results.keys())
    attn_names = list(next(iter(results.values())).keys())
    
    lines = []
    
    # Header
    header = "| Method |"
    sep = "|---|"
    for task in task_names:
        header += f" {task.replace('_', ' ').title()} |"
        sep += "---:|"
    lines.append(header)
    lines.append(sep)
    
    # Data rows
    for attn in attn_names:
        row = f"| {attn} |"
        for task in task_names:
            metrics = results[task].get(attn, {})
            acc = metrics.get("final_acc", float("nan"))
            loss = metrics.get("final_loss", float("nan"))
            if acc != acc:  # NaN
                row += " -- |"
            else:
                row += f" {acc:.3f} |"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Synthetic task benchmarks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer steps")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Tasks to run (default: all)",
    )
    parser.add_argument(
        "--attentions",
        nargs="*",
        default=None,
        help="Attention types to benchmark",
    )
    args = parser.parse_args()
    
    # Configure
    cfg = TaskConfig(
        device=args.device,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
    )
    
    if args.quick:
        cfg.num_steps = 100
        cfg.eval_every = 20
        cfg.batch_size = 16
    
    # Select tasks and attentions
    task_names = args.tasks if args.tasks else list(TASKS.keys())
    attention_names = args.attentions if args.attentions else TASK_ATTENTIONS
    
    print("=" * 70)
    print("SLAY Synthetic Task Benchmarks")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"Embed dim: {cfg.embed_dim}")
    print(f"Heads: {cfg.n_heads}")
    print(f"Layers: {cfg.n_layers}")
    print(f"Training steps: {cfg.num_steps}")
    print(f"Tasks: {task_names}")
    print(f"Attentions: {attention_names}")
    print("=" * 70)
    
    # Run benchmarks
    results = run_task_benchmarks(cfg, attention_names, task_names)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY (Accuracy)")
    print("=" * 70)
    print(format_markdown_task_table(results))
    
    # Save results
    os.makedirs("tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    # LaTeX
    latex = format_latex_task_table(results)
    with open("tables/task_performance.tex", "w") as f:
        f.write(latex)
    print(f"\nWrote tables/task_performance.tex")
    
    # Markdown
    md = format_markdown_task_table(results)
    with open("artifacts/task_performance.md", "w") as f:
        f.write(f"# Synthetic Task Performance\n\n{md}\n")
    print(f"Wrote artifacts/task_performance.md")
    
    # JSON
    with open("artifacts/task_results.json", "w") as f:
        # Remove non-serializable items
        json_results = {}
        for task, attn_results in results.items():
            json_results[task] = {}
            for attn, metrics in attn_results.items():
                json_results[task][attn] = {
                    k: v for k, v in metrics.items()
                    if k not in ["loss_curve", "acc_curve"]  # These can be large
                }
        json.dump({
            "config": asdict(cfg),
            "results": json_results,
        }, f, indent=2)
    print(f"Wrote artifacts/task_results.json")


if __name__ == "__main__":
    main()
