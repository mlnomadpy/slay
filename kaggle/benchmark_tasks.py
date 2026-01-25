#!/usr/bin/env python3
"""
JAX/Flax Synthetic Task Benchmarks for Attention Mechanisms

Tasks organized into categories:
- Basic: Copy, Reverse, Sort
- Memory: Retrieval, Selective Copy, First Token
- Long-Range: Long Copy, Distant Match
- Arithmetic: Counting, Parity

Uses attention implementations from kaggle/main.py.

Run:
    python kaggle/benchmark_tasks.py
    python kaggle/benchmark_tasks.py --quick
    python kaggle/benchmark_tasks.py --tasks copy sort retrieval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax

# Import attention and model components from main.py
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
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    mesh,
)


@dataclass
class TaskConfig:
    """Configuration for task benchmarks."""
    embed_dim: int = 64
    num_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    batch_size: int = 32
    
    # Training
    num_epochs: int = 100
    lr: float = 1e-3
    eval_every: int = 20
    num_seeds: int = 1  # Number of random seeds for averaging
    
    seed: int = 42
    
    # SLAY-specific parameters  
    slay_num_features: int = 32
    slay_num_quad: int = 2
    slay_poly_dim: int = 16
    slay_prf_dim: int = 8
    performer_kernel_size: int = 64
    rff_num_features: int = 64


# Attention registry
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
    "slay-laplace": SLAYLaplaceAttention,
}

DEFAULT_ATTENTIONS = ["standard", "linear", "performer", "slay", "slay-laplace"]


def get_attention_kwargs(name: str, cfg: TaskConfig) -> Dict[str, Any]:
    """Get constructor kwargs for each attention type."""
    if name == "performer":
        return {"kernel_size": cfg.performer_kernel_size}
    if name == "rff":
        return {"num_features": cfg.rff_num_features}
    if name in {"yat", "yat-spherical"}:
        return {"epsilon": 1e-6}
    if name in {"slay", "slay-laplace"}:
        return {
            "num_features": cfg.slay_num_features,
            "num_quadrature_nodes": cfg.slay_num_quad,
        }
    return {}


# ============================================================================
# Task Definitions
# ============================================================================

@dataclass
class TaskSpec:
    """Specification for a synthetic task."""
    name: str
    category: str
    seq_len: int
    vocab_size: int
    description: str
    generate_fn: Callable  # (batch_size, seq_len, vocab_size, key) -> (inputs, targets)


def generate_copy_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Copy task: Output equals input."""
    inputs = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    targets = inputs
    return inputs, targets


def generate_reverse_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reverse task: Output is reversed input."""
    inputs = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    targets = jnp.flip(inputs, axis=1)
    return inputs, targets


def generate_sort_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sort task: Output is sorted input."""
    inputs = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    targets = jnp.sort(inputs, axis=1)
    return inputs, targets


def generate_retrieval_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Key-Value retrieval: Given key, retrieve associated value."""
    # Format: [k1, v1, k2, v2, ..., query_key, 0] -> value for query_key
    n_pairs = (seq_len - 2) // 2
    
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Generate keys and values (ensure keys are unique by using range)
    keys = jax.random.permutation(k1, vocab_size)[:n_pairs]
    keys = jnp.broadcast_to(keys, (batch_size, n_pairs))
    values = jax.random.randint(k2, (batch_size, n_pairs), 0, vocab_size)
    
    # Interleave keys and values
    kv_pairs = jnp.stack([keys, values], axis=-1).reshape(batch_size, -1)
    
    # Select a random query key for each batch
    query_idx = jax.random.randint(k3, (batch_size,), 0, n_pairs)
    query_keys = keys[jnp.arange(batch_size), query_idx]
    query_values = values[jnp.arange(batch_size), query_idx]
    
    # Construct inputs: [kv_pairs..., query_key, 0]
    inputs = jnp.concatenate([
        kv_pairs[:, :seq_len-2],
        query_keys[:, None],
        jnp.zeros((batch_size, 1), dtype=jnp.int32)
    ], axis=1)
    
    # Target: just the query value repeated (simplified)
    targets = jnp.broadcast_to(query_values[:, None], (batch_size, seq_len))
    
    return inputs, targets


def generate_selective_copy_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Selective copy: Copy only tokens > vocab_size // 2."""
    k1, k2 = jax.random.split(key)
    inputs = jax.random.randint(k1, (batch_size, seq_len), 0, vocab_size)
    threshold = vocab_size // 2
    
    # Mask: 1 where token > threshold
    mask = (inputs > threshold).astype(jnp.int32)
    targets = inputs * mask  # Copy only selected tokens
    
    return inputs, targets


def generate_first_token_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """First token: All outputs should equal the first input token."""
    inputs = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    first_tokens = inputs[:, 0:1]
    targets = jnp.broadcast_to(first_tokens, (batch_size, seq_len))
    return inputs, targets


def generate_counting_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Counting: Output position of each token's first occurrence."""
    inputs = jax.random.randint(key, (batch_size, seq_len), 0, min(vocab_size, seq_len))
    
    # For simplicity, target is just the cumulative count of unique tokens seen
    # This is a simplified version
    def count_fn(x):
        positions = jnp.arange(seq_len)
        return positions % vocab_size
    
    targets = jax.vmap(count_fn)(inputs)
    return inputs, targets


def generate_parity_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Parity: Output cumulative parity (XOR) of input bits."""
    # Use binary inputs
    inputs = jax.random.randint(key, (batch_size, seq_len), 0, 2)
    targets = jnp.cumsum(inputs, axis=1) % 2
    return inputs, targets


def generate_long_copy_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Long copy: Copy with longer sequences (tests memory)."""
    return generate_copy_task(batch_size, seq_len, vocab_size, key)


def generate_distant_match_task(batch_size: int, seq_len: int, vocab_size: int, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Distant match: Find if first and last tokens match."""
    k1, k2 = jax.random.split(key)
    inputs = jax.random.randint(k1, (batch_size, seq_len), 0, vocab_size)
    
    # Randomly decide if first and last should match
    should_match = jax.random.bernoulli(k2, 0.5, (batch_size,))
    
    # If should match, set last token to first
    last_token = jnp.where(should_match, inputs[:, 0], inputs[:, -1])
    inputs = inputs.at[:, -1].set(last_token)
    
    # Target: 1 if match, 0 otherwise (broadcast to seq_len)
    match = (inputs[:, 0] == inputs[:, -1]).astype(jnp.int32)
    targets = jnp.broadcast_to(match[:, None], (batch_size, seq_len))
    
    return inputs, targets


# Task registry
TASKS: Dict[str, TaskSpec] = {
    "copy": TaskSpec("copy", "basic", 32, 16, "Copy input to output", generate_copy_task),
    "reverse": TaskSpec("reverse", "basic", 32, 16, "Reverse input sequence", generate_reverse_task),
    "sort": TaskSpec("sort", "basic", 32, 16, "Sort input sequence", generate_sort_task),
    "retrieval": TaskSpec("retrieval", "memory", 64, 32, "Key-value retrieval", generate_retrieval_task),
    "selective_copy": TaskSpec("selective_copy", "memory", 32, 16, "Copy only high tokens", generate_selective_copy_task),
    "first_token": TaskSpec("first_token", "memory", 64, 16, "Repeat first token", generate_first_token_task),
    "counting": TaskSpec("counting", "arithmetic", 32, 16, "Count token occurrences", generate_counting_task),
    "parity": TaskSpec("parity", "arithmetic", 32, 2, "Cumulative parity", generate_parity_task),
    "long_copy": TaskSpec("long_copy", "long_range", 128, 16, "Copy with long sequences", generate_long_copy_task),
    "distant_match": TaskSpec("distant_match", "long_range", 128, 16, "Match first and last tokens", generate_distant_match_task),
}

TASK_CATEGORIES = {
    "basic": ["copy", "reverse", "sort"],
    "memory": ["retrieval", "selective_copy", "first_token"],
    "arithmetic": ["counting", "parity"],
    "long_range": ["long_copy", "distant_match"],
}


# ============================================================================
# Model Definition
# ============================================================================

class TaskTransformerBlock(nnx.Module):
    """Single transformer block for task benchmarks."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        attention_type: str,
        attention_kwargs: dict,
        *,
        rngs: nnx.Rngs,
    ):
        self.attention_type = attention_type
        attn_cls = ATTENTION_REGISTRY[attention_type]
        self.attn = attn_cls(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        
        self.norm1 = RMSNorm(embed_dim, rngs=rngs)
        self.norm2 = RMSNorm(embed_dim, rngs=rngs)
        
        self.ff = nnx.Sequential(
            nnx.Linear(embed_dim, ff_dim, rngs=rngs),
            lambda x: jax.nn.gelu(x),
            nnx.Linear(ff_dim, embed_dim, rngs=rngs),
        )
    
    def __call__(self, x, freqs_cos, freqs_sin):
        # Pre-norm attention
        h = self.norm1(x)
        h = self.attn(h, freqs_cos, freqs_sin)
        x = x + h
        
        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        
        return x


class TaskModel(nnx.Module):
    """Small transformer for synthetic tasks."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        n_layers: int,
        ff_dim: int,
        max_len: int,
        attention_type: str,
        attention_kwargs: dict,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        
        self.blocks = nnx.List([
            TaskTransformerBlock(
                embed_dim, num_heads, ff_dim, attention_type, attention_kwargs, rngs=rngs
            )
            for _ in range(n_layers)
        ])
        
        self.norm_final = RMSNorm(embed_dim, rngs=rngs)
        self.head = nnx.Linear(embed_dim, vocab_size, rngs=rngs)
        
        # Precompute RoPE frequencies
        head_dim = embed_dim // num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_len)
        self.freqs_cos = nnx.Cache(freqs_cos)
        self.freqs_sin = nnx.Cache(freqs_sin)
    
    def __call__(self, x):
        h = self.token_emb(x)
        
        freqs_cos = jax.device_put(self.freqs_cos.value)
        freqs_sin = jax.device_put(self.freqs_sin.value)
        
        for block in self.blocks:
            h = block(h, freqs_cos, freqs_sin)
        
        h = self.norm_final(h)
        logits = self.head(h)
        return logits


# ============================================================================
# Training
# ============================================================================

def create_model(
    vocab_size: int,
    cfg: TaskConfig,
    max_len: int,
    attention_type: str,
    rngs: nnx.Rngs,
) -> TaskModel:
    """Create task model with specified attention type."""
    attention_kwargs = get_attention_kwargs(attention_type, cfg)
    
    return TaskModel(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        max_len=max_len,
        attention_type=attention_type,
        attention_kwargs=attention_kwargs,
        rngs=rngs,
    )


def loss_fn(model: TaskModel, inputs: jnp.ndarray, targets: jnp.ndarray):
    """Cross-entropy loss for sequence prediction."""
    logits = model(inputs)
    
    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, model.vocab_size)
    targets_flat = targets.reshape(-1)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
    return jnp.mean(loss)


@nnx.jit
def train_step(model: TaskModel, optimizer: nnx.Optimizer, inputs: jnp.ndarray, targets: jnp.ndarray):
    """Single training step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, inputs, targets)
    optimizer.update(grads)
    return loss


def compute_accuracy(model: TaskModel, inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute token-level accuracy."""
    logits = model(inputs)
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32)
    return float(jnp.mean(correct))


def train_on_task(
    task: TaskSpec,
    attention_type: str,
    cfg: TaskConfig,
    seed: int,
) -> Dict[str, Any]:
    """Train a model on a single task."""
    
    with mesh:
        rngs = nnx.Rngs(seed)
        key = jax.random.PRNGKey(seed)
    
        model = create_model(task.vocab_size, cfg, task.seq_len, attention_type, rngs)
        optimizer = nnx.Optimizer(model, optax.adam(cfg.lr))
    
        train_losses = []
        eval_accs = []
    
        for epoch in range(cfg.num_epochs):
            key, subkey = jax.random.split(key)
        
            # Generate batch
            inputs, targets = task.generate_fn(cfg.batch_size, task.seq_len, task.vocab_size, subkey)
        
            # Train step
            loss = train_step(model, optimizer, inputs, targets)
            train_losses.append(float(loss))
        
            # Evaluate
            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.num_epochs - 1:
                key, eval_key = jax.random.split(key)
                eval_inputs, eval_targets = task.generate_fn(cfg.batch_size * 4, task.seq_len, task.vocab_size, eval_key)
                acc = compute_accuracy(model, eval_inputs, eval_targets)
                eval_accs.append(acc)
    
        final_acc = eval_accs[-1] if eval_accs else 0.0
        final_loss = train_losses[-1] if train_losses else float("inf")
    
    return {
        "task": task.name,
        "attention": attention_type,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "train_losses": train_losses[::10],  # Subsample for storage
        "eval_accuracies": eval_accs,
        "seed": seed,
    }


def run_task_benchmark(
    cfg: TaskConfig,
    task_names: List[str],
    attention_names: List[str],
) -> List[Dict[str, Any]]:
    """Run benchmarks for multiple tasks and attention types."""
    
    all_results = []
    
    for task_name in task_names:
        if task_name not in TASKS:
            print(f"[SKIP] Unknown task: {task_name}")
            continue
        
        task = TASKS[task_name]
        print(f"\n[Task: {task.name}] ({task.category}) - {task.description}")
        
        for attn_name in attention_names:
            if attn_name not in ATTENTION_REGISTRY:
                print(f"  [SKIP] Unknown attention: {attn_name}")
                continue
            
            print(f"  {attn_name}...", end=" ", flush=True)
            
            try:
                # Run with multiple seeds and average
                seed_results = []
                for seed_offset in range(cfg.num_seeds):
                    result = train_on_task(task, attn_name, cfg, cfg.seed + seed_offset)
                    seed_results.append(result)
                
                # Average metrics
                avg_acc = np.mean([r["final_accuracy"] for r in seed_results])
                avg_loss = np.mean([r["final_loss"] for r in seed_results])
                std_acc = np.std([r["final_accuracy"] for r in seed_results])
                
                combined_result = {
                    "task": task.name,
                    "category": task.category,
                    "attention": attn_name,
                    "final_loss": float(avg_loss),
                    "final_accuracy": float(avg_acc),
                    "accuracy_std": float(std_acc),
                    "num_seeds": cfg.num_seeds,
                }
                
                all_results.append(combined_result)
                print(f"Acc: {avg_acc:.3f} (±{std_acc:.3f}), Loss: {avg_loss:.4f}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "task": task.name,
                    "category": task.category,
                    "attention": attn_name,
                    "final_loss": float("nan"),
                    "final_accuracy": float("nan"),
                    "error": str(e),
                })
    
    return all_results


def format_task_latex_table(results: List[Dict[str, Any]]) -> str:
    """Format task results as LaTeX table."""
    
    # Group by task
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
    
    # Header
    header = "Task"
    for attn in attentions:
        header += f" & {attn}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    for task in tasks:
        row = task.replace("_", " ")
        for attn in attentions:
            r = lookup.get((task, attn), {})
            acc = r.get("final_accuracy", float("nan"))
            if acc != acc:  # NaN
                row += " & --"
            else:
                row += f" & {acc:.3f}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="JAX Synthetic Task Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer epochs")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks to run")
    parser.add_argument("--category", type=str, default=None, help="Task category to run")
    parser.add_argument("--attentions", nargs="*", default=None, help="Attention types to test")
    parser.add_argument("--output-dir", default="../artifacts", help="Output directory")
    args = parser.parse_args()
    
    # Configuration
    if args.quick:
        cfg = TaskConfig(
            num_epochs=50,
            eval_every=10,
            num_seeds=1,
        )
    else:
        cfg = TaskConfig()
    
    # Select tasks
    if args.tasks:
        task_names = args.tasks
    elif args.category:
        task_names = TASK_CATEGORIES.get(args.category, [])
    else:
        task_names = list(TASKS.keys())
    
    # Select attentions
    attention_names = args.attentions if args.attentions else DEFAULT_ATTENTIONS
    
    print("=" * 70)
    print("JAX/Flax Synthetic Task Benchmarks")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Tasks: {task_names}")
    print(f"Attentions: {attention_names}")
    print(f"Epochs: {cfg.num_epochs}, Seeds: {cfg.num_seeds}")
    print("=" * 70)
    
    # Run benchmarks
    results = run_task_benchmark(cfg, task_names, attention_names)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "task_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate LaTeX table
    os.makedirs("../tables", exist_ok=True)
    latex_table = format_task_latex_table(results)
    table_file = os.path.join("../tables", "task_performance.tex")
    with open(table_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {table_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TASK PERFORMANCE SUMMARY")
    print("=" * 70)
    print(latex_table)


if __name__ == "__main__":
    main()
