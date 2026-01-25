"""
Comprehensive Synthetic Task Benchmarks for Attention Mechanisms

25 tasks organized into categories:
- Basic: Copy, Sort, Reverse
- Memory: Retrieval, KVRecall, FirstToken, SelectiveCopy
- Long-Range: LongCopy, DistantMatch, VeryLongCopy
- Reasoning: MultiHop, Stack, Induction
- Arithmetic: Counting, Parity, Addition, ModularArith
- Pattern: Pattern, Bigram, Majority
- Robustness: NoisyCopy, AdversarialRetrieval
- Compression: Compression, Histogram

Run:
    python tests/benchmark_tasks.py --device cuda
    python tests/benchmark_tasks.py --device cuda --quick
    python tests/benchmark_tasks.py --device cuda --tasks copy sort retrieval
    python tests/benchmark_tasks.py --device cuda --category basic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import get_attention_class

def get_best_device(requested: str | None = None) -> torch.device:
    """
    Select the best available device.
    Priority:
      1. User-requested device (if valid)
      2. CUDA (NVIDIA)
      3. MPS (Apple Silicon)
      4. CPU
    """
    if requested:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        print(f"⚠️ Requested device '{requested}' not available, auto-selecting.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def is_mps(device) -> bool:
    return isinstance(device, torch.device) and device.type == "mps"

@dataclass
class TaskConfig:
    """Configuration for task benchmarks."""
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    batch_size: int = 32
    
    # Training
    num_epochs: int = 200
    lr: float = 1e-3
    eval_every: int = 20
    num_seeds: int = 3  # For statistical significance
    
    device: torch.device = torch.device("cpu")
    seed: int = 42
    
    # Yat-specific parameters
    yat_num_features: int = 32
    yat_num_quadrature_nodes: int = 2
    yat_poly_dim: int = 32
    yat_prf_dim: int = 16
    yat_epsilon: float = 1e-6


# Default attention types for benchmarks
DEFAULT_ATTENTIONS = [
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
    """Attention-only model for synthetic tasks (no FFN, minimal LN)."""
    
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
        self.pos_embed = nn.Embedding(4096, embed_dim)  # Max seq len
        
        # Pure attention layers - no FFN
        self.attns = nn.ModuleList([
            attn_class(embed_dim, n_heads, **attn_kwargs)
            for _ in range(n_layers)
        ])
        
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        
        h = self.embed(x) + self.pos_embed(positions)
        
        for attn in self.attns:
            h = h + attn(h)
        
        return self.head(h)


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def generate_copy_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Copy task: reproduce input sequence."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, x, "sequence"


def generate_sort_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Sort task: sort input sequence."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y, _ = torch.sort(x, dim=1)
    return x, y, "sequence"


def generate_reverse_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Reverse task: reverse input sequence."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.flip(x, dims=[1])
    return x, y, "sequence"


def generate_retrieval_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Retrieval (needle in haystack): find special token position."""
    NEEDLE = vocab_size - 1
    QUERY = vocab_size - 2
    
    x = torch.randint(0, vocab_size - 2, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        needle_pos = torch.randint(0, seq_len - 2, (1,)).item()
        x[b, needle_pos] = NEEDLE
        x[b, -1] = QUERY
        y[b, -1] = NEEDLE
    
    return x, y, "last"


def generate_kv_recall_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Key-value recall: retrieve value for queried key."""
    SEP = vocab_size - 1
    num_pairs = (seq_len - 2) // 2
    
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        keys = torch.randperm(vocab_size - 1)[:num_pairs]
        vals = torch.randint(0, vocab_size - 1, (num_pairs,))
        
        for i, (k, v) in enumerate(zip(keys, vals)):
            x[b, 2 * i] = k
            x[b, 2 * i + 1] = v
        
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        x[b, -2] = SEP
        x[b, -1] = keys[query_idx]
        y[b, -1] = vals[query_idx]
    
    return x, y, "last"


def generate_first_token_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """First token recall: predict first token at last position."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y[:, -1] = x[:, 0]
    return x, y, "last"


def generate_selective_copy_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Selective copy: copy only marked tokens."""
    MARKER = vocab_size - 1
    half = seq_len // 2
    
    x = torch.randint(0, vocab_size - 1, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        # Mark ~25% of first half tokens
        num_marked = max(1, half // 4)
        marked_positions = torch.randperm(half)[:num_marked].sort()[0]
        
        for i, pos in enumerate(marked_positions):
            x[b, pos] = MARKER
            if i < len(marked_positions):
                y[b, half + i] = x[b, pos + 1] if pos + 1 < half else 0
    
    return x, y, "sequence"


def generate_long_copy_data(batch_size: int, seq_len: int, vocab_size: int, device):
    """
    Long copy task.
    On MPS, cap sequence length to avoid MPSNDArray INT_MAX crash.
    """
    if is_mps(device):
        # Safe upper bound for Apple GPU
        actual_len = min(128, seq_len * 2)
    else:
        actual_len = min(256, seq_len * 4)

    x = torch.randint(0, vocab_size, (batch_size, actual_len), device=device)
    return x, x, "sequence"


def generate_distant_match_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Distant match: match token from beginning to end."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Target at position seq_len//2 should match position 0
    y[:, seq_len // 2] = x[:, 0]
    return x, y, "position"


def generate_multihop_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Multi-hop reasoning: follow chain of references."""
    x = torch.randint(0, vocab_size - 2, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        # Create chain: pos0 -> pos1 -> pos2
        pos0 = torch.randint(0, seq_len // 3, (1,)).item()
        pos1 = torch.randint(seq_len // 3, 2 * seq_len // 3, (1,)).item()
        pos2 = torch.randint(2 * seq_len // 3, seq_len - 1, (1,)).item()
        
        x[b, pos0] = vocab_size - 2  # Pointer 1
        x[b, pos1] = vocab_size - 1  # Pointer 2
        
        # Answer is what's after pos1
        y[b, -1] = x[b, pos1 + 1] if pos1 + 1 < seq_len else 0
    
    return x, y, "last"


def generate_stack_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Stack operations: PUSH/POP simulation."""
    PUSH = vocab_size - 2
    POP = vocab_size - 1
    
    x = torch.randint(0, vocab_size - 2, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        stack = []
        for t in range(seq_len):
            if torch.rand(1) < 0.3 and len(stack) > 0:
                x[b, t] = POP
                y[b, t] = stack.pop()
            elif torch.rand(1) < 0.5:
                val = torch.randint(0, vocab_size - 2, (1,)).item()
                x[b, t] = PUSH
                stack.append(val)
    
    return x, y, "sequence"


def generate_induction_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Induction heads: complete [A, B, ..., A] -> B pattern."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x.clone()
    
    for b in range(batch_size):
        trigger_pos = torch.randint(1, seq_len // 2, (1,)).item()
        trigger_token = x[b, trigger_pos].item()
        following_token = x[b, trigger_pos + 1].item()
        
        query_pos = seq_len - 2
        x[b, query_pos] = trigger_token
        y[b, query_pos + 1] = following_token
    
    return x, y, "last"


def generate_counting_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Counting: count occurrences of token 0."""
    x = torch.randint(0, min(vocab_size, 10), (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        count = (x[b] == 0).sum().item()
        y[b, -1] = min(count, vocab_size - 1)
    
    return x, y, "last"


def generate_parity_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Parity: is count of 1s even (0) or odd (1)?"""
    x = torch.randint(0, 2, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        count_ones = (x[b] == 1).sum().item()
        y[b, -1] = count_ones % 2
    
    return x, y, "last"


def generate_addition_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Addition: sum of binary digits (mod vocab_size)."""
    x = torch.randint(0, 2, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        total = x[b].sum().item()
        y[b, -1] = total % vocab_size
    
    return x, y, "last"


def generate_modular_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Modular arithmetic: sum mod small prime."""
    MOD = 7
    x = torch.randint(0, MOD, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        total = x[b].sum().item() % MOD
        y[b, -1] = total
    
    return x, y, "last"


def generate_pattern_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Pattern: predict next in repeating pattern."""
    pattern_len = 4
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        pattern = torch.randint(0, vocab_size, (pattern_len,))
        for t in range(seq_len):
            x[b, t] = pattern[t % pattern_len]
            y[b, t] = pattern[(t + 1) % pattern_len]
    
    return x, y, "sequence"


def generate_bigram_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Bigram: predict based on previous token (learned bigram table)."""
    # Fixed bigram table
    torch.manual_seed(0)
    bigram_table = torch.randint(0, vocab_size, (vocab_size,))
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = bigram_table[x]
    
    return x, y, "sequence"


def generate_majority_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Majority: predict most frequent token."""
    x = torch.randint(0, min(vocab_size, 5), (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        counts = torch.bincount(x[b], minlength=5)
        y[b, -1] = counts.argmax()
    
    return x, y, "last"


def generate_noisy_copy_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Noisy copy: copy with random noise tokens inserted."""
    NOISE = vocab_size - 1
    half = seq_len // 2
    
    x = torch.randint(0, vocab_size - 1, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Insert noise in second half
    for b in range(batch_size):
        noise_mask = torch.rand(half) < 0.3
        x[b, half:][noise_mask] = NOISE
        y[b, :half] = x[b, :half]  # Copy first half
    
    return x, y, "sequence"


def generate_compression_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Compression: remove repeated tokens."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Make input have runs
    for b in range(batch_size):
        pos = 0
        while pos < seq_len:
            val = torch.randint(0, vocab_size, (1,)).item()
            run_len = torch.randint(1, 4, (1,)).item()
            for i in range(min(run_len, seq_len - pos)):
                x[b, pos + i] = val
            pos += run_len
        
        # Target: compressed (unique consecutive)
        out_pos = 0
        prev = -1
        for t in range(seq_len):
            if x[b, t].item() != prev:
                y[b, out_pos] = x[b, t]
                prev = x[b, t].item()
                out_pos += 1
    
    return x, y, "sequence"


def generate_histogram_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Histogram: output token counts."""
    NUM_TOKENS = min(5, vocab_size)
    x = torch.randint(0, NUM_TOKENS, (batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        counts = torch.bincount(x[b], minlength=NUM_TOKENS)
        y[b, :NUM_TOKENS] = counts.clamp(max=vocab_size - 1)
    
    return x, y, "sequence"


# Task registry with metadata
TASKS = {
    # Basic (3)
    "copy": {"fn": generate_copy_data, "seq_len": 32, "vocab_size": 32, "category": "basic"},
    "sort": {"fn": generate_sort_data, "seq_len": 16, "vocab_size": 32, "category": "basic"},
    "reverse": {"fn": generate_reverse_data, "seq_len": 32, "vocab_size": 32, "category": "basic"},
    
    # Memory (4)
    "retrieval": {"fn": generate_retrieval_data, "seq_len": 64, "vocab_size": 64, "category": "memory"},
    "kv_recall": {"fn": generate_kv_recall_data, "seq_len": 22, "vocab_size": 64, "category": "memory"},
    "first_token": {"fn": generate_first_token_data, "seq_len": 64, "vocab_size": 32, "category": "memory"},
    "selective_copy": {"fn": generate_selective_copy_data, "seq_len": 64, "vocab_size": 32, "category": "memory"},
    
    # Long-Range (3)
    "long_copy": {"fn": generate_long_copy_data, "seq_len": 64, "vocab_size": 32, "category": "long_range"},
    "distant_match": {"fn": generate_distant_match_data, "seq_len": 128, "vocab_size": 32, "category": "long_range"},
    "multihop": {"fn": generate_multihop_data, "seq_len": 64, "vocab_size": 32, "category": "long_range"},
    
    # Reasoning (3)
    "stack": {"fn": generate_stack_data, "seq_len": 32, "vocab_size": 32, "category": "reasoning"},
    "induction": {"fn": generate_induction_data, "seq_len": 64, "vocab_size": 64, "category": "reasoning"},
    "pattern": {"fn": generate_pattern_data, "seq_len": 32, "vocab_size": 16, "category": "reasoning"},
    
    # Arithmetic (4)
    "counting": {"fn": generate_counting_data, "seq_len": 32, "vocab_size": 32, "category": "arithmetic"},
    "parity": {"fn": generate_parity_data, "seq_len": 32, "vocab_size": 2, "category": "arithmetic"},
    "addition": {"fn": generate_addition_data, "seq_len": 16, "vocab_size": 32, "category": "arithmetic"},
    "modular": {"fn": generate_modular_data, "seq_len": 16, "vocab_size": 8, "category": "arithmetic"},
    
    # Pattern (2)
    "bigram": {"fn": generate_bigram_data, "seq_len": 32, "vocab_size": 32, "category": "pattern"},
    "majority": {"fn": generate_majority_data, "seq_len": 32, "vocab_size": 8, "category": "pattern"},
    
    # Robustness (2)
    "noisy_copy": {"fn": generate_noisy_copy_data, "seq_len": 64, "vocab_size": 32, "category": "robustness"},
    "compression": {"fn": generate_compression_data, "seq_len": 32, "vocab_size": 16, "category": "robustness"},
    
    # Other (1)
    "histogram": {"fn": generate_histogram_data, "seq_len": 32, "vocab_size": 16, "category": "other"},
}

# Task categories
CATEGORIES = {
    "basic": ["copy", "sort", "reverse"],
    "memory": ["retrieval", "kv_recall", "first_token", "selective_copy"],
    "long_range": ["long_copy", "distant_match", "multihop"],
    "reasoning": ["stack", "induction", "pattern"],
    "arithmetic": ["counting", "parity", "addition", "modular"],
    "pattern": ["bigram", "majority"],
    "robustness": ["noisy_copy", "compression"],
    "other": ["histogram"],
}

# Attention-only tasks (truly test attention, not just FFN)
ATTENTION_ONLY_TASKS = [
    "copy", "retrieval", "kv_recall", "first_token", "selective_copy",
    "long_copy", "distant_match", "multihop", "induction",
]


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, output_type: str) -> torch.Tensor:
    """Compute loss based on task type."""
    if output_type == "sequence":
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    elif output_type in ("last", "position"):
        # Only compute loss on specific positions
        if output_type == "last":
            return F.cross_entropy(logits[:, -1, :], targets[:, -1])
        else:
            # Position-specific - find non-zero targets
            mask = targets != 0
            if mask.any():
                return F.cross_entropy(logits[mask], targets[mask])
            return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, output_type: str) -> float:
    """Compute accuracy based on task type."""
    preds = logits.argmax(dim=-1)
    
    if output_type == "sequence":
        return (preds == targets).float().mean().item()
    elif output_type == "last":
        return (preds[:, -1] == targets[:, -1]).float().mean().item()
    elif output_type == "position":
        mask = targets != 0
        if mask.any():
            return (preds[mask] == targets[mask]).float().mean().item()
        return 0.0
    return (preds == targets).float().mean().item()


def train_task(
    cfg: TaskConfig,
    attention_name: str,
    task_name: str,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a model on a single task and return metrics."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    task = TASKS[task_name]
    generate_fn = task["fn"]
    vocab_size = task["vocab_size"]
    seq_len = task["seq_len"]
    
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
    
    for epoch in range(cfg.num_epochs):
        model.train()
        x, y, output_type = generate_fn(cfg.batch_size, seq_len, vocab_size, cfg.device)
        
        logits = model(x)
        loss = compute_loss(logits, y, output_type)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Eval
        if (epoch + 1) % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_eval, y_eval, output_type = generate_fn(cfg.batch_size * 2, seq_len, vocab_size, cfg.device)
                logits_eval = model(x_eval)
                acc = compute_accuracy(logits_eval, y_eval, output_type)
                accs.append(acc)
                
                if verbose:
                    print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.3f}")
    
    train_time = time.time() - start_time
    
    # Final eval
    model.eval()
    with torch.no_grad():
        x_eval, y_eval, output_type = generate_fn(cfg.batch_size * 4, seq_len, vocab_size, cfg.device)
        logits_eval = model(x_eval)
        final_acc = compute_accuracy(logits_eval, y_eval, output_type)
        final_loss = compute_loss(logits_eval, y_eval, output_type).item()
    
    return {
        "final_loss": final_loss,
        "final_acc": final_acc,
        "train_time": train_time,
        "loss_curve": losses[::10],
        "acc_curve": accs,
    }


def run_task_benchmarks(
    cfg: TaskConfig,
    attention_names: List[str],
    task_names: List[str],
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run all task benchmarks with multiple seeds.
    
    Returns:
        results[task_name][attention_name] = {
            "mean_acc": float,
            "std_acc": float,
            "mean_loss": float,
            "mean_time": float,
            "per_seed": [...]
        }
    """
    results = {}
    
    for task_name in task_names:
        results[task_name] = {}
        task_info = TASKS[task_name]
        
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({task_info['category']})")
        print(f"  seq_len={task_info['seq_len']}, vocab={task_info['vocab_size']}")
        print(f"{'='*60}")
        
        for attn_name in attention_names:
            print(f"  {attn_name}...", end=" ", flush=True)
            
            seed_results = []
            
            try:
                for seed in range(cfg.num_seeds):
                    metrics = train_task(
                        cfg, attn_name, task_name, 
                        seed=cfg.seed + seed,
                        verbose=False
                    )
                    seed_results.append(metrics)
                
                accs = [r["final_acc"] for r in seed_results]
                losses = [r["final_loss"] for r in seed_results]
                times = [r["train_time"] for r in seed_results]
                
                results[task_name][attn_name] = {
                    "mean_acc": np.mean(accs),
                    "std_acc": np.std(accs),
                    "mean_loss": np.mean(losses),
                    "std_loss": np.std(losses),
                    "mean_time": np.mean(times),
                    "per_seed": seed_results,
                }
                
                print(f"OK acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}")
                
            except Exception as e:
                results[task_name][attn_name] = {
                    "mean_acc": 0.0,
                    "std_acc": 0.0,
                    "mean_loss": float("nan"),
                    "mean_time": float("nan"),
                    "error": str(e),
                }
                print(f"ERROR: {e}")
    
    return results


def format_latex_table(results: Dict[str, Dict[str, Dict[str, Any]]], category: str = None) -> str:
    """Format results as LaTeX table."""
    
    task_names = list(results.keys())
    if category:
        task_names = [t for t in task_names if TASKS[t]["category"] == category]
    
    if not task_names:
        return ""
    
    attn_names = list(results[task_names[0]].keys())
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    cat_str = f" ({category})" if category else ""
    lines.append(f"\\caption{{Synthetic task performance{cat_str}. Accuracy (mean $\\pm$ std over 3 seeds).}}")
    lines.append(f"\\label{{tab:task-{category or 'all'}}}")
    
    col_spec = "l" + "c" * len(task_names)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header
    header = "Method"
    for task in task_names:
        task_display = task.replace("_", " ").title()
        header += f" & {task_display}"
    header += r"\\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    for attn in attn_names:
        row = attn.replace("_", "\\_").replace("-", "-")
        for task in task_names:
            r = results[task].get(attn, {})
            acc = r.get("mean_acc", 0)
            std = r.get("std_acc", 0)
            if "error" in r:
                row += " & --"
            else:
                row += f" & {acc:.2f}$\\pm${std:.2f}"
        row += r"\\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def format_markdown_table(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format results as Markdown table."""
    
    task_names = list(results.keys())
    attn_names = list(results[task_names[0]].keys())
    
    lines = []
    
    # Header
    header = "| Method |"
    sep = "|---|"
    for task in task_names:
        header += f" {task} |"
        sep += "---:|"
    lines.append(header)
    lines.append(sep)
    
    # Data rows
    for attn in attn_names:
        row = f"| {attn} |"
        for task in task_names:
            r = results[task].get(attn, {})
            acc = r.get("mean_acc", 0)
            std = r.get("std_acc", 0)
            if "error" in r:
                row += " -- |"
            else:
                row += f" {acc:.2f}+/-{std:.2f} |"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive synthetic task benchmarks")
    parser.add_argument(
    "--device",
    default=None,
    choices=["cpu", "cuda", "mps"],
    help="Device to use: cpu, cuda, or mps (Apple GPU)"
    )
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer epochs, 1 seed)")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tasks", nargs="*", default=None, help="Tasks to run")
    parser.add_argument("--category", default=None, choices=list(CATEGORIES.keys()), help="Task category")
    parser.add_argument("--attention-only", action="store_true", help="Only run attention-testing tasks")
    parser.add_argument("--attentions", nargs="*", default=None, help="Attention types")
    args = parser.parse_args()
    
    # Configure
    device = get_best_device(args.device)

    cfg = TaskConfig(
        device=device,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_seeds=args.num_seeds,
        lr=args.lr,
    )
    
    if args.quick:
        cfg.num_epochs = 50
        cfg.num_seeds = 1
        cfg.eval_every = 10
        cfg.batch_size = 16
    
    # MPS safety adjustments
    if cfg.device.type == "mps":
        if cfg.batch_size > 16:
            print(f"⚠️  Reducing batch size for MPS: {cfg.batch_size} → 16")
            cfg.batch_size = 16

    # Select tasks
    if args.tasks:
        task_names = args.tasks
    elif args.category:
        task_names = CATEGORIES[args.category]
    elif args.attention_only:
        task_names = ATTENTION_ONLY_TASKS
    else:
        task_names = list(TASKS.keys())
    
    # Select attentions
    attention_names = args.attentions if args.attentions else DEFAULT_ATTENTIONS
    
    print("=" * 70)
    print("SLAY Comprehensive Task Benchmarks")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available:  {torch.backends.mps.is_available()}")
    print(f"Model: embed={cfg.embed_dim}, heads={cfg.n_heads}, layers={cfg.n_layers}")
    print(f"Training: epochs={cfg.num_epochs}, seeds={cfg.num_seeds}")
    print(f"Tasks ({len(task_names)}): {', '.join(task_names)}")
    print(f"Attentions ({len(attention_names)}): {', '.join(attention_names)}")
    print("=" * 70)
    
    # Run benchmarks
    results = run_task_benchmarks(cfg, attention_names, task_names)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(format_markdown_table(results))
    
    # Save results
    os.makedirs("tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    # LaTeX - one table per category
    for cat in CATEGORIES.keys():
        cat_tasks = [t for t in task_names if TASKS[t]["category"] == cat]
        if cat_tasks:
            latex = format_latex_table(results, category=cat)
            with open(f"tables/task_{cat}.tex", "w") as f:
                f.write(latex)
            print(f"Wrote tables/task_{cat}.tex")
    
    # Full LaTeX table
    latex_all = format_latex_table(results)
    with open("tables/task_performance.tex", "w") as f:
        f.write(latex_all)
    print("Wrote tables/task_performance.tex")
    
    # Markdown
    md = format_markdown_table(results)
    with open("artifacts/task_performance.md", "w") as f:
        f.write(f"# Synthetic Task Performance\n\n{md}\n")
    print("Wrote artifacts/task_performance.md")
    
    # JSON (without per-seed details for size)
    json_results = {}
    for task, attn_results in results.items():
        json_results[task] = {}
        for attn, metrics in attn_results.items():
            json_results[task][attn] = {
                k: v for k, v in metrics.items() if k != "per_seed"
            }
    
    with open("artifacts/task_results.json", "w") as f:
        json.dump({"config": asdict(cfg), "results": json_results}, f, indent=2)
    print("Wrote artifacts/task_results.json")
    
    # Print best performers per task
    print("\n" + "=" * 70)
    print("BEST PERFORMERS")
    print("=" * 70)
    for task in task_names:
        best_attn = max(results[task].keys(), key=lambda a: results[task][a].get("mean_acc", 0))
        acc = results[task][best_attn]["mean_acc"]
        print(f"  {task}: {best_attn} ({acc:.3f})")


if __name__ == "__main__":
    main()
