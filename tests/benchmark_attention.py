"""
Standalone Benchmark for Attention Mechanisms

Benchmarks speed, memory, and task performance of all attention implementations.
Ported from synth_benchmark.py and adapted for src.attention package.

Run with: python tests/benchmark_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import (
    ATTENTION_CLASSES,
    get_attention_class,
    list_attention_types,
)


# ============================================================================
# Benchmark Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    embed_dim: int = 64
    n_heads: int = 4
    batch_size: int = 16
    warmup_iters: int = 5
    bench_iters: int = 20
    seq_lengths: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True
    save_plots: bool = True

    # Task-benchmark knobs
    task_steps: int = 100
    long_retrieval_seq_len: int = 256

    # Yat / Spherical-Performer knobs (kept small for benchmarks)
    yat_num_features: int = 32
    yat_num_quadrature_nodes: int = 2
    yat_chunk_size: int = 512
    yat_poly_dim: int = 64
    yat_prf_dim: int = 8
    yat_poly_sketch_dim: int = 64
    yat_spherical_epsilon: float = 1e-2


# ============================================================================
# Models
# ============================================================================

class SimpleModel(nn.Module):
    """Simple Transformer model for synthetic tasks."""
    def __init__(self, attn_class, vocab_size, embed_dim, n_heads, n_layers=1, attn_kwargs=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        attn_kwargs = attn_kwargs or {}
        self.layers = nn.ModuleList([
            attn_class(embed_dim, n_heads, **attn_kwargs) for _ in range(n_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ============================================================================
# 1. Micro-Benchmarks (Speed & Memory)
# ============================================================================

def benchmark_forward(attn: nn.Module, x: torch.Tensor, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        with torch.no_grad(): _ = attn(x)
    if x.device.type == "cuda": torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad(): _ = attn(x)
    if x.device.type == "cuda": torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iters * 1000  # ms


def iter_attention_types(selected: List[str] | None = None) -> List[str]:
    names = list_attention_types()
    if not selected:
        return names
    missing = [n for n in selected if n not in ATTENTION_CLASSES]
    if missing:
        raise ValueError(f"Unknown attention types: {missing}. Available: {list_attention_types()}")
    return selected


def attention_kwargs(name: str, config: BenchmarkConfig) -> Dict[str, Any]:
    """Per-attention constructor kwargs for fair/default benchmarking."""
    if name in {"yat-performer", "yat-performer-hadamard"}:
        return {
            "num_features": config.yat_num_features,
            "num_quadrature_nodes": config.yat_num_quadrature_nodes,
        }
    if name in {"yat-performer-laplace", "yat-performer-laplace-only"}:
        return {
            "num_features": config.yat_num_features,
            "num_quadrature_nodes": config.yat_num_quadrature_nodes,
            "chunk_size": config.yat_chunk_size,
        }
    if name in {"yat-performer-anchors", "yat-performer-anchor"}:
        return {
            "num_prf_features": config.yat_prf_dim,
            "num_quadrature_nodes": config.yat_num_quadrature_nodes,
            "poly_dim": config.yat_poly_dim,
            "chunk_size": config.yat_chunk_size,
        }
    if name == "yat-performer-nystrom":
        return {
            "num_prf_features": config.yat_prf_dim,
            "num_quadrature_nodes": config.yat_num_quadrature_nodes,
            "poly_dim": config.yat_poly_dim,
            "chunk_size": config.yat_chunk_size,
        }
    if name in {"yat-performer-rm", "yat-performer-random-maclaurin"}:
        return {
            "num_prf_features": config.yat_prf_dim,
            "num_quadrature_nodes": config.yat_num_quadrature_nodes,
            "poly_dim": config.yat_poly_dim,
            "chunk_size": config.yat_chunk_size,
        }
    if name in {"yat-performer-tensor", "yat-performer-tensorsketch"}:
        return {
            "num_prf_features": config.yat_prf_dim,
            "num_quadrature_nodes": config.yat_num_quadrature_nodes,
            "poly_sketch_dim": config.yat_poly_sketch_dim,
            "chunk_size": config.yat_chunk_size,
        }
    if name in {"yat-spherical", "yat-exact-spherical"}:
        return {"epsilon": config.yat_spherical_epsilon}
    return {}


def run_speed_benchmark(config: BenchmarkConfig, selected: List[str] | None = None) -> Dict[str, Dict[int, float]]:
    print(f"\n{'='*70}\nSpeed Benchmark (device={config.device})\n{'='*70}")
    print(f"embed_dim={config.embed_dim}, n_heads={config.n_heads}, batch_size={config.batch_size}")
    
    results = {}
    header = f"{'Attention':<20}" + "".join(f"{T:>10}" for T in config.seq_lengths)
    print(header + "\n" + "-" * len(header))
    
    for name in iter_attention_types(selected):
        attn_class = get_attention_class(name)
        results[name] = {}
        row = f"{name:<20}"
        
        for seq_len in config.seq_lengths:
            try:
                attn = attn_class(config.embed_dim, config.n_heads, **attention_kwargs(name, config)).to(config.device)
                x = torch.randn(config.batch_size, seq_len, config.embed_dim, device=config.device)
                ms = benchmark_forward(attn, x, config.warmup_iters, config.bench_iters)
                results[name][seq_len] = ms
                row += f"{ms:>9.2f}ms"
                del attn, x
                if config.device == "cuda": torch.cuda.empty_cache()
            except Exception:
                results[name][seq_len] = float('nan')
                row += f"{'ERROR':>10}"
        print(row)
    return results


def run_memory_benchmark(config: BenchmarkConfig, selected: List[str] | None = None) -> Dict[str, Dict[int, float]]:
    if config.device != "cuda": return {}
    print(f"\n{'='*70}\nMemory Benchmark (device={config.device})\n{'='*70}")
    
    results = {}
    header = f"{'Attention':<20}" + "".join(f"{T:>10}" for T in config.seq_lengths)
    print(header + "\n" + "-" * len(header))
    
    for name in iter_attention_types(selected):
        attn_class = get_attention_class(name)
        results[name] = {}
        row = f"{name:<20}"
        
        for seq_len in config.seq_lengths:
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                attn = attn_class(config.embed_dim, config.n_heads, **attention_kwargs(name, config)).to(config.device)
                x = torch.randn(config.batch_size, seq_len, config.embed_dim, device=config.device)
                with torch.no_grad(): _ = attn(x)
                mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                results[name][seq_len] = mb
                row += f"{mb:>9.1f}MB"
                del attn, x
            except Exception:
                results[name][seq_len] = float('nan')
                row += f"{'OOM':>10}"
        print(row)
    return results


# ============================================================================
# 2. Approximation Quality
# ============================================================================

def run_approximation_quality(config: BenchmarkConfig):
    print(f"\n{'='*70}\nApproximation Quality (MSE vs Standard)\n{'='*70}")
    
    # Using Standard as ground truth
    try:
        Standard = get_attention_class('standard')
    except:
        print("Standard attention not found, skipping.")
        return

    seq_lens = [64, 128, 256]
    results = {}
    
    for name in list_attention_types():
        if name == 'standard': continue
        attn_class = get_attention_class(name)
        results[name] = {'mse': [], 'cosine': []}
        
        print(f"\n--- {name} ---")
        for T in seq_lens:
            try:
                # Same initialization for fair comparison? 
                # Diff hard due to different internal params (projections vs weights)
                # We compare outputs given same weights if possible, but structure differs.
                # So we just train them? No, this is kernel approximation test.
                # For kernel approximation, we should really test the kernels themselves.
                # But here we test the full attention module output similarity "out of the box"
                # which might be high variance due to random init.
                # BETTER: For approximations like Performer/RFF, we check if they approximate Softmax
                # BUT: They have random projections.
                
                # We'll just run them and see if they produce finite outputs and look reasonable.
                # For strictly approximation quality we need to copy weights which is hard.
                # So we will skip the strict MSE against Standard for now unless they share weights.
                pass
            except:
                pass
    
    # Re-implementing the specific kernel approximation test from synth_benchmark
    # This requires access to internal features which is hard with the unified API.
    # Instead, we will rely on the Task Performance benchmarks.
    print("Skipping direct MSE comparison (requires weight tying). Relying on Task Benchmarks.")


def run_kernel_approximation_test():
    """Specific test for RFF/Performer approximating Gaussian/Softmax kernels."""
    pass # Todo if needed


# ============================================================================
# 3. Task Benchmarks
# ============================================================================

def train_task(config, task_name, get_batch_fn, vocab_size, seq_len, num_steps: int, selected: List[str] | None = None):
    print(f"\n--- {task_name} ---")
    results = {}
    
    for name in iter_attention_types(selected):
        attn_class = get_attention_class(name)
        try:
            # Increase model capacity for harder tasks
            n_layers = 2 if task_name == 'Sorting' else 1
            model = SimpleModel(
                attn_class,
                vocab_size,
                config.embed_dim,
                config.n_heads,
                n_layers=n_layers,
                attn_kwargs=attention_kwargs(name, config),
            ).to(config.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            losses = []
            
            start = time.time()
            for step in range(num_steps):
                x, y = get_batch_fn(config.batch_size, seq_len, vocab_size, config.device)
                
                logits = model(x)
                if 'Retrieval' in task_name:
                    # Last token prediction
                    # Check if y is (B,) or (B, T) - for retrieval it should be (B,)
                    if y.dim() == 1:
                        loss = F.cross_entropy(logits[:, -1, :], y)
                    else:
                        # Fallback if y is sequence
                        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                else:
                    # Sequence prediction
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            duration = (time.time() - start) * 1000 / num_steps
            final_loss = np.mean(losses[-10:])
            
            results[name] = {'loss': final_loss, 'time': duration}
            print(f"{name:<20} Loss: {final_loss:.4f} | Time: {duration:.1f}ms/step")
            
            del model, optimizer
            if config.device == "cuda": torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{name:<20} ERROR: {str(e)[:50]}...")
            results[name] = {'loss': float('nan'), 'time': float('nan')}
            
    return results


def run_copy_task(config, selected: List[str] | None = None):
    """Experiment 1: Copy Input Sequence"""
    def get_batch(B, T, V, device):
        x = torch.randint(0, V, (B, T), device=device)
        return x, x
    
    return train_task(config, "Copy Task", get_batch, vocab_size=16, seq_len=32, num_steps=config.task_steps, selected=selected)


def run_sorting_task(config, selected: List[str] | None = None):
    """Experiment 2: Sort Input Sequence"""
    def get_batch(B, T, V, device):
        x = torch.randint(0, V, (B, T), device=device)
        y, _ = torch.sort(x, dim=1)
        return x, y
    
    return train_task(config, "Sorting Task", get_batch, vocab_size=16, seq_len=16, num_steps=config.task_steps, selected=selected)


def run_associative_recall(config, selected: List[str] | None = None):
    """Experiment 3: Associative Recall (Key-Value pairs)"""
    def get_batch(B, T, V, device):
        # Format: k1 v1 k2 v2 ... k_query
        # Target: v_query
        # Actually simplified: Copy but with mapping logic
        # Synth benchmark uses strict KV pairs.
        
        # We'll implement a simplified version:
        # x: [k1, v1, k2, v2, ..., k_query]
        # y: [..., v_query]
        
        num_pairs = (T - 1) // 2
        batch_x = []
        batch_y = []
        
        for _ in range(B):
            keys = torch.randperm(V - 1)[:num_pairs] + 1 # avoid 0
            vals = torch.randint(1, V, (num_pairs,))
            
            seq = torch.zeros(T, dtype=torch.long)
            seq[0:num_pairs*2:2] = keys
            seq[1:num_pairs*2:2] = vals
            
            query_idx = torch.randint(0, num_pairs, (1,)).item()
            query_key = keys[query_idx]
            target_val = vals[query_idx]
            
            seq[-1] = query_key
            batch_x.append(seq)
            batch_y.append(target_val)
            
        x = torch.stack(batch_x).to(device)
        y = torch.as_tensor(batch_y, device=device)
        
        return x, y # Note: this requires Retrieval mode (last token)

    # Note: Train function handles 'Retrieval' task name specifically
    task_results = train_task(config, "Retrieval", get_batch, vocab_size=20, seq_len=11, num_steps=config.task_steps, selected=selected) # 5 pairs + 1 query
    return task_results


def run_long_retrieval(config, selected: List[str] | None = None):
    """Experiment 7: Needle in Haystack"""
    def get_batch(B, T, V, device):
        needle_id = V - 1
        x = torch.randint(0, V-1, (B, T), device=device)
        
        # Insert needle at random positions
        pos = torch.randint(0, T-1, (B,)) # keep last for query
        for i in range(B):
            x[i, pos[i]] = needle_id
            
        # Last token is query (can be special token or just implicit)
        # For this task, we want to detect if needle was present or count it?
        # Original benchmark: Recall the needle value? Or position?
        # Let's say: Last token is NEEDLE_QUERY, target is NEEDLE_ID (if we used values)
        # Simplified: Just output NEEDLE_ID at last step
        
        y = torch.full((B,), needle_id, dtype=torch.long, device=device)
        return x, y

    T = config.long_retrieval_seq_len
    return train_task(config, f"Retrieval (T={T})", get_batch, vocab_size=50, seq_len=T, num_steps=config.task_steps, selected=selected)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run fast benchmarks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--attentions",
        nargs="*",
        default=None,
        help="Optional list of attention types to run (default: all).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is False. "
            "This environment appears to be using a CPU-only PyTorch build. "
            "Install a CUDA-enabled PyTorch wheel and ensure an NVIDIA GPU + driver are available."
        )
    
    config = BenchmarkConfig(device=args.device)
    if args.quick:
        config.seq_lengths = (64, 128, 256)
        config.bench_iters = 5
        config.warmup_iters = 2
        config.task_steps = 20
        config.long_retrieval_seq_len = 64

    # CPU runs can be extremely slow for some attention variants.
    if config.device == "cpu":
        config.task_steps = min(config.task_steps, 10)
        config.long_retrieval_seq_len = min(config.long_retrieval_seq_len, 64)
    
    print(f"Running Benchmarks on {config.device}...")

    selected = args.attentions
    
    # 1. Micro-Benchmarks
    run_speed_benchmark(config, selected=selected)
    run_memory_benchmark(config, selected=selected)
    
    # 2. Task Benchmarks
    print(f"\n{'='*70}\nTask Benchmarks\n{'='*70}")
    run_copy_task(config, selected=selected)
    run_sorting_task(config, selected=selected)
    run_associative_recall(config, selected=selected)
    run_long_retrieval(config, selected=selected)
    
    print("\nBenchmark Complete!")


if __name__ == "__main__":
    main()
