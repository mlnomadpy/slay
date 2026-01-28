"""
Comprehensive benchmark for all attention mechanisms.

Tests:
1. Speed (forward pass time)
2. Memory (peak GPU memory)
3. Approximation quality (cosine similarity to standard softmax)
4. Gradient flow (backward pass)

Run: python benchmark.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys

# Import all attention mechanisms
from standard import StandardCausalAttention
from linear import LinearCausalAttention
from cosformer import CosformerCausalAttention
from performer import FastAttention
from rff import RFFCausalAttention
from yat import YatCausalAttention
from yat_spherical import YatSphericalCausalAttention
from yat_performer import YatPerformerCausalAttention
from yat_performer_laplace import YatPerformerLaplaceCausalAttention
from yat_performer_tensor import YatPerformerTensorCausalAttention
from yat_performer_poly_alt import YatPerformerAnchorCausalAttention


# ============================================================================
# Attention Registry
# ============================================================================

def create_attention_modules(embed_dim: int, n_heads: int, device: torch.device):
    """Create all attention modules with the same architecture."""
    modules = {}
    
    # Standard (baseline)
    modules["Standard"] = StandardCausalAttention(embed_dim, n_heads).to(device)
    
    # Linear attention family
    modules["Linear"] = LinearCausalAttention(embed_dim, n_heads).to(device)
    modules["Cosformer"] = CosformerCausalAttention(embed_dim, n_heads).to(device)
    modules["Performer"] = FastAttention(embed_dim, n_heads, kernel_size=64).to(device)
    modules["RFF"] = RFFCausalAttention(embed_dim, n_heads, num_features=64).to(device)
    
    # Yat attention family
    modules["Yat"] = YatCausalAttention(embed_dim, n_heads).to(device)
    modules["YatSpherical"] = YatSphericalCausalAttention(embed_dim, n_heads).to(device)
    
    # YatPerformer variants
    modules["YatPerformer"] = YatPerformerCausalAttention(
        embed_dim, n_heads, num_features=32
    ).to(device)
    modules["YatPerformerLaplace"] = YatPerformerLaplaceCausalAttention(
        embed_dim, n_heads, num_features=32
    ).to(device)
    modules["YatPerformerTensor"] = YatPerformerTensorCausalAttention(
        embed_dim, n_heads, num_prf_features=8, poly_sketch_dim=64
    ).to(device)
    modules["YatPerformerAnchor"] = YatPerformerAnchorCausalAttention(
        embed_dim, n_heads, num_features=32
    ).to(device)
    
    return modules


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_speed(
    modules: Dict[str, nn.Module],
    batch_size: int,
    seq_lengths: List[int],
    embed_dim: int,
    device: torch.device,
    warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, Dict[int, float]]:
    """Benchmark forward pass speed for all modules."""
    results = {name: {} for name in modules}
    
    for T in seq_lengths:
        x = torch.randn(batch_size, T, embed_dim, device=device)
        
        for name, module in modules.items():
            module.eval()
            
            try:
                # Warmup
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = module(x)
                torch.cuda.synchronize()
                
                # Timed runs
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = module(x)
                end.record()
                torch.cuda.synchronize()
                
                results[name][T] = start.elapsed_time(end) / num_runs
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[name][T] = float('inf')
                    torch.cuda.empty_cache()
                else:
                    raise
    
    return results


def benchmark_memory(
    modules: Dict[str, nn.Module],
    batch_size: int,
    seq_lengths: List[int],
    embed_dim: int,
    device: torch.device,
) -> Dict[str, Dict[int, float]]:
    """Benchmark peak GPU memory usage."""
    results = {name: {} for name in modules}
    
    for T in seq_lengths:
        for name, module in modules.items():
            module.eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            x = torch.randn(batch_size, T, embed_dim, device=device)
            
            try:
                with torch.no_grad():
                    _ = module(x)
                
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                results[name][T] = peak_mem
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[name][T] = float('inf')
                    torch.cuda.empty_cache()
                else:
                    raise
            
            del x
            torch.cuda.empty_cache()
    
    return results


def benchmark_approximation(
    modules: Dict[str, nn.Module],
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Measure approximation quality vs standard softmax attention."""
    results = {}
    
    # Get standard output as reference
    standard = modules["Standard"]
    standard.eval()
    
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    with torch.no_grad():
        out_standard = standard(x).float()
    
    for name, module in modules.items():
        if name == "Standard":
            continue
            
        module.eval()
        
        try:
            with torch.no_grad():
                out = module(x).float()
            
            # Compute metrics
            diff = out - out_standard
            rel_l2 = diff.norm() / out_standard.norm()
            cosine = F.cosine_similarity(out.view(-1), out_standard.view(-1), dim=0)
            mse = (diff ** 2).mean()
            
            results[name] = {
                "rel_l2": rel_l2.item(),
                "cosine": cosine.item(),
                "mse": mse.item(),
            }
            
        except RuntimeError:
            results[name] = {
                "rel_l2": float('inf'),
                "cosine": 0.0,
                "mse": float('inf'),
            }
            torch.cuda.empty_cache()
    
    return results


def benchmark_gradient(
    modules: Dict[str, nn.Module],
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    device: torch.device,
) -> Dict[str, bool]:
    """Test if gradients flow correctly."""
    results = {}
    
    for name, module in modules.items():
        module.train()
        
        try:
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
            module.zero_grad()
            
            out = module(x)
            loss = out.sum()
            loss.backward()
            
            has_grad = x.grad is not None
            grad_finite = has_grad and torch.isfinite(x.grad).all().item()
            results[name] = has_grad and grad_finite
            
        except Exception:
            results[name] = False
            torch.cuda.empty_cache()
    
    return results


# ============================================================================
# Printing Utilities
# ============================================================================

def print_speed_results(results: Dict[str, Dict[int, float]], seq_lengths: List[int]):
    """Print speed benchmark results."""
    print("\n" + "="*100)
    print("SPEED BENCHMARK (ms per forward pass)")
    print("="*100)
    
    # Header
    header = f"{'Attention':<25}"
    for T in seq_lengths:
        header += f"{'T='+str(T):<12}"
    print(header)
    print("-"*100)
    
    # Data
    for name, times in results.items():
        row = f"{name:<25}"
        for T in seq_lengths:
            t = times.get(T, float('inf'))
            if t == float('inf'):
                row += f"{'OOM':<12}"
            else:
                row += f"{t:<12.2f}"
        print(row)


def print_memory_results(results: Dict[str, Dict[int, float]], seq_lengths: List[int]):
    """Print memory benchmark results."""
    print("\n" + "="*100)
    print("MEMORY BENCHMARK (MB)")
    print("="*100)
    
    # Header
    header = f"{'Attention':<25}"
    for T in seq_lengths:
        header += f"{'T='+str(T):<12}"
    print(header)
    print("-"*100)
    
    # Data
    for name, mems in results.items():
        row = f"{name:<25}"
        for T in seq_lengths:
            m = mems.get(T, float('inf'))
            if m == float('inf'):
                row += f"{'OOM':<12}"
            else:
                row += f"{m:<12.1f}"
        print(row)


def print_approximation_results(results: Dict[str, Dict[str, float]]):
    """Print approximation quality results."""
    print("\n" + "="*80)
    print("APPROXIMATION QUALITY (vs Standard Softmax)")
    print("="*80)
    
    print(f"{'Attention':<25} {'Cosine↑':<12} {'Rel L2↓':<12} {'MSE↓':<12}")
    print("-"*65)
    
    # Sort by cosine similarity
    sorted_results = sorted(results.items(), key=lambda x: x[1]["cosine"], reverse=True)
    
    for name, metrics in sorted_results:
        cosine = metrics["cosine"]
        rel_l2 = metrics["rel_l2"]
        mse = metrics["mse"]
        
        if cosine == 0.0:
            print(f"{name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        else:
            print(f"{name:<25} {cosine:<12.4f} {rel_l2:<12.4f} {mse:<12.2e}")


def print_gradient_results(results: Dict[str, bool]):
    """Print gradient flow results."""
    print("\n" + "="*60)
    print("GRADIENT FLOW TEST")
    print("="*60)
    
    print(f"{'Attention':<25} {'Status':<15}")
    print("-"*40)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<25} {status:<15}")


def print_summary(
    speed_results: Dict[str, Dict[int, float]],
    memory_results: Dict[str, Dict[int, float]],
    approx_results: Dict[str, Dict[str, float]],
    grad_results: Dict[str, bool],
    ref_seq_len: int,
):
    """Print summary comparison table."""
    print("\n" + "="*100)
    print(f"SUMMARY (T={ref_seq_len})")
    print("="*100)
    
    print(f"{'Attention':<25} {'Speed(ms)':<12} {'Memory(MB)':<12} {'Cosine':<10} {'Gradient':<10} {'Complexity':<12}")
    print("-"*100)
    
    # Determine complexity class
    linear_attns = {"Linear", "Cosformer", "Performer", "RFF", 
                    "YatPerformer", "YatPerformerLaplace", "YatPerformerTensor", "YatPerformerAnchor"}
    
    for name in speed_results.keys():
        speed = speed_results[name].get(ref_seq_len, float('inf'))
        memory = memory_results[name].get(ref_seq_len, float('inf'))
        
        if name == "Standard":
            cosine = 1.0
        else:
            cosine = approx_results.get(name, {}).get("cosine", 0.0)
        
        grad = "✓" if grad_results.get(name, False) else "✗"
        complexity = "O(T)" if name in linear_attns else "O(T²)"
        
        speed_str = f"{speed:.2f}" if speed != float('inf') else "OOM"
        memory_str = f"{memory:.1f}" if memory != float('inf') else "OOM"
        
        print(f"{name:<25} {speed_str:<12} {memory_str:<12} {cosine:<10.4f} {grad:<10} {complexity:<12}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*100)
    print(" COMPREHENSIVE ATTENTION MECHANISM BENCHMARK")
    print("="*100)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)
    
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Configuration
    embed_dim = 256
    n_heads = 4
    batch_size = 1
    seq_lengths = [512, 1024, 2048, 4096]
    approx_seq_len = 256  # Shorter for approximation test
    
    print(f"\nConfiguration:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  n_heads: {n_heads}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_lengths: {seq_lengths}")
    
    # Create modules
    print("\nCreating attention modules...")
    modules = create_attention_modules(embed_dim, n_heads, device)
    print(f"  Created {len(modules)} modules")
    
    # Run benchmarks
    print("\n--- Running Speed Benchmark ---")
    speed_results = benchmark_speed(modules, batch_size, seq_lengths, embed_dim, device)
    print_speed_results(speed_results, seq_lengths)
    
    print("\n--- Running Memory Benchmark ---")
    memory_results = benchmark_memory(modules, batch_size, seq_lengths, embed_dim, device)
    print_memory_results(memory_results, seq_lengths)
    
    print("\n--- Running Approximation Benchmark ---")
    approx_results = benchmark_approximation(modules, batch_size, approx_seq_len, embed_dim, device)
    print_approximation_results(approx_results)
    
    print("\n--- Running Gradient Benchmark ---")
    grad_results = benchmark_gradient(modules, batch_size, 64, embed_dim, device)
    print_gradient_results(grad_results)
    
    # Summary
    print_summary(speed_results, memory_results, approx_results, grad_results, 1024)
    
    print("\n" + "="*100)
    print(" BENCHMARK COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
