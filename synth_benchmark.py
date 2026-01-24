"""
COMPLETE PERFORMER VALIDATION SUITE
All experiments and debugging tools in one script

Usage:
    python performer_complete.py --mode debug    # Run debugging tests
    python performer_complete.py --mode quick    # Run quick validation
    python performer_complete.py --mode full     # Run full experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import math  # Added for attention implementations

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# BENCHMARK CONFIGURATION AND RESULT CLASSES
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    num_seeds: int = 3                    # Number of random seeds for statistical significance
    num_epochs: int = 100                 # Training epochs per task
    batch_size: int = 32                  # Batch size
    seq_len: int = 64                     # Default sequence length
    embed_dim: int = 64                   # Embedding dimension
    vocab_size: int = 10                  # Vocabulary size
    learning_rate: float = 0.001          # Learning rate
    weight_decay: float = 0.01            # Weight decay for AdamW
    dropout: float = 0.1                  # Dropout rate
    num_random_features: int = 64         # Random features for Performer/Yat-Performer
    num_quadrature_nodes: int = 4         # Quadrature nodes for Yat-Performer
    use_positional_encoding: bool = True  # Use sinusoidal positional encoding
    use_layer_norm: bool = True           # Use layer normalization
    use_ffn: bool = False                 # Use feedforward network
    warmup_epochs: int = 5                # Learning rate warmup
    early_stopping_patience: int = 20     # Early stopping patience
    eval_every: int = 10                  # Evaluate every N epochs
    log_file: str = "benchmark_results.txt"


@dataclass  
class TaskResult:
    """Result of a single task evaluation"""
    task_name: str
    attention_name: str
    accuracies: List[float] = field(default_factory=list)  # Per-seed final accuracies
    losses: List[float] = field(default_factory=list)      # Per-seed final losses
    train_times: List[float] = field(default_factory=list) # Training time per seed
    
    @property
    def mean_accuracy(self) -> float:
        return np.mean(self.accuracies) if self.accuracies else 0.0
    
    @property
    def std_accuracy(self) -> float:
        return np.std(self.accuracies) if len(self.accuracies) > 1 else 0.0
    
    @property
    def mean_loss(self) -> float:
        return np.mean(self.losses) if self.losses else 0.0
    
    @property
    def std_loss(self) -> float:
        return np.std(self.losses) if len(self.losses) > 1 else 0.0
    
    def __str__(self) -> str:
        return f"{self.mean_accuracy:.4f} ± {self.std_accuracy:.4f}"


@dataclass
class BenchmarkResults:
    """Collection of all benchmark results"""
    config: BenchmarkConfig
    task_results: Dict[str, Dict[str, TaskResult]] = field(default_factory=dict)  # task -> attention -> result
    speed_results: Dict[str, Dict[int, float]] = field(default_factory=dict)       # attention -> seq_len -> time
    memory_results: Dict[str, Dict[int, float]] = field(default_factory=dict)      # attention -> seq_len -> memory
    
    def add_result(self, result: TaskResult):
        if result.task_name not in self.task_results:
            self.task_results[result.task_name] = {}
        self.task_results[result.task_name][result.attention_name] = result
    
    def get_summary_table(self) -> str:
        """Generate summary table as string"""
        lines = []
        tasks = list(self.task_results.keys())
        if not tasks:
            return "No results yet"
        
        attentions = list(self.task_results[tasks[0]].keys())
        
        # Header
        header = f"{'Attention':<25}" + "".join(f"{t:<15}" for t in tasks)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for attn in attentions:
            row = f"{attn:<25}"
            for task in tasks:
                result = self.task_results[task].get(attn)
                if result:
                    row += f"{result.mean_accuracy:.3f}±{result.std_accuracy:.2f}  "
                else:
                    row += f"{'N/A':<15}"
            lines.append(row)
        
        return "\n".join(lines)

# ============================================================================
# CAUSAL ATTENTION IMPLEMENTATIONS (Updated - Multi-head with causal masking)
# ============================================================================

class StandardCausalAttention(nn.Module):
    """Standard softmax attention with causal masking."""
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class LinearCausalAttention(nn.Module):
    """Linear attention with ELU+1 kernel and causal masking."""
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.eps = 1e-6
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # ELU+1 kernel
        q_prime = F.elu(q) + 1
        k_prime = F.elu(k) + 1
        
        # Causal linear attention via cumsum
        kv_prod = torch.einsum('bhtd,bhte->bhtde', k_prime, v)
        kv_cumsum = torch.cumsum(kv_prod, dim=2)
        context = torch.einsum('bhtd,bhtde->bhte', q_prime, kv_cumsum)
        
        # Normalization
        k_cumsum = torch.cumsum(k_prime, dim=2)
        norm = torch.einsum('bhtd,bhtd->bht', q_prime, k_cumsum)
        
        out = context / (norm.unsqueeze(-1) + self.eps)
        out = out.to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class CosformerCausalAttention(nn.Module):
    """Cosformer with cos-based reweighting and causal masking."""
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.eps = 1e-6
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        cos_w = torch.cos(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        sin_w = torch.sin(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        
        q_prime = F.relu(q)
        k_prime = F.relu(k)
        
        q_cos, q_sin = q_prime * cos_w, q_prime * sin_w
        k_cos, k_sin = k_prime * cos_w, k_prime * sin_w
        
        # Causal cumsum for cos component
        kv_cos = torch.einsum('bhtd,bhte->bhtde', k_cos, v)
        kv_cos_cumsum = torch.cumsum(kv_cos, dim=2)
        context_cos = torch.einsum('bhtd,bhtde->bhte', q_cos, kv_cos_cumsum)
        
        # Causal cumsum for sin component
        kv_sin = torch.einsum('bhtd,bhte->bhtde', k_sin, v)
        kv_sin_cumsum = torch.cumsum(kv_sin, dim=2)
        context_sin = torch.einsum('bhtd,bhtde->bhte', q_sin, kv_sin_cumsum)
        
        context = context_cos + context_sin
        
        # Normalization
        k_cos_cumsum = torch.cumsum(k_cos, dim=2)
        k_sin_cumsum = torch.cumsum(k_sin, dim=2)
        norm = (torch.einsum('bhtd,bhtd->bht', q_cos, k_cos_cumsum) +
                torch.einsum('bhtd,bhtd->bht', q_sin, k_sin_cumsum))
        
        out = context / (norm.unsqueeze(-1) + self.eps)
        out = out.to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class RFFCausalAttention(nn.Module):
    """Random Fourier Features attention (Gaussian kernel) with causal masking."""
    def __init__(self, embed_dim, n_heads, num_features=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.num_features = num_features
        self.eps = 1e-6
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.register_buffer('omega', torch.randn(n_heads, self.head_dim, num_features))
        self.register_buffer('bias', torch.rand(n_heads, num_features) * 2 * math.pi)
        
    def _rff_features(self, x):
        proj = torch.einsum('bhtd,hdm->bhtm', x, self.omega.float())
        proj = proj + self.bias.float().unsqueeze(0).unsqueeze(2)
        return math.sqrt(2.0 / self.num_features) * torch.cos(proj)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        q_prime = self._rff_features(q)
        k_prime = self._rff_features(k)
        
        # Causal linear attention via cumsum
        kv_prod = torch.einsum('bhtm,bhtd->bhtmd', k_prime, v)
        kv_cumsum = torch.cumsum(kv_prod, dim=2)
        context = torch.einsum('bhtm,bhtmd->bhtd', q_prime, kv_cumsum)
        
        # Normalization
        k_cumsum = torch.cumsum(k_prime, dim=2)
        norm = torch.einsum('bhtm,bhtm->bht', q_prime, k_cumsum)
        
        out = context / (norm.unsqueeze(-1) + self.eps)
        out = out.to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class YatCausalAttention(nn.Module):
    """Yat-product attention (exact) with causal masking.
    
    Uses kernel: (q·k)² / (||q||² + ||k||² - 2*q·k + eps)
    """
    def __init__(self, embed_dim, n_heads, epsilon=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = epsilon
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute dot product: q·k
        dot_product = torch.matmul(q, k.transpose(-2, -1))
        
        # Compute squared norms
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True)
        
        # Yat kernel: (q·k)² / (||q||² + ||k||² - 2*q·k + eps)
        numerator = dot_product ** 2
        denominator = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * dot_product + self.epsilon
        
        scores = numerator / denominator
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class YatPerformerCausalAttention(nn.Module):
    """Spherical ⵟ-Performer: Linearized Yat attention (FAST).
    
    Optimizations:
    1. Hadamard product (element-wise) instead of tensor product: M features vs M×M
    2. Reduced feature dimensions for speed
    3. Larger chunk size (512)
    4. Vectorized PRF computation
    """
    def __init__(self, embed_dim, n_heads, num_features=32, num_quadrature_nodes=2, epsilon=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.num_features = num_features  # Same for poly and PRF (for Hadamard)
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        # Gauss-Laguerre quadrature nodes and weights
        nodes, weights = self._gauss_laguerre_nodes(num_quadrature_nodes)
        self.register_buffer('quad_nodes', nodes / self.C)
        self.register_buffer('quad_weights', weights / self.C)
        
        # SHARED omega for both poly and PRF features (enables Hadamard product)
        # Shape: (R, H, D, M) - one omega per quadrature node
        self.register_buffer('omega', torch.randn(num_quadrature_nodes, n_heads, self.head_dim, num_features))
    
    def _gauss_laguerre_nodes(self, n):
        nodes, weights = np.polynomial.laguerre.laggauss(n)
        return torch.tensor(nodes, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)
    
    def _compute_features_fast(self, x):
        """Compute features using Hadamard product: φ_poly ⊙ φ_PRF.
        
        Instead of tensor product (M×M dims), we use element-wise product (M dims).
        This is valid because we share the random projection ω:
            E[(ω·q)² exp(√2s ω·q)] · E[(ω·k)² exp(√2s ω·k)] ≈ kernel(q,k)
        """
        B, H, T, D = x.shape
        R = self.num_quadrature_nodes
        M = self.num_features
        
        # Normalize input
        x_norm = F.normalize(x, p=2, dim=-1)  # (B, H, T, D)
        
        omega = self.omega.float()  # (R, H, D, M)
        
        # Compute all projections at once: (R, B, H, T, M)
        proj = torch.einsum('bhtd,rhdm->rbhtm', x_norm, omega)
        
        # Polynomial features: (ω·x)² / sqrt(M)
        poly_feat = (proj ** 2) / math.sqrt(M)  # (R, B, H, T, M)
        
        # PRF features for each quadrature node
        sqrt_2s = torch.sqrt(2.0 * self.quad_nodes.clamp(min=0)).view(R, 1, 1, 1, 1)
        s_vals = self.quad_nodes.view(R, 1, 1, 1, 1)
        
        exp_arg = torch.clamp(proj * sqrt_2s - s_vals, min=-20.0, max=20.0)
        prf_feat = torch.exp(exp_arg) / math.sqrt(M)  # (R, B, H, T, M)
        
        # Hadamard product: poly ⊙ prf (element-wise instead of tensor product)
        # This gives M features per node instead of M×M
        fused = poly_feat * prf_feat  # (R, B, H, T, M)
        
        # Apply quadrature weights
        sq_weights = torch.sqrt(self.quad_weights.clamp(min=0)).view(R, 1, 1, 1, 1)
        fused = fused * sq_weights  # (R, B, H, T, M)
        
        # Reshape to (B, H, T, R*M)
        fused = fused.permute(1, 2, 3, 0, 4).reshape(B, H, T, R * M)
        
        return fused
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        q_features = self._compute_features_fast(q)  # (B, H, T, R*M)
        k_features = self._compute_features_fast(k)
        
        feat_dim = q_features.shape[-1]  # R * M (much smaller than R * M_poly * M_prf)
        
        # --- Chunked Causal Linear Attention (larger chunks) ---
        chunk_size = 512
        num_chunks = (T + chunk_size - 1) // chunk_size
        
        kv_state = torch.zeros(B, self.n_heads, feat_dim, self.head_dim, 
                               device=x.device, dtype=q.dtype)
        k_state = torch.zeros(B, self.n_heads, feat_dim, 
                              device=x.device, dtype=q.dtype)
        
        out_chunks = []
        
        for i in range(num_chunks):
            st = i * chunk_size
            ed = min(st + chunk_size, T)
            
            q_chunk = q_features[:, :, st:ed]
            k_chunk = k_features[:, :, st:ed]
            v_chunk = v[:, :, st:ed]
            
            kv_chunk_prod = torch.einsum('bhtf,bhtd->bhtfd', k_chunk, v_chunk)
            kv_local_cumsum = torch.cumsum(kv_chunk_prod, dim=2)
            k_local_cumsum = torch.cumsum(k_chunk, dim=2)
            
            kv_current = kv_local_cumsum + kv_state.unsqueeze(2)
            k_current = k_local_cumsum + k_state.unsqueeze(2)
            
            context_chunk = torch.einsum('bhtf,bhtfd->bhtd', q_chunk, kv_current)
            denom_chunk = torch.einsum('bhtf,bhtf->bht', q_chunk, k_current)
            
            denom_chunk = torch.clamp(denom_chunk, min=1e-6)
            out_chunk = context_chunk / denom_chunk.unsqueeze(-1)
            out_chunks.append(out_chunk)
            
            kv_state = kv_current[:, :, -1]
            k_state = k_current[:, :, -1]
            
        out = torch.cat(out_chunks, dim=2)
        out = out.to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class YatSphericalCausalAttention(nn.Module):
    """Exact Spherical Yat attention with causal masking.
    
    Uses kernel: K(q,k) = x² / (C - 2x)
    where x = <q̂, k̂> (dot product of normalized vectors)
    and C = 2 + ε
    
    This is the exact target kernel for YatPerformer.
    """
    def __init__(self, embed_dim, n_heads, epsilon=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Normalize to unit sphere
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # x = <q̂, k̂>
        x_dot = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        # Kernel: x² / (C - 2x)
        denominator = self.C - 2 * x_dot
        denominator = torch.clamp(denominator, min=1e-6)
        scores = (x_dot ** 2) / denominator
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        
        # Normalize rows
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.matmul(scores, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.out(out)


class FastAttention(nn.Module):
    """Performer-style attention with ReLU random features."""
    def __init__(self, embed_dim, n_heads, kernel_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.kernel_size = kernel_size
        self.eps = 1e-6
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.register_buffer('proj_matrix', torch.randn(n_heads, self.head_dim, kernel_size))
    
    def _feature_map(self, x):
        proj = torch.einsum('bhtd,hdm->bhtm', x.float(), self.proj_matrix.float())
        return F.relu(proj) / math.sqrt(self.kernel_size)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        q_prime = self._feature_map(q)
        k_prime = self._feature_map(k)
        
        # Causal linear attention via cumsum
        kv_prod = torch.einsum('bhtm,bhtd->bhtmd', k_prime, v)
        kv_cumsum = torch.cumsum(kv_prod, dim=2)
        context = torch.einsum('bhtm,bhtmd->bhtd', q_prime, kv_cumsum)
        
        k_cumsum = torch.cumsum(k_prime, dim=2)
        norm = torch.einsum('bhtm,bhtm->bht', q_prime, k_cumsum)
        
        out = context / (norm.unsqueeze(-1) + self.eps)
        out = out.to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# Note: Old legacy attention classes removed - using new causal implementations above

    
    def _get_gauss_laguerre_weights(self) -> torch.Tensor:
        """Get Gauss-Laguerre quadrature weights"""
        if self.num_quad_nodes == 4:
            weights = torch.tensor([0.6031, 0.3575, 0.0388, 0.0005])
        elif self.num_quad_nodes == 8:
            weights = torch.tensor([
                0.3691, 0.4187, 0.1757, 0.0334,
                0.0028, 0.0001, 0.0000, 0.0000
            ])
        elif self.num_quad_nodes == 16:
            # Uniform weights as approximation
            weights = torch.ones(self.num_quad_nodes) / self.num_quad_nodes
        else:
            weights = torch.ones(self.num_quad_nodes) / self.num_quad_nodes
        return weights
    
    def _create_projection_matrix(self) -> torch.Tensor:
        """Create random projection matrix"""
        proj = torch.randn(self.num_features, self.dim)
        
        if self.use_orthogonal and self.num_features <= self.dim:
            q, _ = torch.linalg.qr(proj.T)
            proj = q.T[:self.num_features]
        
        return proj
    
    def _polynomial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute polynomial features for x^2 = (q^T k)^2
        Using outer product: vec(xx^T)
        
        For efficiency, we approximate with random projection
        """
        # Exact: vec(xx^T) would be d^2 dimensional
        # Approximation: random projection to lower dimension
        batch_size, seq_len, dim = x.shape
        
        # Simple approximation: [x, x^2 elementwise]
        # This is not exact but computationally efficient
        features = torch.cat([x, x ** 2], dim=-1)  # (batch, seq_len, 2*dim)
        
        return features
    
    def _positive_random_features(
        self,
        x: torch.Tensor,
        s: float,
        projection: torch.Tensor
    ) -> torch.Tensor:
        """
        Positive random features for exp(2s * q^T k)
        φ(x) = exp(√(2s) ω^T x - s) / √m
        """
        device = x.device
        projection = projection.to(device)
        
        # Project: ω^T x
        omega_x = torch.matmul(x, projection.T)  # (batch, seq_len, num_features)
        
        # Compute features: exp(√(2s) ω^T x - s)
        scale = np.sqrt(2 * s)
        features = torch.exp(scale * omega_x - s)
        
        # Normalize
        features = features / np.sqrt(self.num_features)
        
        return features
    
    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute combined features for all quadrature nodes
        Ψ(x) = concat[√w_r (φ_poly(x) ⊗ φ_PRF(x; s_r))] for r=1..R
        """
        batch_size, seq_len, dim = x.shape
        
        # Normalize to unit sphere
        x_normalized = F.normalize(x, p=2, dim=-1)
        
        all_features = []
        
        for r in range(self.num_quad_nodes):
            s_r = self.quad_nodes_scaled[r].item()
            w_r = self.quad_weights_scaled[r].item()
            
            # Polynomial features
            poly_feat = self._polynomial_features(x_normalized)  # (batch, seq, 2*dim)
            
            # PRF features
            prf_feat = self._positive_random_features(
                x_normalized, s_r, self.projection_matrices[r]
            )  # (batch, seq, num_features)
            
            # Tensor product (outer product flattened)
            # Approximate with concatenation for efficiency
            combined = torch.cat([poly_feat, prf_feat], dim=-1)
            
            # Scale by quadrature weight
            weighted = combined * np.sqrt(w_r)
            
            all_features.append(weighted)
        
        # Concatenate all quadrature nodes
        features = torch.cat(all_features, dim=-1)
        
        return features
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Spherical Yat-Performer attention
        
        Args:
            Q, K, V: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = Q.shape
        
        # Compute features
        Q_features = self._compute_features(Q)  # (batch, seq_len, feature_dim)
        K_features = self._compute_features(K)
        
        # Linear attention computation: Q'((K')^T V) / Q'((K')^T 1)
        # Numerator: (batch, feature_dim, dim_v)
        KV = torch.matmul(K_features.transpose(-2, -1), V)
        
        # (batch, seq_len, dim_v)
        QKV = torch.matmul(Q_features, KV)
        
        # Denominator: (batch, seq_len, 1)
        K_sum = K_features.sum(dim=1, keepdim=True)  # (batch, 1, feature_dim)
        normalizer = torch.matmul(Q_features, K_sum.transpose(-2, -1))
        
        # Stabilize
        normalizer = torch.clamp(normalizer, min=1e-8)
        
        # Normalize
        output = QKV / normalizer
        
        return output


# ============================================================================
# PERFORMER ATTENTION IMPLEMENTATION (CORRECTED)
# ============================================================================

class PerformerAttention(nn.Module):
    """
    Improved Performer attention using FAVOR+ mechanism
    Based on: "Rethinking Attention with Performers" (Choromanski et al., 2021)
    
    Improvements:
    - Option to redraw random features for training stability
    - Support for both softmax and ReLU feature maps
    - Better orthogonal projection initialization
    - Improved numerical stability
    """
    
    def __init__(
        self,
        dim: int,
        num_features: int = 256,
        use_orthogonal: bool = True,
        feature_type: str = 'softmax',  # 'softmax' or 'relu'
        redraw_features: bool = False,  # Redraw features each forward pass
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.use_orthogonal = use_orthogonal
        self.feature_type = feature_type
        self.redraw_features = redraw_features
        self.eps = eps
        
        # Scale for softmax kernel: 1/sqrt(d)
        self.scale = 1.0 / np.sqrt(dim)
        
        # Initialize projection matrix
        self.register_buffer('projection_matrix', torch.zeros(num_features, dim))
        self._initialize_projection()
    
    def _initialize_projection(self):
        """Create random projection matrix with orthogonal rows if possible"""
        if self.use_orthogonal:
            # Create orthogonal random features via QR decomposition
            # If num_features > dim, we need multiple orthogonal blocks
            num_blocks = (self.num_features + self.dim - 1) // self.dim
            blocks = []
            
            for _ in range(num_blocks):
                random_matrix = torch.randn(self.dim, self.dim)
                q, _ = torch.linalg.qr(random_matrix)
                # Scale by sqrt(dim) to preserve norm after projection
                blocks.append(q * np.sqrt(self.dim))
            
            # Stack and truncate to num_features
            proj = torch.cat(blocks, dim=0)[:self.num_features]
        else:
            # IID Gaussian
            proj = torch.randn(self.num_features, self.dim)
        
        self.projection_matrix.copy_(proj)
    
    def _redraw_projection(self):
        """Redraw random features - call during training for better stability"""
        self._initialize_projection()
    
    def _softmax_kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positive random features for softmax kernel.
        phi(x) = exp(omega^T x - ||x||^2/2) / sqrt(m)
        This gives: E[phi(x)^T phi(y)] ≈ exp(x^T y)
        """
        # Apply scaling for attention
        x_scaled = x * np.sqrt(self.scale)
        
        # Project: (batch, seq_len, num_features)
        proj_matrix = self.projection_matrix.to(x.device)
        projection = torch.matmul(x_scaled, proj_matrix.T)
        
        # Compute ||x||^2 / 2
        x_squared_norm = torch.sum(x_scaled ** 2, dim=-1, keepdim=True) / 2.0
        
        # Softmax kernel features
        features = torch.exp(projection - x_squared_norm)
        
        # Normalize by sqrt(m)
        features = features / np.sqrt(self.num_features)
        
        return features
    
    def _relu_kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        ReLU random features: simpler and sometimes more stable.
        phi(x) = relu(omega^T x + b) / sqrt(m)
        """
        proj_matrix = self.projection_matrix.to(x.device)
        projection = torch.matmul(x, proj_matrix.T)
        
        # ReLU activation
        features = F.relu(projection)
        
        # Normalize
        features = features / np.sqrt(self.num_features)
        
        return features
    
    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features based on feature_type"""
        if self.feature_type == 'softmax':
            return self._softmax_kernel_features(x)
        elif self.feature_type == 'relu':
            return self._relu_kernel_features(x)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Performer attention: O(L*d*r) complexity
        
        Args:
            Q, K, V: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Optionally redraw features
        if self.redraw_features and self.training:
            self._redraw_projection()
        
        # Get feature maps
        Q_prime = self._get_features(Q)  # (batch, seq_len, num_features)
        K_prime = self._get_features(K)  # (batch, seq_len, num_features)
        
        # FAVOR+ linear attention
        # Instead of: softmax(QK^T) V  [O(L²)]
        # We compute: Q' (K'^T V)       [O(Lr + rd) = O(Lrd)]
        
        # K'^T V: (batch, num_features, dim)
        KV = torch.einsum('bsf,bsd->bfd', K_prime, V)
        
        # Q' (K'^T V): (batch, seq_len, dim)  
        numerator = torch.einsum('bsf,bfd->bsd', Q_prime, KV)
        
        # Normalization: sum over keys
        K_sum = K_prime.sum(dim=1, keepdim=True)  # (batch, 1, num_features)
        denominator = torch.einsum('bsf,btf->bs', Q_prime, K_sum)  # (batch, seq_len)
        denominator = denominator.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Clamp for stability
        denominator = torch.clamp(denominator, min=self.eps)
        
        # Normalize
        output = numerator / denominator
        
        return output


class StandardAttention(nn.Module):
    """Standard softmax attention for comparison"""
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Standard attention: O(L^2*d) complexity
        Att(Q,K,V) = softmax(QK^T / sqrt(d)) V
        """
        d_k = Q.size(-1)
        
        # Compute attention scores with proper scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)
        
        return output


class LinearAttention(nn.Module):
    """
    Linear Attention using ELU+1 feature map
    Based on: "Transformers are RNNs" (Katharopoulos et al., 2020)
    
    Uses φ(x) = elu(x) + 1 as the feature mapping, which gives:
    - Non-negative features (necessary for stable attention)
    - Linear O(L*d*d) complexity
    
    Time complexity: O(L d²) - linear in sequence length
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ELU+1 feature map: φ(x) = elu(x) + 1"""
        return F.elu(x) + 1
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Linear attention with ELU+1 features
        
        Args:
            Q, K, V: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Apply feature map to Q and K
        Q_prime = self._feature_map(Q)  # (batch, seq_len, dim)
        K_prime = self._feature_map(K)  # (batch, seq_len, dim)
        
        # Compute K'^T V: (batch, dim, dim_v)
        KV = torch.einsum('bsd,bsv->bdv', K_prime, V)
        
        # Compute Q' (K'^T V): (batch, seq_len, dim_v)
        QKV = torch.einsum('bsd,bdv->bsv', Q_prime, KV)
        
        # Compute normalization: K' sum over sequence
        K_sum = K_prime.sum(dim=1, keepdim=True)  # (batch, 1, dim)
        
        # Q' dot K_sum for each position
        normalizer = torch.einsum('bsd,btd->bs', Q_prime, K_sum)  # (batch, seq_len)
        normalizer = normalizer.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Add epsilon for stability
        normalizer = torch.clamp(normalizer, min=self.eps)
        
        # Normalize
        output = QKV / normalizer
        
        return output


class CosineLinearAttention(nn.Module):
    """
    Linear Attention using cosine similarity / ReLU features
    Alternative to ELU+1, using simple ReLU features
    
    Time complexity: O(L d²) - linear in sequence length
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU feature map"""
        return F.relu(x)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Linear attention with ReLU features
        
        Args:
            Q, K, V: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Apply feature map to Q and K
        Q_prime = self._feature_map(Q)  # (batch, seq_len, dim)
        K_prime = self._feature_map(K)  # (batch, seq_len, dim)
        
        # Compute K'^T V: (batch, dim, dim_v)
        KV = torch.einsum('bsd,bsv->bdv', K_prime, V)
        
        # Compute Q' (K'^T V): (batch, seq_len, dim_v)
        QKV = torch.einsum('bsd,bdv->bsv', Q_prime, KV)
        
        # Compute normalization
        K_sum = K_prime.sum(dim=1, keepdim=True)  # (batch, 1, dim)
        normalizer = torch.einsum('bsd,btd->bs', Q_prime, K_sum)  # (batch, seq_len)
        normalizer = normalizer.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Add epsilon for stability
        normalizer = torch.clamp(normalizer, min=self.eps)
        
        # Normalize
        output = QKV / normalizer
        
        return output


# ============================================================================
# ADDITIONAL ATTENTION MECHANISMS FOR COMPARISON
# ============================================================================

class CosformerAttention(nn.Module):
    """
    Cosformer: Rethinking Softmax in Attention (Qin et al., 2022)
    
    Uses cosine-based reweighting with ReLU for linear attention.
    Key insight: cos(π/2 * (i-j)/M) provides good positional bias.
    
    Time complexity: O(Ld²) - linear in sequence length
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = Q.shape
        
        # Apply ReLU to Q and K (non-negative features)
        Q_prime = F.relu(Q)
        K_prime = F.relu(K)
        
        # Compute position indices for cos-based reweighting
        positions = torch.arange(seq_len, device=Q.device, dtype=Q.dtype)
        
        # Cos-based positional weights: cos(π * i / (2 * M))
        M = seq_len
        cos_weights = torch.cos(np.pi / 2 * positions / M)
        sin_weights = torch.sin(np.pi / 2 * positions / M)
        
        # Apply cos/sin weights to Q and K
        Q_cos = Q_prime * cos_weights.view(1, -1, 1)
        Q_sin = Q_prime * sin_weights.view(1, -1, 1)
        K_cos = K_prime * cos_weights.view(1, -1, 1)
        K_sin = K_prime * sin_weights.view(1, -1, 1)
        
        # Linear attention with cos-reweighted features
        KV_cos = torch.einsum('bsd,bsv->bdv', K_cos, V)
        KV_sin = torch.einsum('bsd,bsv->bdv', K_sin, V)
        
        QKV = torch.einsum('bsd,bdv->bsv', Q_cos, KV_cos) + torch.einsum('bsd,bdv->bsv', Q_sin, KV_sin)
        
        # Normalization
        K_cos_sum = K_cos.sum(dim=1, keepdim=True)
        K_sin_sum = K_sin.sum(dim=1, keepdim=True)
        normalizer = (torch.einsum('bsd,btd->bs', Q_cos, K_cos_sum) + 
                      torch.einsum('bsd,btd->bs', Q_sin, K_sin_sum))
        normalizer = normalizer.unsqueeze(-1).clamp(min=self.eps)
        
        return QKV / normalizer


class RFFAttention(nn.Module):
    """
    Random Fourier Features (RFF) Attention
    Classic RF baseline from Rahimi & Recht (2007)
    
    Uses random Fourier features to approximate the Gaussian/RBF kernel:
    k(x,y) = exp(-||x-y||² / (2σ²))
    
    φ(x) = sqrt(2/D) * cos(ωx + b)
    
    Time complexity: O(L*d*r) where r = num_features
    """
    
    def __init__(self, dim: int, num_features: int = 256, sigma: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.sigma = sigma
        self.eps = eps
        
        # Random frequencies ω ~ N(0, 1/σ²)
        self.register_buffer('omega', torch.randn(num_features, dim) / sigma)
        # Random phase shifts b ~ Uniform(0, 2π)
        self.register_buffer('bias', torch.rand(num_features) * 2 * np.pi)
    
    def _rff_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute random Fourier features: φ(x) = sqrt(2/D) * cos(ωx + b)"""
        projection = torch.matmul(x, self.omega.T) + self.bias
        return np.sqrt(2.0 / self.num_features) * torch.cos(projection)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Q_prime = self._rff_features(Q)
        K_prime = self._rff_features(K)
        
        # Linear attention with RFF features
        KV = torch.einsum('bsf,bsd->bfd', K_prime, V)
        numerator = torch.einsum('bsf,bfd->bsd', Q_prime, KV)
        
        K_sum = K_prime.sum(dim=1, keepdim=True)
        denominator = torch.einsum('bsf,btf->bs', Q_prime, K_sum).unsqueeze(-1).clamp(min=self.eps)
        
        return numerator / denominator


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def debug_kernel_features():
    """Test if kernel feature map is implemented correctly"""
    print("\n" + "="*60)
    print("DEBUG 1: Kernel Feature Map")
    print("="*60)
    
    dim = 64
    num_features = 256
    projection = torch.randn(num_features, dim)
    x = torch.randn(1, 10, dim)
    
    # Compute kernel features
    omega_x = torch.matmul(x, projection.T)
    x_norm = torch.sum(x ** 2, dim=-1, keepdim=True) / 2.0
    features = torch.exp(omega_x - x_norm) / np.sqrt(num_features)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Features stats: mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"All positive: {(features >= 0).all().item()}")
    
    if (features >= 0).all().item():
        print("✓ PASS: All features are positive")
    else:
        print("✗ FAIL: Some features are negative")


def debug_softmax_kernel():
    """Verify we're approximating the softmax kernel correctly"""
    print("\n" + "="*60)
    print("DEBUG 2: Softmax Kernel Approximation")
    print("="*60)
    
    dim = 64
    num_features_list = [64, 128, 256, 512]
    num_trials = 100
    
    # Generate random vectors
    x = torch.randn(dim)
    y = torch.randn(dim)
    
    # The scaling we use in attention
    scale_factor = 1.0 / np.sqrt(dim)
    scale = np.sqrt(scale_factor)
    
    # True kernel value with scaling: exp((x*scale)^T (y*scale))
    true_kernel = torch.exp(torch.dot(x * scale, y * scale)).item()
    print(f"True scaled softmax kernel: exp((x*s)^T(y*s)) = {true_kernel:.6f}")
    print(f"(where s = sqrt(1/sqrt(d)) = {scale:.6f})\n")
    
    print("Approximation with different feature counts:")
    for num_features in num_features_list:
        estimates = []
        
        for _ in range(num_trials):
            # Random projection
            omega = torch.randn(num_features, dim)
            
            # Apply to scaled inputs
            x_scaled = x * scale
            y_scaled = y * scale
            
            # Kernel features
            omega_x = torch.matmul(omega, x_scaled)
            omega_y = torch.matmul(omega, y_scaled)
            
            x_norm = torch.sum(x_scaled ** 2) / 2.0
            y_norm = torch.sum(y_scaled ** 2) / 2.0
            
            phi_x = torch.exp(omega_x - x_norm) / np.sqrt(num_features)
            phi_y = torch.exp(omega_y - y_norm) / np.sqrt(num_features)
            
            # Estimate: phi(x)^T phi(y)
            estimate = torch.dot(phi_x, phi_y).item()
            estimates.append(estimate)
        
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        error = abs(mean_estimate - true_kernel) / true_kernel * 100
        
        print(f"  m={num_features:4d}: {mean_estimate:8.4f} ± {std_estimate:6.4f} | Error: {error:5.2f}%")
        
        if num_features == 512:
            if error < 20:
                print("  ✓ PASS: Error < 20%")
            else:
                print("  ✗ FAIL: Error too high")


def debug_attention_output():
    """Compare Performer vs Standard attention outputs"""
    print("\n" + "="*60)
    print("DEBUG 3: Attention Output Comparison")
    print("="*60)
    
    batch_size, seq_len, dim = 2, 16, 64
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    # Standard attention
    standard = StandardAttention()
    standard_out = standard(Q, K, V)
    
    print(f"\nInput shapes: Q/K/V = {Q.shape}")
    print(f"Output shape: {standard_out.shape}")
    
    # Let's also manually compute to verify
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)
    manual_out = torch.matmul(attn_weights, V)
    
    print(f"\nStandard attention verification:")
    print(f"  Attention weights sum: {attn_weights.sum(dim=-1).mean():.6f} (should be ~1.0)")
    print(f"  Manual vs module match: {torch.allclose(standard_out, manual_out)}")
    
    # Test different feature counts
    feature_counts = [128, 256, 512]
    print("\nPerformer approximation quality:")
    
    for num_features in feature_counts:
        performer = PerformerAttention(dim, num_features=num_features)
        performer_out = performer(Q, K, V)
        
        # Check the kernel features
        Q_prime = performer._kernel_features(Q)
        K_prime = performer._kernel_features(K)
        
        print(f"\n  Features: {num_features}")
        print(f"    Q' stats: mean={Q_prime.mean():.4f}, std={Q_prime.std():.4f}, min={Q_prime.min():.4f}")
        print(f"    K' stats: mean={K_prime.mean():.4f}, std={K_prime.std():.4f}, min={K_prime.min():.4f}")
        print(f"    All positive: Q'={Q_prime.min()>=0}, K'={K_prime.min()>=0}")
        
        # Compare approximate vs exact attention matrix
        # Approximate: Q' K'^T (before normalization)
        approx_attn = torch.matmul(Q_prime, K_prime.transpose(-2, -1))
        
        # Exact: exp(QK^T / sqrt(d))
        exact_scores = torch.exp(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k))
        
        # These should be similar (before softmax normalization)
        matrix_cos_sim = F.cosine_similarity(
            approx_attn.flatten().unsqueeze(0),
            exact_scores.flatten().unsqueeze(0)
        ).item()
        
        print(f"    Attention matrix cosine sim: {matrix_cos_sim:.6f}")
        
        mse = F.mse_loss(performer_out, standard_out).item()
        cos_sim = F.cosine_similarity(
            performer_out.flatten().unsqueeze(0),
            standard_out.flatten().unsqueeze(0)
        ).item()
        
        print(f"    Output MSE: {mse:.6f}")
        print(f"    Output cosine sim: {cos_sim:.6f}")
        
        if num_features == 512:
            if cos_sim > 0.9:
                print("    ✓ PASS: Good approximation (cosine sim > 0.9)")
            elif cos_sim > 0.7:
                print("    ⚠ WARN: Moderate approximation (0.7 < cosine sim < 0.9)")
            else:
                print("    ✗ FAIL: Poor approximation (cosine sim < 0.7)")


def run_all_debug_tests():
    """Run all debugging tests"""
    print("\n" + "="*70)
    print(" "*15 + "PERFORMER IMPLEMENTATION DEBUGGER")
    print("="*70)
    
    debug_kernel_features()
    debug_softmax_kernel()
    debug_attention_output()
    
    print("\n" + "="*70)
    print("Debug tests completed!")
    print("="*70 + "\n")


# ============================================================================
# EXPERIMENT 1: COPY TASK
# ============================================================================

def run_copy_task(attention_type="performer", num_epochs=100, verbose=True):
    """
    Copy Task: Model must copy input sequence to output
    Tests: Basic attention mechanism functionality
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT 1: Copy Task ({attention_type.upper()})")
        print(f"{'='*60}")
    
    vocab_size, embed_dim, seq_len, batch_size = 10, 64, 20, 32
    
    # Model
    embedding = nn.Embedding(vocab_size, embed_dim)
    if attention_type == "performer":
        attention = PerformerAttention(embed_dim, num_features=256)
    else:
        attention = StandardAttention()
    output_proj = nn.Linear(embed_dim, vocab_size)
    
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + 
        list(attention.parameters()) + 
        list(output_proj.parameters()),
        lr=0.001
    )
    
    losses, accuracies = [], []
    
    for epoch in range(num_epochs):
        # Generate data
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = inputs.clone()
        
        # Forward
        x = embedding(inputs)
        attn_out = attention(x, x, x)
        logits = output_proj(attn_out)
        
        # Loss
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")
    
    if verbose:
        final_acc = accuracies[-1]
        print(f"\nFinal Accuracy: {final_acc:.4f}")
        if final_acc > 0.95:
            print("✓ PASS: Achieved >95% accuracy")
        else:
            print("✗ FAIL: Did not achieve >95% accuracy")
    
    return losses, accuracies


# ============================================================================
# EXPERIMENT 2: SORTING TASK
# ============================================================================

def run_sorting_task(attention_type="performer", num_epochs=300, verbose=True):
    """
    Sorting Task: Sort a sequence of numbers
    Tests: Long-range dependencies, global reasoning
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT 2: Sorting Task ({attention_type.upper()})")
        print(f"{'='*60}")
    
    # Make it easier: smaller vocab, shorter sequences
    max_value, embed_dim, seq_len, batch_size = 10, 128, 5, 32
    
    # Model with more capacity
    embedding = nn.Embedding(max_value, embed_dim)
    if attention_type == "performer":
        attention = PerformerAttention(embed_dim, num_features=256)
    else:
        attention = StandardAttention()
    
    # Add a second attention layer for more capacity
    attention2 = PerformerAttention(embed_dim, num_features=256) if attention_type == "performer" else StandardAttention()
    output_proj = nn.Linear(embed_dim, max_value)
    
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + 
        list(attention.parameters()) + 
        list(attention2.parameters()) +
        list(output_proj.parameters()),
        lr=0.001
    )
    
    losses, accuracies = [], []
    
    for epoch in range(num_epochs):
        # Generate data
        inputs = torch.randint(0, max_value, (batch_size, seq_len))
        targets = torch.sort(inputs, dim=1)[0]
        
        # Forward with 2 attention layers
        x = embedding(inputs)
        x = attention(x, x, x)
        x = attention2(x, x, x)
        logits = output_proj(x)
        
        # Loss
        loss = F.cross_entropy(logits.view(-1, max_value), targets.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(embedding.parameters()) + 
            list(attention.parameters()) + 
            list(attention2.parameters()) +
            list(output_proj.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        # Metrics
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        if verbose and (epoch + 1) % 60 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")
    
    if verbose:
        final_acc = accuracies[-1]
        print(f"\nFinal Accuracy: {final_acc:.4f}")
    
    return losses, accuracies


# ============================================================================
# EXPERIMENT 3: ASSOCIATIVE RECALL
# ============================================================================

def run_associative_recall(attention_type="performer", num_epochs=500, verbose=True):
    """
    Associative Recall: Memorize key-value pairs
    Tests: Associative memory, retrieval
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT 3: Associative Recall ({attention_type.upper()})")
        print(f"{'='*60}")
    
    # Easier version: fewer pairs, smaller vocab
    vocab_size, embed_dim, num_pairs, batch_size = 10, 128, 3, 32
    seq_len = num_pairs * 2 + 1
    
    # Model
    embedding = nn.Embedding(vocab_size, embed_dim)
    if attention_type == "performer":
        attention = PerformerAttention(embed_dim, num_features=256)
    else:
        attention = StandardAttention()
    output_proj = nn.Linear(embed_dim, vocab_size)
    
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + 
        list(attention.parameters()) + 
        list(output_proj.parameters()),
        lr=0.003  # Higher learning rate
    )
    
    losses, accuracies = [], []
    
    for epoch in range(num_epochs):
        # Generate data
        sequences = []
        targets = []
        
        for _ in range(batch_size):
            keys = torch.randperm(vocab_size)[:num_pairs]
            values = torch.randint(0, vocab_size, (num_pairs,))
            
            seq = torch.zeros(seq_len, dtype=torch.long)
            for i in range(num_pairs):
                seq[i * 2] = keys[i]
                seq[i * 2 + 1] = values[i]
            
            query_idx = torch.randint(0, num_pairs, (1,)).item()
            seq[-1] = keys[query_idx]
            target = values[query_idx]
            
            sequences.append(seq)
            targets.append(target)
        
        inputs = torch.stack(sequences)
        targets = torch.tensor(targets)
        
        # Forward
        x = embedding(inputs)
        attn_out = attention(x, x, x)
        logits = output_proj(attn_out[:, -1, :])
        
        # Loss
        loss = F.cross_entropy(logits, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")
    
    if verbose:
        final_acc = accuracies[-1]
        print(f"\nFinal Accuracy: {final_acc:.4f}")
        if final_acc > 0.8:
            print("✓ PASS: Achieved >80% accuracy")
        elif final_acc > 0.5:
            print("⚠ WARN: Moderate performance (50-80%)")
        else:
            print("✗ FAIL: Did not achieve >50% accuracy")
    
    return losses, accuracies


# ============================================================================
# EXPERIMENT 4: APPROXIMATION QUALITY
# ============================================================================

def run_approximation_quality():
    """
    Direct measurement of approximation quality
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4: Approximation Quality")
    print(f"{'='*60}")
    
    seq_lens = [16, 32, 64, 128, 256, 512]
    dim = 64
    num_features_list = [64, 128, 256, 512]
    
    results = {}
    
    for num_features in num_features_list:
        mse_scores = []
        cosine_scores = []
        
        for seq_len in seq_lens:
            Q = torch.randn(1, seq_len, dim)
            K = torch.randn(1, seq_len, dim)
            V = torch.randn(1, seq_len, dim)
            
            # Standard
            standard = StandardAttention()
            standard_out = standard(Q, K, V)
            
            # Performer
            performer = PerformerAttention(dim, num_features)
            performer_out = performer(Q, K, V)
            
            # Metrics
            mse = F.mse_loss(performer_out, standard_out).item()
            cos_sim = F.cosine_similarity(
                performer_out.flatten().unsqueeze(0),
                standard_out.flatten().unsqueeze(0)
            ).item()
            
            mse_scores.append(mse)
            cosine_scores.append(cos_sim)
        
        results[num_features] = {
            'mse': mse_scores,
            'cosine': cosine_scores,
        }
        
        print(f"\nFeatures: {num_features}")
        print(f"  Avg MSE: {np.mean(mse_scores):.6f}")
        print(f"  Avg Cosine Similarity: {np.mean(cosine_scores):.6f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE
    for num_features, data in results.items():
        axes[0].plot(seq_lens, data['mse'], marker='o', label=f'r={num_features}')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Approximation Error vs Sequence Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cosine similarity
    for num_features, data in results.items():
        axes[1].plot(seq_lens, data['cosine'], marker='o', label=f'r={num_features}')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Approximation Quality vs Sequence Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('approximation_quality.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: approximation_quality.png")
    
    return results


# ============================================================================
# EXPERIMENT 5: SPEED BENCHMARK
# ============================================================================

def run_speed_benchmark():
    """
    Measure actual computational speed
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 5: Speed Benchmark")
    print(f"{'='*60}")
    
    seq_lens = [64, 128, 256, 512, 1024, 2048]
    dim = 64
    batch_size = 16
    num_trials = 10
    
    standard_times = []
    performer_times = []
    
    for seq_len in seq_lens:
        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)
        
        # Standard attention
        standard = StandardAttention()
        
        # Warmup
        for _ in range(3):
            _ = standard(Q, K, V)
        
        start = time.time()
        for _ in range(num_trials):
            _ = standard(Q, K, V)
        std_time = (time.time() - start) / num_trials * 1000
        
        # Performer attention
        performer = PerformerAttention(dim, num_features=256)
        
        # Warmup
        for _ in range(3):
            _ = performer(Q, K, V)
        
        start = time.time()
        for _ in range(num_trials):
            _ = performer(Q, K, V)
        perf_time = (time.time() - start) / num_trials * 1000
        
        standard_times.append(std_time)
        performer_times.append(perf_time)
        
        speedup = std_time / perf_time
        print(f"L={seq_len:4d} | Std: {std_time:7.2f}ms | Perf: {perf_time:7.2f}ms | Speedup: {speedup:.2f}x")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, standard_times, marker='o', label='Standard', linewidth=2)
    plt.plot(seq_lens, performer_times, marker='s', label='Performer', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Attention Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('speed_benchmark.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: speed_benchmark.png")
    
    return standard_times, performer_times


# ============================================================================
# EXPERIMENT 8: COMPREHENSIVE ATTENTION COMPARISON
# ============================================================================

def run_comprehensive_attention_comparison(num_epochs=100):
    """
    Compare all attention mechanisms:
    1. Standard Softmax
    2. Performer (FAVOR+)
    3. Yat-Product
    4. Spherical Yat-Performer
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 8: Comprehensive Attention Comparison")
    print(f"{'='*60}\n")
    
    vocab_size = 10
    embed_dim = 64
    seq_len = 128
    batch_size = 32
    
    attention_types = {
        'Standard': StandardAttention(),
        'Performer': PerformerAttention(embed_dim, num_features=256),
        'Yat': YatAttention(epsilon=0.1),
        'Yat-Performer': SphericalYatPerformer(
            embed_dim,
            num_random_features=256,
            num_quadrature_nodes=8,
            epsilon=0.1
        ),
    }
    
    results = {name: {'losses': [], 'accuracies': []} for name in attention_types}
    
    for name, attention in attention_types.items():
        print(f"\n--- Training with {name} Attention ---")
        
        # Model
        embedding = nn.Embedding(vocab_size, embed_dim)
        output_proj = nn.Linear(embed_dim, vocab_size)
        
        optimizer = torch.optim.Adam(
            list(embedding.parameters()) + 
            list(attention.parameters()) + 
            list(output_proj.parameters()),
            lr=0.001
        )
        
        # Training
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs.clone()
            
            # Forward
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            # Loss
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            results[name]['losses'].append(loss.item())
            results[name]['accuracies'].append(accuracy)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")
        
        final_acc = results[name]['accuracies'][-1]
        print(f"Final Accuracy: {final_acc:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'green', 'red', 'purple']
    for (name, data), color in zip(results.items(), colors):
        axes[0].plot(data['accuracies'], label=name, alpha=0.7, linewidth=2, color=color)
        axes[1].plot(data['losses'], label=name, alpha=0.7, linewidth=2, color=color)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('attention_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: attention_comparison.png")
    
    # Summary table
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Attention Type':<20} {'Final Acc':<12} {'Final Loss':<12}")
    print(f"{'-'*60}")
    for name, data in results.items():
        final_acc = data['accuracies'][-1]
        final_loss = data['losses'][-1]
        print(f"{name:<20} {final_acc:<12.4f} {final_loss:<12.4f}")
    
    return results


# ============================================================================
# EXPERIMENT 9: SPEED COMPARISON (ALL ATTENTION TYPES)
# ============================================================================

def run_speed_comparison_all():
    """
    Compare speed of all attention mechanisms across sequence lengths
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 9: Speed Benchmark (All Attention Types)")
    print(f"{'='*60}\n")
    
    seq_lens = [64, 128, 256, 512, 1024]
    dim = 64
    batch_size = 16
    num_trials = 10
    
    attention_modules = {
        'Standard': StandardAttention(),
        'Performer': PerformerAttention(dim, num_features=256),
        'Yat': YatAttention(epsilon=0.1),
        'Yat-Performer': SphericalYatPerformer(
            dim,
            num_random_features=128,  # Smaller for speed
            num_quadrature_nodes=4,
            epsilon=0.1
        ),
    }
    
    results = {name: [] for name in attention_modules}
    
    for seq_len in seq_lens:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)
        
        for name, attention in attention_modules.items():
            # Warmup
            for _ in range(3):
                _ = attention(Q, K, V)
            
            # Timing
            start = time.time()
            for _ in range(num_trials):
                _ = attention(Q, K, V)
            elapsed = (time.time() - start) / num_trials * 1000  # ms
            
            results[name].append(elapsed)
            print(f"{name:<20} {elapsed:>8.2f} ms")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple']
    
    for (name, times), color in zip(results.items(), colors):
        plt.plot(seq_lens, times, marker='o', label=name, linewidth=2, color=color)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Attention Mechanism Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.savefig('speed_comparison_all.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: speed_comparison_all.png")
    
    return results


# ============================================================================
# EXPERIMENT 10: APPROXIMATION QUALITY (YAT vs YAT-PERFORMER)
# ============================================================================

def run_yat_approximation_quality():
    """
    Test how well Yat-Performer approximates Yat-Product
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 10: Yat vs Yat-Performer Approximation")
    print(f"{'='*60}\n")
    
    seq_lens = [16, 32, 64, 128, 256]
    dim = 64
    batch_size = 4
    
    num_features_list = [64, 128, 256, 512]
    
    results = {}
    
    for num_features in num_features_list:
        mse_scores = []
        cosine_scores = []
        
        for seq_len in seq_lens:
            Q = torch.randn(batch_size, seq_len, dim)
            K = torch.randn(batch_size, seq_len, dim)
            V = torch.randn(batch_size, seq_len, dim)
            
            # Yat-Product (ground truth)
            yat = YatAttention(epsilon=0.1)
            yat_out = yat(Q, K, V)
            
            # Yat-Performer (approximation)
            yat_perf = SphericalYatPerformer(
                dim,
                num_random_features=num_features,
                num_quadrature_nodes=8,
                epsilon=0.1
            )
            yat_perf_out = yat_perf(Q, K, V)
            
            # Metrics
            mse = F.mse_loss(yat_perf_out, yat_out).item()
            cos_sim = F.cosine_similarity(
                yat_perf_out.flatten().unsqueeze(0),
                yat_out.flatten().unsqueeze(0)
            ).item()
            
            mse_scores.append(mse)
            cosine_scores.append(cos_sim)
        
        results[num_features] = {
            'mse': mse_scores,
            'cosine': cosine_scores,
        }
        
        print(f"\nFeatures: {num_features}")
        print(f"  Avg MSE: {np.mean(mse_scores):.6f}")
        print(f"  Avg Cosine Similarity: {np.mean(cosine_scores):.6f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for num_features, data in results.items():
        axes[0].plot(seq_lens, data['mse'], marker='o', label=f'r={num_features}')
        axes[1].plot(seq_lens, data['cosine'], marker='o', label=f'r={num_features}')
    
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Yat-Performer Approximation Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Yat-Performer Approximation Quality')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yat_approximation.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: yat_approximation.png")
    
    return results


# ============================================================================
# LONG CONTEXT EXPERIMENTS (UPDATED WITH YAT)
# ============================================================================

def run_long_context_copy_task(seq_lens=[64, 128, 256, 512, 1024, 2048], num_epochs=50):
    """
    Test copy task at different sequence lengths
    This shows where Performer really shines
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 6: Long Context Copy Task")
    print(f"{'='*60}\n")
    
    vocab_size = 10
    embed_dim = 64
    batch_size = 16
    
    results = {
        'seq_lens': seq_lens,
        'performer_acc': [],
        'performer_time': [],
        'standard_acc': [],
        'standard_time': [],
    }
    
    for seq_len in seq_lens:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # Test Performer
        embedding_perf = nn.Embedding(vocab_size, embed_dim)
        attention_perf = PerformerAttention(embed_dim, num_features=256)
        output_perf = nn.Linear(embed_dim, vocab_size)
        
        optimizer_perf = torch.optim.Adam(
            list(embedding_perf.parameters()) + 
            list(attention_perf.parameters()) + 
            list(output_perf.parameters()),
            lr=0.001
        )
        
        # Train Performer
        start_time = time.time()
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs.clone()
            
            x = embedding_perf(inputs)
            attn_out = attention_perf(x, x, x)
            logits = output_perf(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            optimizer_perf.zero_grad()
            loss.backward()
            optimizer_perf.step()
        
        perf_time = time.time() - start_time
        
        # Final eval
        with torch.no_grad():
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs.clone()
            x = embedding_perf(inputs)
            attn_out = attention_perf(x, x, x)
            logits = output_perf(attn_out)
            predictions = logits.argmax(dim=-1)
            perf_acc = (predictions == targets).float().mean().item()
        
        results['performer_acc'].append(perf_acc)
        results['performer_time'].append(perf_time)
        
        print(f"Performer: Acc={perf_acc:.4f}, Time={perf_time:.2f}s")
        
        # Test Standard (skip very long sequences to avoid OOM)
        if seq_len <= 512:
            embedding_std = nn.Embedding(vocab_size, embed_dim)
            attention_std = StandardAttention()
            output_std = nn.Linear(embed_dim, vocab_size)
            
            optimizer_std = torch.optim.Adam(
                list(embedding_std.parameters()) + 
                list(attention_std.parameters()) + 
                list(output_std.parameters()),
                lr=0.001
            )
            
            # Train Standard
            start_time = time.time()
            for epoch in range(num_epochs):
                inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
                targets = inputs.clone()
                
                x = embedding_std(inputs)
                attn_out = attention_std(x, x, x)
                logits = output_std(attn_out)
                
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                
                optimizer_std.zero_grad()
                loss.backward()
                optimizer_std.step()
            
            std_time = time.time() - start_time
            
            # Final eval
            with torch.no_grad():
                inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
                targets = inputs.clone()
                x = embedding_std(inputs)
                attn_out = attention_std(x, x, x)
                logits = output_std(attn_out)
                predictions = logits.argmax(dim=-1)
                std_acc = (predictions == targets).float().mean().item()
            
            results['standard_acc'].append(std_acc)
            results['standard_time'].append(std_time)
            
            print(f"Standard:  Acc={std_acc:.4f}, Time={std_time:.2f}s")
            print(f"Speedup: {std_time/perf_time:.2f}x")
        else:
            results['standard_acc'].append(None)
            results['standard_time'].append(None)
            print(f"Standard:  Skipped (too long, would OOM)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    valid_lens_std = [l for l, a in zip(seq_lens, results['standard_acc']) if a is not None]
    valid_acc_std = [a for a in results['standard_acc'] if a is not None]
    
    axes[0].plot(valid_lens_std, valid_acc_std, marker='o', label='Standard', linewidth=2)
    axes[0].plot(seq_lens, results['performer_acc'], marker='s', label='Performer', linewidth=2)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Final Accuracy')
    axes[0].set_title('Copy Task Accuracy vs Sequence Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log', base=2)
    
    # Training time comparison
    valid_time_std = [t for t in results['standard_time'] if t is not None]
    
    axes[1].plot(valid_lens_std, valid_time_std, marker='o', label='Standard', linewidth=2)
    axes[1].plot(seq_lens, results['performer_time'], marker='s', label='Performer', linewidth=2)
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Training Time vs Sequence Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('long_context_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: long_context_results.png")
    
    return results


# ============================================================================
# EXPERIMENT 7: LONG SEQUENCE RETRIEVAL
# ============================================================================

def run_long_sequence_retrieval(seq_lens=[128, 256, 512, 1024, 2048], num_epochs=100):
    """
    Needle in haystack: Find a specific token in a long sequence
    Tests: Long-range attention, retrieval from long context
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 7: Long Sequence Retrieval (Needle in Haystack)")
    print(f"{'='*60}\n")
    
    vocab_size = 50
    embed_dim = 64
    batch_size = 16
    special_token = vocab_size - 1  # The "needle"
    
    results = {
        'seq_lens': seq_lens,
        'performer_acc': [],
        'standard_acc': [],
    }
    
    for seq_len in seq_lens:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # Test Performer
        embedding_perf = nn.Embedding(vocab_size, embed_dim)
        attention_perf = PerformerAttention(embed_dim, num_features=256)
        output_perf = nn.Linear(embed_dim, vocab_size)
        
        optimizer_perf = torch.optim.Adam(
            list(embedding_perf.parameters()) + 
            list(attention_perf.parameters()) + 
            list(output_perf.parameters()),
            lr=0.001
        )
        
        # Train Performer
        for epoch in range(num_epochs):
            # Generate sequences with random position of special token
            inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
            positions = torch.randint(0, seq_len, (batch_size,))
            
            # Insert special token at random position
            for i, pos in enumerate(positions):
                inputs[i, pos] = special_token
            
            # Task: predict the position of the special token
            x = embedding_perf(inputs)
            attn_out = attention_perf(x, x, x)
            # Use last position to predict
            logits = output_perf(attn_out[:, -1, :])
            
            # Target is the special token (simplified: just predict if it exists)
            targets = torch.ones(batch_size, dtype=torch.long) * special_token
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer_perf.zero_grad()
            loss.backward()
            optimizer_perf.step()
        
        # Eval
        with torch.no_grad():
            inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
            positions = torch.randint(0, seq_len, (batch_size,))
            for i, pos in enumerate(positions):
                inputs[i, pos] = special_token
            
            x = embedding_perf(inputs)
            attn_out = attention_perf(x, x, x)
            logits = output_perf(attn_out[:, -1, :])
            predictions = logits.argmax(dim=-1)
            targets = torch.ones(batch_size, dtype=torch.long) * special_token
            perf_acc = (predictions == targets).float().mean().item()
        
        results['performer_acc'].append(perf_acc)
        print(f"Performer: Acc={perf_acc:.4f}")
        
        # Test Standard (only for shorter sequences)
        if seq_len <= 512:
            embedding_std = nn.Embedding(vocab_size, embed_dim)
            attention_std = StandardAttention()
            output_std = nn.Linear(embed_dim, vocab_size)
            
            optimizer_std = torch.optim.Adam(
                list(embedding_std.parameters()) + 
                list(attention_std.parameters()) + 
                list(output_std.parameters()),
                lr=0.001
            )
            
            for epoch in range(num_epochs):
                inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
                positions = torch.randint(0, seq_len, (batch_size,))
                for i, pos in enumerate(positions):
                    inputs[i, pos] = special_token
                
                x = embedding_std(inputs)
                attn_out = attention_std(x, x, x)
                logits = output_std(attn_out[:, -1, :])
                targets = torch.ones(batch_size, dtype=torch.long) * special_token
                
                loss = F.cross_entropy(logits, targets)
                optimizer_std.zero_grad()
                loss.backward()
                optimizer_std.step()
            
            with torch.no_grad():
                inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
                positions = torch.randint(0, seq_len, (batch_size,))
                for i, pos in enumerate(positions):
                    inputs[i, pos] = special_token
                
                x = embedding_std(inputs)
                attn_out = attention_std(x, x, x)
                logits = output_std(attn_out[:, -1, :])
                predictions = logits.argmax(dim=-1)
                targets = torch.ones(batch_size, dtype=torch.long) * special_token
                std_acc = (predictions == targets).float().mean().item()
            
            results['standard_acc'].append(std_acc)
            print(f"Standard:  Acc={std_acc:.4f}")
        else:
            results['standard_acc'].append(None)
            print(f"Standard:  Skipped (too long)")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    valid_lens = [l for l, a in zip(seq_lens, results['standard_acc']) if a is not None]
    valid_acc = [a for a in results['standard_acc'] if a is not None]
    
    plt.plot(valid_lens, valid_acc, marker='o', label='Standard', linewidth=2)
    plt.plot(seq_lens, results['performer_acc'], marker='s', label='Performer', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Retrieval Accuracy')
    plt.title('Long Sequence Retrieval: Needle in Haystack')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig('long_sequence_retrieval.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: long_sequence_retrieval.png")
    
    return results


# ============================================================================
# FULL EXPERIMENT SUITE (UPDATED WITH LONG CONTEXT TESTS)
# ============================================================================

def run_full_experiments():
    """Run all experiments including Yat attention comparisons"""
    print("\n" + "="*70)
    print(" "*15 + "COMPLETE ATTENTION VALIDATION SUITE")
    print(" "*10 + "(Standard | Performer | Yat | Yat-Performer)")
    print("="*70)
    
    # Run all experiments
    print("\n[1/10] Copy Task (Standard vs Performer)")
    copy_perf_loss, copy_perf_acc = run_copy_task("performer", num_epochs=100)
    copy_std_loss, copy_std_acc = run_copy_task("standard", num_epochs=100)
    
    print("\n[2/10] Sorting Task")
    sort_perf_loss, sort_perf_acc = run_sorting_task("performer", num_epochs=300)
    sort_std_loss, sort_std_acc = run_sorting_task("standard", num_epochs=300)
    
    print("\n[3/10] Associative Recall")
    recall_perf_loss, recall_perf_acc = run_associative_recall("performer", num_epochs=500)
    recall_std_loss, recall_std_acc = run_associative_recall("standard", num_epochs=500)
    
    print("\n[4/10] Approximation Quality (Performer)")
    approx_results = run_approximation_quality()
    
    print("\n[5/10] Speed Benchmark (Standard vs Performer)")
    speed_results = run_speed_benchmark()
    
    print("\n[6/10] Long Context Copy Task")
    long_copy_results = run_long_context_copy_task(
        seq_lens=[64, 128, 256, 512, 1024, 2048],
        num_epochs=50
    )
    
    print("\n[7/10] Long Sequence Retrieval")
    retrieval_results = run_long_sequence_retrieval(
        seq_lens=[128, 256, 512, 1024, 2048],
        num_epochs=100
    )
    
    print("\n[8/10] Comprehensive Attention Comparison")
    comparison_results = run_comprehensive_attention_comparison(num_epochs=100)
    
    print("\n[9/10] Speed Comparison (All Attention Types)")
    speed_all_results = run_speed_comparison_all()
    
    print("\n[10/10] Yat Approximation Quality")
    yat_approx_results = run_yat_approximation_quality()
    
    # Generate summary plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Copy task
    axes[0, 0].plot(copy_std_acc, label='Standard', alpha=0.7, linewidth=2)
    axes[0, 0].plot(copy_perf_acc, label='Performer', alpha=0.7, linewidth=2)
    axes[0, 0].set_title('Copy Task - Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sorting task
    axes[0, 1].plot(sort_std_acc, label='Standard', alpha=0.7, linewidth=2)
    axes[0, 1].plot(sort_perf_acc, label='Performer', alpha=0.7, linewidth=2)
    axes[0, 1].set_title('Sorting Task - Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Associative recall
    axes[0, 2].plot(recall_std_acc, label='Standard', alpha=0.7, linewidth=2)
    axes[0, 2].plot(recall_perf_acc, label='Performer', alpha=0.7, linewidth=2)
    axes[0, 2].set_title('Associative Recall - Accuracy', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Losses
    axes[1, 0].plot(copy_std_loss, label='Standard', alpha=0.7, linewidth=2)
    axes[1, 0].plot(copy_perf_loss, label='Performer', alpha=0.7, linewidth=2)
    axes[1, 0].set_title('Copy Task - Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(sort_std_loss, label='Standard', alpha=0.7, linewidth=2)
    axes[1, 1].plot(sort_perf_loss, label='Performer', alpha=0.7, linewidth=2)
    axes[1, 1].set_title('Sorting Task - Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(recall_std_loss, label='Standard', alpha=0.7, linewidth=2)
    axes[1, 2].plot(recall_perf_loss, label='Performer', alpha=0.7, linewidth=2)
    axes[1, 2].set_title('Associative Recall - Loss', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: full_summary.png")
    
    # Print summary
    print("\n" + "="*70)
    print(" "*25 + "FINAL SUMMARY")
    print("="*70)
    print(f"\nBasic Tasks:")
    print(f"  Copy Task:")
    print(f"    Standard:  {copy_std_acc[-1]:.4f}")
    print(f"    Performer: {copy_perf_acc[-1]:.4f}")
    print(f"  Sorting Task:")
    print(f"    Standard:  {sort_std_acc[-1]:.4f}")
    print(f"    Performer: {sort_perf_acc[-1]:.4f}")
    print(f"  Associative Recall:")
    print(f"    Standard:  {recall_std_acc[-1]:.4f}")
    print(f"    Performer: {recall_perf_acc[-1]:.4f}")
    
    print(f"\nLong Context Performance:")
    print(f"  Copy Task (L=2048):")
    print(f"    Performer: {long_copy_results['performer_acc'][-1]:.4f}")
    print(f"    Standard:  N/A (OOM)")
    print(f"  Retrieval (L=2048):")
    print(f"    Performer: {retrieval_results['performer_acc'][-1]:.4f}")
    print(f"    Standard:  N/A (OOM)")
    
    print(f"\nSpeed (L=1024):")
    perf_idx = 4  # Index for L=1024
    std_time = speed_results[0][perf_idx]
    perf_time = speed_results[1][perf_idx]
    print(f"    Standard:  {std_time:.2f} ms")
    print(f"    Performer: {perf_time:.2f} ms")
    print(f"    Speedup:   {std_time/perf_time:.2f}x")
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("Generated plots:")
    print("  - full_summary.png")
    print("  - approximation_quality.png")
    print("  - speed_benchmark.png")
    print("  - long_context_results.png")
    print("  - long_sequence_retrieval.png")
    print("="*70 + "\n")


# ============================================================================
# QUICK VALIDATION
# ============================================================================

def run_quick_validation():
    """Quick validation with reduced epochs"""
    print("\n" + "="*70)
    print(" "*20 + "QUICK VALIDATION MODE")
    print("="*70)
    
    print("\n[1/4] Copy Task (20 epochs)")
    copy_perf_loss, copy_perf_acc = run_copy_task("performer", num_epochs=20, verbose=False)
    copy_std_loss, copy_std_acc = run_copy_task("standard", num_epochs=20, verbose=False)
    
    print(f"  Standard:  Final Acc = {copy_std_acc[-1]:.4f}")
    print(f"  Performer: Final Acc = {copy_perf_acc[-1]:.4f}")
    
    print("\n[2/4] Direct Approximation Test")
    batch_size, seq_len, dim = 4, 32, 64
    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)
    V = torch.randn(batch_size, seq_len, dim)
    
    standard = StandardAttention()
    standard_out = standard(Q, K, V)
    
    performer = PerformerAttention(dim, num_features=256)
    performer_out = performer(Q, K, V)
    
    cos_sim = F.cosine_similarity(
        performer_out.flatten().unsqueeze(0),
        standard_out.flatten().unsqueeze(0)
    ).item()
    
    print(f"  Cosine Similarity: {cos_sim:.6f}")
    
    if cos_sim > 0.9:
        print("  ✓ PASS: Excellent approximation")
    elif cos_sim > 0.7:
        print("  ⚠ WARN: Moderate approximation")
    else:
        print("  ✗ FAIL: Poor approximation")
    
    print("\n[3/4] Speed Test (L=1024)")
    Q = torch.randn(16, 1024, 64)
    K = torch.randn(16, 1024, 64)
    V = torch.randn(16, 1024, 64)
    
    # Standard
    start = time.time()
    for _ in range(10):
        _ = standard(Q, K, V)
    std_time = (time.time() - start) / 10 * 1000
    
    # Performer
    start = time.time()
    for _ in range(10):
        _ = performer(Q, K, V)
    perf_time = (time.time() - start) / 10 * 1000
    
    print(f"  Standard:  {std_time:.2f} ms")
    print(f"  Performer: {perf_time:.2f} ms")
    print(f"  Speedup:   {std_time/perf_time:.2f}x")
    
    print("\n[4/4] Long Context Copy (L=1024, 20 epochs)")
    vocab_size = 10
    embed_dim = 64
    seq_len = 1024
    batch_size = 8
    
    # Performer
    embedding_perf = nn.Embedding(vocab_size, embed_dim)
    attention_perf = PerformerAttention(embed_dim, num_features=256)
    output_perf = nn.Linear(embed_dim, vocab_size)
    optimizer_perf = torch.optim.Adam(
        list(embedding_perf.parameters()) + 
        list(attention_perf.parameters()) + 
        list(output_perf.parameters()),
        lr=0.001
    )
    
    for epoch in range(20):
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = inputs.clone()
        x = embedding_perf(inputs)
        attn_out = attention_perf(x, x, x)
        logits = output_perf(attn_out)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        optimizer_perf.zero_grad()
        loss.backward()
        optimizer_perf.step()
    
    # Eval
    with torch.no_grad():
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = inputs.clone()
        x = embedding_perf(inputs)
        attn_out = attention_perf(x, x, x)
        logits = output_perf(attn_out)
        predictions = logits.argmax(dim=-1)
        perf_acc = (predictions == targets).float().mean().item()
    
    print(f"  Performer Accuracy: {perf_acc:.4f}")
    
    if perf_acc > 0.9:
        print("  ✓ PASS: Performer handles long context well")
    elif perf_acc > 0.7:
        print("  ⚠ WARN: Moderate performance on long context")
    else:
        print("  ✗ FAIL: Poor long context performance")
    
    print("\n" + "="*70)
    print("Quick validation completed!")
    print("="*70 + "\n")


# ============================================================================
# EXPERIMENT 11: COMPLETE ATTENTION BENCHMARK WITH LOGGING
# ============================================================================

def run_complete_attention_benchmark(
    num_epochs: int = 100,
    log_file: str = "attention_benchmark_results.txt",
    seq_lens_speed: List[int] = [64, 128, 256, 512, 1024, 2048],
    seq_lens_approx: List[int] = [16, 32, 64, 128, 256],
    num_random_features: int = 64,  # Reduced from 256 for faster Yat-Performer
    num_quadrature_nodes: int = 4,  # Reduced from 8
    use_ffn: bool = True,           # Enable/disable FFN layer
    attention_only: bool = False,    # Run only attention-testing tasks
):
    """
    Complete benchmark comparing ALL attention mechanisms:
    1. Standard Softmax (quadratic)
    2. Performer (FAVOR+ - linear approximation of softmax)
    3. Linear Attention (ELU+1 features)
    4. Cosine Linear Attention (ReLU features)
    5. Yat-Product (quadratic - exact)
    6. Spherical Yat-Performer (linear approximation of Yat)
    
    Tests include:
    - Copy Task
    - Sorting Task
    - Retrieval Task (Needle in Haystack)
    - Pattern Matching Task
    
    Benchmarks include:
    - Speed (latency)
    - Memory usage
    - Approximation quality
    
    Results are logged to a txt file.
    """
    from datetime import datetime
    import gc
    
    # Open log file
    log_lines = []
    
    def log(msg: str):
        """Print and log message"""
        print(msg)
        log_lines.append(msg)
    
    def get_memory_usage():
        """Get current GPU memory usage if available, else CPU"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def reset_memory():
        """Reset memory tracking"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log("=" * 80)
    log(f"COMPLETE ATTENTION MECHANISM BENCHMARK")
    log(f"Timestamp: {timestamp}")
    log(f"Random Features: {num_random_features}, Quadrature Nodes: {num_quadrature_nodes}")
    log("=" * 80)
    log("")
    
    embed_dim = 64
    vocab_size = 10
    batch_size = 32
    
    # Define all attention mechanisms with REDUCED features
    def create_attention_modules(dim: int, n_heads: int = 4):
        """Create attention modules using updated causal implementations from performer.py"""
        return {
            'Standard (Softmax)': StandardCausalAttention(dim, n_heads),
            'Performer (FAVOR+)': FastAttention(dim, n_heads, kernel_size=num_random_features),
            'Linear (ELU+1)': LinearCausalAttention(dim, n_heads),
            'Cosformer': CosformerCausalAttention(dim, n_heads),
            'RFF (Gaussian)': RFFCausalAttention(dim, n_heads, num_features=num_random_features),
            'Yat (Exact)': YatCausalAttention(dim, n_heads, epsilon=0.01),
            'Yat-Performer': YatPerformerCausalAttention(
                dim, n_heads,
                num_features=num_random_features,
                num_quadrature_nodes=num_quadrature_nodes,
                epsilon=0.01
            ),
            'Yat-Spherical': YatSphericalCausalAttention(dim, n_heads, epsilon=0.01),
        }
    
    # ================================================================
    # IMPROVED MODEL ARCHITECTURE
    # ================================================================
    
    class SinusoidalPositionalEncoding(nn.Module):
        """Sinusoidal positional encoding from 'Attention is All You Need'"""
        def __init__(self, dim: int, max_len: int = 1024):
            super().__init__()
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]
    
    class AttentionModel(nn.Module):
        """
        Improved model with:
        - Sinusoidal positional encodings
        - Layer normalization
        - Residual connections
        - Optional feedforward layer
        """
        def __init__(self, vocab_size: int, embed_dim: int, attention: nn.Module, 
                     use_ffn: bool = True, dropout: float = 0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoding = SinusoidalPositionalEncoding(embed_dim)
            self.attention = attention
            self.norm1 = nn.LayerNorm(embed_dim)
            self.use_ffn = use_ffn
            
            if use_ffn:
                self.ffn = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                )
                self.norm2 = nn.LayerNorm(embed_dim)
            
            self.output_proj = nn.Linear(embed_dim, vocab_size)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # Embedding + positional encoding
            h = self.embedding(x)
            h = self.pos_encoding(h)
            h = self.dropout(h)
            
            # Self-attention with residual (causal attention uses forward(x) interface)
            attn_out = self.attention(h)
            h = self.norm1(h + attn_out)
            
            # FFN with residual
            if self.use_ffn:
                ffn_out = self.ffn(h)
                h = self.norm2(h + ffn_out)
            
            # Output projection
            logits = self.output_proj(h)
            return logits
    
    # ================================================================
    # IMPROVED TASK RUNNER with better architecture
    # ================================================================
    
    def run_task_improved(task_fn, attention, num_epochs, batch_size, seq_len, vocab_size, 
                          use_improved_model: bool = True, use_ffn: bool = False):
        """Run a task with improved model architecture"""
        
        if use_improved_model:
            model = AttentionModel(vocab_size, embed_dim, attention, use_ffn=use_ffn)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        else:
            # Legacy simple model
            embedding = nn.Embedding(vocab_size, embed_dim)
            output_proj = nn.Linear(embed_dim, vocab_size)
            optimizer = torch.optim.Adam(
                list(embedding.parameters()) + 
                list(attention.parameters()) + 
                list(output_proj.parameters()),
                lr=0.001
            )
            scheduler = None
            model = None
        
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            result = task_fn(
                attention=attention,
                embedding=model.embedding if model else embedding,
                output_proj=model.output_proj if model else output_proj,
                optimizer=optimizer,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                model=model,  # Pass full model for improved path
            )
            
            if scheduler:
                scheduler.step()
            
            losses.append(result['loss'])
            accuracies.append(result['accuracy'])
        
        return losses, accuracies
    
    # ================================================================
    # TASK DEFINITIONS (Updated with model support)
    # ================================================================
    
    def run_copy_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Copy Task: Model must copy input sequence to output"""
        losses, accuracies = [], []
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs.clone()
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_sorting_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Sorting Task: Sort a sequence of numbers"""
        losses, accuracies = [], []
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.sort(inputs, dim=1)[0]
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(embedding.parameters()) + list(attention.parameters()) + list(output_proj.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_retrieval_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Retrieval Task: Find a special token in the sequence (needle in haystack)"""
        special_token = vocab_size - 1
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            # Generate sequences with random position of special token
            inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
            positions = torch.randint(0, seq_len, (batch_size,))
            
            # Insert special token at random position
            for i, pos in enumerate(positions):
                inputs[i, pos] = special_token
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            # Predict from last position
            logits = output_proj(attn_out[:, -1, :])
            
            # Target: predict the special token
            targets = torch.ones(batch_size, dtype=torch.long) * special_token
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_pattern_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Pattern Matching Task: Predict next token based on repeating pattern"""
        losses, accuracies = [], []
        pattern_length = min(4, seq_len // 2)
        
        for epoch in range(num_epochs):
            # Create repeating patterns
            patterns = torch.randint(0, vocab_size, (batch_size, pattern_length))
            # Repeat pattern to fill sequence
            repeats = (seq_len + pattern_length - 1) // pattern_length
            inputs = patterns.repeat(1, repeats)[:, :seq_len]
            
            # Target: shifted version (predict next in pattern)
            targets = torch.roll(inputs, -1, dims=1)
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    # ================================================================
    # NEW TASKS: Long-Range Dependency Tests
    # ================================================================
    
    def run_long_copy_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Long Copy Task: Copy task with longer sequence (L=256) to test long-range memory"""
        long_seq_len = 256  # Fixed longer sequence
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, long_seq_len))
            targets = inputs.clone()
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_first_token_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """First Token Retrieval: Predict the first token from position L (tests true long-range memory)"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            # Target: predict the first token
            targets = inputs[:, 0]
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            # Predict from last position
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_distant_match_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Distant Matching: Detect if first and last tokens match (binary classification)"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Randomly decide if first and last should match
            match = torch.randint(0, 2, (batch_size,))
            for i in range(batch_size):
                if match[i] == 1:
                    inputs[i, -1] = inputs[i, 0]  # Make them match
            
            targets = match  # Binary: 0=no match, 1=match
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            # Use last position for classification
            logits = output_proj(attn_out[:, -1, :])[:, :2]  # Binary output
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    # ================================================================
    # NEW TASKS: Associative Memory Tasks
    # ================================================================
    
    def run_kv_recall_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Key-Value Recall: Given pairs (K1→V1, K2→V2, ...), query K1 → V1"""
        num_pairs = min(3, seq_len // 3)
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            sequences = []
            targets_list = []
            
            for _ in range(batch_size):
                # Generate unique keys and values
                keys = torch.randperm(vocab_size)[:num_pairs]
                values = torch.randint(0, vocab_size, (num_pairs,))
                
                # Build sequence: K1, V1, K2, V2, ..., Query
                seq = torch.zeros(seq_len, dtype=torch.long)
                for i in range(num_pairs):
                    if i * 2 + 1 < seq_len - 1:
                        seq[i * 2] = keys[i]
                        seq[i * 2 + 1] = values[i]
                
                # Query: pick a random key
                query_idx = torch.randint(0, num_pairs, (1,)).item()
                seq[-1] = keys[query_idx]
                target = values[query_idx]
                
                sequences.append(seq)
                targets_list.append(target)
            
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    # ================================================================
    # NEW TASKS: Arithmetic/Counting Tasks
    # ================================================================
    
    def run_counting_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Token Counting: Count occurrences of the first token in the sequence"""
        losses, accuracies = [], []
        max_count = min(seq_len, vocab_size)  # Cap at vocab size for output
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Count occurrences of first token
            first_tokens = inputs[:, 0].unsqueeze(1)  # (batch, 1)
            counts = (inputs == first_tokens).sum(dim=1)  # (batch,)
            targets = torch.clamp(counts, max=max_count - 1)  # Cap to valid range
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_parity_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Parity Task: Predict if count of token '1' is even (0) or odd (1)"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            # Generate binary sequences (0s and 1s only)
            inputs = torch.randint(0, 2, (batch_size, seq_len))
            
            # Count 1s and compute parity
            count_ones = inputs.sum(dim=1)
            targets = count_ones % 2  # 0 if even, 1 if odd
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])[:, :2]  # Binary output
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    # ================================================================
    # NEW TASKS: Synthetic Language Tasks
    # ================================================================
    
    def run_reverse_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Reverse Sequence: Output input in reverse order"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.flip(inputs, dims=[1])  # Reverse
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_induction_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Induction Heads: [A][B]...[A] → predict [B] (key transformer capability)"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            sequences = []
            targets_list = []
            
            for _ in range(batch_size):
                # Generate sequence with induction pattern
                seq = torch.randint(0, vocab_size, (seq_len,))
                
                # Pick random positions for [A][B]...[A]
                if seq_len > 4:
                    first_pos = torch.randint(0, seq_len // 2, (1,)).item()
                    A = torch.randint(0, vocab_size, (1,)).item()
                    B = torch.randint(0, vocab_size, (1,)).item()
                    
                    seq[first_pos] = A
                    seq[first_pos + 1] = B
                    seq[-1] = A  # Query with A at the end
                    
                    target = B  # Should predict B
                else:
                    target = seq[0].item()
                
                sequences.append(seq)
                targets_list.append(target)
            
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    # ================================================================
    # NEW TASKS: Geometry-Specific Tasks (For Yat paper)
    # ================================================================
    
    def run_similarity_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Cluster Similarity: Identify which tokens are most similar to the first token"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            # Create sequences where some tokens match the first token
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Target: position of last occurrence of first token (or 0 if none)
            first_token = inputs[:, 0]
            targets = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                matches = (inputs[i, 1:] == first_token[i]).nonzero()
                if len(matches) > 0:
                    targets[i] = matches[-1].item() + 1  # +1 for offset
                else:
                    targets[i] = 0
            
            # Clamp to vocab size
            targets = torch.clamp(targets, max=vocab_size - 1)
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, 0, :])  # Query from first position
            
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    # ================================================================
    # ADDITIONAL TASKS: Extended Suite (12 new tasks)
    # ================================================================
    
    def run_very_long_copy_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Very Long Copy: Copy with L=512 to stress long-range memory"""
        long_seq_len = 512
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size // 2, long_seq_len))  # Smaller batch for memory
            targets = inputs.clone()
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_selective_copy_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Selective Copy: Copy only tokens that follow a marker token (★ = vocab_size-1)"""
        marker = vocab_size - 1
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            # Generate sequences with markers
            inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
            # Place 2-3 markers at random positions
            num_markers = min(3, seq_len // 4)
            marker_positions = torch.randperm(seq_len - 1)[:num_markers]
            for pos in marker_positions:
                inputs[:, pos] = marker
            
            # Target: tokens that follow markers (at last position)
            # Simplified: predict the token after the first marker
            targets = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                marker_pos = (inputs[i] == marker).nonzero()
                if len(marker_pos) > 0 and marker_pos[0].item() < seq_len - 1:
                    targets[i] = inputs[i, marker_pos[0].item() + 1]
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_addition_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Binary Addition: Add two 4-bit numbers, output result"""
        # Use 4-bit numbers (values 0-15), output 0-31
        max_val = min(vocab_size - 1, 8)
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            # Generate pairs of numbers
            a = torch.randint(0, max_val, (batch_size,))
            b = torch.randint(0, max_val, (batch_size,))
            targets = torch.clamp(a + b, max=vocab_size - 1)
            
            # Create sequence: [a, +, b, =] where + and = are special tokens
            inputs = torch.zeros(batch_size, 4, dtype=torch.long)
            inputs[:, 0] = a
            inputs[:, 1] = vocab_size - 2  # '+' token
            inputs[:, 2] = b
            inputs[:, 3] = vocab_size - 1  # '=' token
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_modular_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Modular Arithmetic: Compute (a + b) mod m"""
        m = vocab_size  # Modulus equals vocab size
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            a = torch.randint(0, m, (batch_size,))
            b = torch.randint(0, m, (batch_size,))
            targets = (a + b) % m
            
            # Sequence: [a, b, 0, 0, ...]
            inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
            inputs[:, 0] = a
            inputs[:, 1] = b
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_multihop_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """MultiHop Reasoning: A→B, B→C at start, query A at end → predict C"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            sequences = []
            targets_list = []
            
            for _ in range(batch_size):
                seq = torch.zeros(seq_len, dtype=torch.long)
                # Create chain: A→B, B→C
                A, B, C = torch.randperm(vocab_size)[:3]
                seq[0], seq[1] = A, B  # A→B
                seq[2], seq[3] = B, C  # B→C
                seq[-1] = A  # Query A
                
                sequences.append(seq)
                targets_list.append(C.item())  # Answer: C
            
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_stack_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Stack Operations: Simulate push/pop, predict top of stack"""
        PUSH, POP = vocab_size - 2, vocab_size - 1
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            sequences = []
            targets_list = []
            
            for _ in range(batch_size):
                seq = torch.zeros(seq_len, dtype=torch.long)
                stack = []
                
                for i in range(0, seq_len - 1, 2):
                    if len(stack) == 0 or torch.rand(1).item() > 0.3:
                        # Push
                        val = torch.randint(0, vocab_size - 2, (1,)).item()
                        seq[i] = PUSH
                        seq[i + 1] = val
                        stack.append(val)
                    else:
                        # Pop
                        seq[i] = POP
                        seq[i + 1] = 0
                        if stack:
                            stack.pop()
                
                target = stack[-1] if stack else 0
                sequences.append(seq)
                targets_list.append(target)
            
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_noisy_copy_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Noisy Copy: Copy task with random noise tokens inserted"""
        losses, accuracies = [], []
        noise_ratio = 0.3  # 30% of sequence is noise
        
        for epoch in range(num_epochs):
            # Original sequence (shorter)
            orig_len = int(seq_len * (1 - noise_ratio))
            original = torch.randint(0, vocab_size, (batch_size, orig_len))
            
            # Insert noise tokens
            inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
            noise_positions = torch.randperm(seq_len)[:seq_len - orig_len]
            
            orig_idx = 0
            for i in range(seq_len):
                if i in noise_positions:
                    inputs[:, i] = vocab_size - 1  # Noise marker
                else:
                    if orig_idx < orig_len:
                        inputs[:, i] = original[:, orig_idx]
                        orig_idx += 1
            
            # Target: copy without noise (at specific positions)
            targets = original[:, :min(orig_len, seq_len)]
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, :targets.size(1), :])
            
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_adversarial_retrieval_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Adversarial Retrieval: Find needle with many distractor needles that look similar"""
        target_token = vocab_size - 1
        distractor = vocab_size - 2  # Similar but wrong
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size - 2, (batch_size, seq_len))
            positions = torch.randint(seq_len // 4, 3 * seq_len // 4, (batch_size,))
            
            # Add true target at one position
            for i, pos in enumerate(positions):
                inputs[i, pos] = target_token
                # Add distractors
                for j in range(3):
                    dist_pos = torch.randint(0, seq_len, (1,)).item()
                    if dist_pos != pos:
                        inputs[i, dist_pos] = distractor
            
            targets = positions % vocab_size  # Position as target
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_bigram_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Bigram Prediction: Learn simple bigram statistics"""
        # Create deterministic bigram table: token i always followed by (i+1) mod vocab_size
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            # Generate sequences following bigram rule
            inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
            inputs[:, 0] = torch.randint(0, vocab_size, (batch_size,))
            for i in range(1, seq_len):
                inputs[:, i] = (inputs[:, i-1] + 1) % vocab_size
            
            # Target: next token (shifted)
            targets = (inputs + 1) % vocab_size
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_majority_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Majority Vote: Predict the most frequent token in the sequence"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Find majority token for each batch
            targets = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                counts = torch.bincount(inputs[i], minlength=vocab_size)
                targets[i] = counts.argmax()
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, -1, :])
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_compression_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Compression: Output unique tokens in order of first appearance"""
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, min(5, vocab_size), (batch_size, seq_len))  # Limited vocab for tractability
            
            # Target: unique tokens in order
            targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
            for i in range(batch_size):
                seen = set()
                unique = []
                for t in inputs[i].tolist():
                    if t not in seen:
                        seen.add(t)
                        unique.append(t)
                for j, t in enumerate(unique[:seq_len]):
                    targets[i, j] = t
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    def run_histogram_task(attention, embedding, output_proj, optimizer, num_epochs, batch_size, seq_len, vocab_size):
        """Histogram: Count frequency of each token (simplified)"""
        max_count = min(seq_len, vocab_size)
        losses, accuracies = [], []
        
        for epoch in range(num_epochs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Target: count of first token
            first_tokens = inputs[:, 0].unsqueeze(1)
            targets = torch.clamp((inputs == first_tokens).sum(dim=1), max=max_count - 1)
            
            x = embedding(inputs)
            attn_out = attention(x, x, x)
            logits = output_proj(attn_out[:, 0, :])  # Query from first position
            
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            losses.append(loss.item())
            accuracies.append(accuracy)
        
        return losses, accuracies
    
    # Task registry - ALL TASKS (25 total)
    tasks = {
        # Basic tasks (4)
        'Copy': run_copy_task,
        'Sorting': run_sorting_task,
        'Retrieval': run_retrieval_task,
        'Pattern': run_pattern_task,
        # Long-range dependency (4)
        'LongCopy': run_long_copy_task,
        'VeryLongCopy': run_very_long_copy_task,
        'FirstToken': run_first_token_task,
        'SelectiveCopy': run_selective_copy_task,
        # Matching/Memory (4)
        'DistantMatch': run_distant_match_task,
        'KVRecall': run_kv_recall_task,
        'MultiHop': run_multihop_task,
        'Stack': run_stack_task,
        # Counting/Arithmetic (4)
        'Counting': run_counting_task,
        'Parity': run_parity_task,
        'Addition': run_addition_task,
        'ModularArith': run_modular_task,
        # Synthetic language (3)
        'Reverse': run_reverse_task,
        'Induction': run_induction_task,
        'Bigram': run_bigram_task,
        # Aggregation (3)
        'Majority': run_majority_task,
        'Compression': run_compression_task,
        'Histogram': run_histogram_task,
        # Geometry/Similarity (2)
        'Similarity': run_similarity_task,
        # Stress tests (1)
        'NoisyCopy': run_noisy_copy_task,
    }
    
    # ATTENTION-ONLY TASKS: These require attention to solve, not just FFN
    # - Copy/LongCopy/VeryLongCopy: preserve information across positions
    # - Retrieval: find specific token via attention
    # - FirstToken: attend to first position
    # - KVRecall: associative retrieval via attention
    # - DistantMatch: compare distant positions
    # - Similarity: token matching via attention
    ATTENTION_ONLY_TASKS = {
        'Copy', 'LongCopy', 'VeryLongCopy', 'Retrieval', 
        'FirstToken', 'KVRecall', 'DistantMatch', 'Similarity'
    }
    
    # Filter tasks if attention_only mode
    if attention_only:
        tasks = {k: v for k, v in tasks.items() if k in ATTENTION_ONLY_TASKS}
        log(f"[ATTENTION-ONLY MODE] Running {len(tasks)} pure attention tasks")
        log(f"FFN: {'Enabled' if use_ffn else 'Disabled'}")
    
    # ================================================================
    # UNIFIED DATA GENERATION FOR ALL TASKS
    # ================================================================
    
    def generate_task_data(task_name: str, batch_size: int, seq_len: int, vocab_size: int):
        """
        Generate (inputs, targets, output_type) for each task.
        output_type: 'sequence', 'last_token', 'first_token', 'partial'
        """
        
        if task_name == 'Copy':
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs.clone()
            return inputs, targets, 'sequence'
        
        elif task_name == 'Sorting':
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.sort(inputs, dim=1)[0]
            return inputs, targets, 'sequence'
        
        elif task_name == 'Retrieval':
            special_token = vocab_size - 1
            inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
            positions = torch.randint(0, seq_len, (batch_size,))
            for i, pos in enumerate(positions):
                inputs[i, pos] = special_token
            targets = torch.ones(batch_size, dtype=torch.long) * special_token
            return inputs, targets, 'last_token'
        
        elif task_name == 'Pattern':
            pattern_length = min(4, seq_len // 2)
            patterns = torch.randint(0, vocab_size, (batch_size, pattern_length))
            repeats = (seq_len + pattern_length - 1) // pattern_length
            inputs = patterns.repeat(1, repeats)[:, :seq_len]
            targets = torch.roll(inputs, -1, dims=1)
            return inputs, targets, 'sequence'
        
        elif task_name == 'LongCopy':
            long_seq_len = 256
            inputs = torch.randint(0, vocab_size, (batch_size, long_seq_len))
            targets = inputs.clone()
            return inputs, targets, 'sequence'
        
        elif task_name == 'VeryLongCopy':
            long_seq_len = 512
            inputs = torch.randint(0, vocab_size, (batch_size // 2, long_seq_len))
            targets = inputs.clone()
            return inputs, targets, 'sequence'
        
        elif task_name == 'FirstToken':
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs[:, 0]
            return inputs, targets, 'last_token'
        
        elif task_name == 'SelectiveCopy':
            marker = vocab_size - 1
            inputs = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
            num_markers = min(3, seq_len // 4)
            marker_positions = torch.randperm(seq_len - 1)[:num_markers]
            for pos in marker_positions:
                inputs[:, pos] = marker
            targets = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                marker_pos = (inputs[i] == marker).nonzero()
                if len(marker_pos) > 0 and marker_pos[0].item() < seq_len - 1:
                    targets[i] = inputs[i, marker_pos[0].item() + 1]
            return inputs, targets, 'last_token'
        
        elif task_name == 'DistantMatch':
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            match = torch.randint(0, 2, (batch_size,))
            for i in range(batch_size):
                if match[i] == 1:
                    inputs[i, -1] = inputs[i, 0]
            targets = match
            return inputs, targets, 'last_token'
        
        elif task_name == 'KVRecall':
            num_pairs = min(3, seq_len // 3)
            sequences = []
            targets_list = []
            for _ in range(batch_size):
                keys = torch.randperm(vocab_size)[:num_pairs]
                values = torch.randint(0, vocab_size, (num_pairs,))
                seq = torch.zeros(seq_len, dtype=torch.long)
                for i in range(num_pairs):
                    if i * 2 + 1 < seq_len - 1:
                        seq[i * 2] = keys[i]
                        seq[i * 2 + 1] = values[i]
                query_idx = torch.randint(0, num_pairs, (1,)).item()
                seq[-1] = keys[query_idx]
                sequences.append(seq)
                targets_list.append(values[query_idx].item())
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            return inputs, targets, 'last_token'
        
        elif task_name == 'MultiHop':
            sequences = []
            targets_list = []
            for _ in range(batch_size):
                seq = torch.zeros(seq_len, dtype=torch.long)
                A, B, C = torch.randperm(vocab_size)[:3]
                seq[0], seq[1] = A, B
                seq[2], seq[3] = B, C
                seq[-1] = A
                sequences.append(seq)
                targets_list.append(C.item())
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            return inputs, targets, 'last_token'
        
        elif task_name == 'Stack':
            PUSH, POP = vocab_size - 2, vocab_size - 1
            sequences = []
            targets_list = []
            for _ in range(batch_size):
                seq = torch.zeros(seq_len, dtype=torch.long)
                stack = []
                for i in range(0, seq_len - 1, 2):
                    if len(stack) == 0 or torch.rand(1).item() > 0.3:
                        val = torch.randint(0, vocab_size - 2, (1,)).item()
                        seq[i] = PUSH
                        seq[i + 1] = val
                        stack.append(val)
                    else:
                        seq[i] = POP
                        seq[i + 1] = 0
                        if stack:
                            stack.pop()
                target = stack[-1] if stack else 0
                sequences.append(seq)
                targets_list.append(target)
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            return inputs, targets, 'last_token'
        
        elif task_name == 'Counting':
            max_count = min(seq_len, vocab_size)
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            first_tokens = inputs[:, 0].unsqueeze(1)
            counts = (inputs == first_tokens).sum(dim=1)
            targets = torch.clamp(counts, max=max_count - 1)
            return inputs, targets, 'last_token'
        
        elif task_name == 'Parity':
            inputs = torch.randint(0, 2, (batch_size, seq_len))
            count_ones = inputs.sum(dim=1)
            targets = count_ones % 2
            return inputs, targets, 'last_token'
        
        elif task_name == 'Addition':
            max_val = min(vocab_size - 1, 8)
            a = torch.randint(0, max_val, (batch_size,))
            b = torch.randint(0, max_val, (batch_size,))
            targets = torch.clamp(a + b, max=vocab_size - 1)
            inputs = torch.zeros(batch_size, 4, dtype=torch.long)
            inputs[:, 0] = a
            inputs[:, 1] = vocab_size - 2
            inputs[:, 2] = b
            inputs[:, 3] = vocab_size - 1
            return inputs, targets, 'last_token'
        
        elif task_name == 'ModularArith':
            m = vocab_size
            a = torch.randint(0, m, (batch_size,))
            b = torch.randint(0, m, (batch_size,))
            targets = (a + b) % m
            inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
            inputs[:, 0] = a
            inputs[:, 1] = b
            return inputs, targets, 'last_token'
        
        elif task_name == 'Reverse':
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.flip(inputs, dims=[1])
            return inputs, targets, 'sequence'
        
        elif task_name == 'Induction':
            sequences = []
            targets_list = []
            for _ in range(batch_size):
                seq = torch.randint(0, vocab_size, (seq_len,))
                if seq_len > 4:
                    first_pos = torch.randint(0, seq_len // 2, (1,)).item()
                    A = torch.randint(0, vocab_size, (1,)).item()
                    B = torch.randint(0, vocab_size, (1,)).item()
                    seq[first_pos] = A
                    seq[first_pos + 1] = B
                    seq[-1] = A
                    target = B
                else:
                    target = seq[0].item()
                sequences.append(seq)
                targets_list.append(target)
            inputs = torch.stack(sequences)
            targets = torch.tensor(targets_list)
            return inputs, targets, 'last_token'
        
        elif task_name == 'Bigram':
            inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
            inputs[:, 0] = torch.randint(0, vocab_size, (batch_size,))
            for i in range(1, seq_len):
                inputs[:, i] = (inputs[:, i-1] + 1) % vocab_size
            targets = (inputs + 1) % vocab_size
            return inputs, targets, 'sequence'
        
        elif task_name == 'Majority':
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                counts = torch.bincount(inputs[i], minlength=vocab_size)
                targets[i] = counts.argmax()
            return inputs, targets, 'last_token'
        
        elif task_name == 'Compression':
            inputs = torch.randint(0, min(5, vocab_size), (batch_size, seq_len))
            targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
            for i in range(batch_size):
                seen = set()
                unique = []
                for t in inputs[i].tolist():
                    if t not in seen:
                        seen.add(t)
                        unique.append(t)
                for j, t in enumerate(unique[:seq_len]):
                    targets[i, j] = t
            return inputs, targets, 'sequence'
        
        elif task_name == 'Histogram':
            max_count = min(seq_len, vocab_size)
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            first_tokens = inputs[:, 0].unsqueeze(1)
            targets = torch.clamp((inputs == first_tokens).sum(dim=1), max=max_count - 1)
            return inputs, targets, 'first_token'
        
        elif task_name == 'Similarity':
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            first_token = inputs[:, 0]
            targets = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                matches = (inputs[i, 1:] == first_token[i]).nonzero()
                if len(matches) > 0:
                    targets[i] = matches[-1].item() + 1
                else:
                    targets[i] = 0
            targets = torch.clamp(targets, max=vocab_size - 1)
            return inputs, targets, 'first_token'
        
        elif task_name == 'NoisyCopy':
            noise_ratio = 0.3
            orig_len = int(seq_len * (1 - noise_ratio))
            original = torch.randint(0, vocab_size, (batch_size, orig_len))
            inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
            noise_positions = set(torch.randperm(seq_len)[:seq_len - orig_len].tolist())
            orig_idx = 0
            for i in range(seq_len):
                if i in noise_positions:
                    inputs[:, i] = vocab_size - 1
                else:
                    if orig_idx < orig_len:
                        inputs[:, i] = original[:, orig_idx]
                        orig_idx += 1
            targets = original
            return inputs, targets, 'partial'
        
        else:
            # Default: copy task
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs.clone()
            return inputs, targets, 'sequence'
    # ================================================================
    # PART 1: MULTI-TASK TRAINING COMPARISON
    # ================================================================
    log("\n" + "=" * 80)
    log("PART 1: MULTI-TASK TRAINING COMPARISON")
    log("=" * 80)
    log(f"Settings: vocab_size={vocab_size}, embed_dim={embed_dim}, seq_len=64, epochs={num_epochs}")
    log(f"Tasks: {', '.join(tasks.keys())}")
    log("")
    
    all_task_results = {}
    
    for task_name, task_fn in tasks.items():
        log(f"\n{'='*40}")
        log(f"TASK: {task_name}")
        log(f"{'='*40}")
        
        task_results = {}
        attention_modules = create_attention_modules(embed_dim)
        
        for name, attention in attention_modules.items():
            log(f"\n  --- {name} ---")
            
            # Use improved model with FFN, positional encoding, layer norm
            model = AttentionModel(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                attention=attention,
                use_ffn=use_ffn,  # Controlled by --no-ffn flag
                dropout=0.1
            )
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=0.01
            )
            
            # Run task with full model forward pass
            losses, accuracies = [], []
            
            for epoch in range(num_epochs):
                # Generate task-specific inputs and targets
                inputs, targets, output_type = generate_task_data(task_name, batch_size, 64, vocab_size)
                
                # Forward pass through FULL model (embedding + pos_enc + attention + FFN + output)
                logits = model(inputs)
                
                # Compute loss based on output type
                if output_type == 'sequence':
                    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == targets).float().mean().item()
                elif output_type == 'last_token':
                    loss = F.cross_entropy(logits[:, -1, :], targets)
                    predictions = logits[:, -1, :].argmax(dim=-1)
                    accuracy = (predictions == targets).float().mean().item()
                elif output_type == 'first_token':
                    loss = F.cross_entropy(logits[:, 0, :], targets)
                    predictions = logits[:, 0, :].argmax(dim=-1)
                    accuracy = (predictions == targets).float().mean().item()
                else:  # partial sequence
                    loss = F.cross_entropy(logits[:, :targets.size(1), :].reshape(-1, vocab_size), targets.reshape(-1))
                    predictions = logits[:, :targets.size(1), :].argmax(dim=-1)
                    accuracy = (predictions == targets).float().mean().item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                losses.append(loss.item())
                accuracies.append(accuracy)
            
            task_results[name] = {
                'final_loss': losses[-1],
                'final_acc': accuracies[-1],
            }
            log(f"    Final: Loss={losses[-1]:.4f}, Acc={accuracies[-1]:.4f}")
        
        all_task_results[task_name] = task_results
    
    # Task summary table
    log("\n" + "=" * 80)
    log("MULTI-TASK RESULTS SUMMARY")
    log("=" * 80)
    log("")
    
    # Header
    header = f"{'Attention Type':<25}"
    for task_name in tasks.keys():
        header += f"{task_name:<12}"
    log(header)
    log("-" * 80)
    
    attention_names = list(create_attention_modules(embed_dim).keys())
    for name in attention_names:
        row = f"{name:<25}"
        for task_name in tasks.keys():
            acc = all_task_results[task_name][name]['final_acc']
            row += f"{acc:<12.4f}"
        log(row)
    
    # ================================================================
    # PART 2: SPEED & MEMORY BENCHMARK
    # ================================================================
    log("\n" + "=" * 80)
    log("PART 2: SPEED & MEMORY BENCHMARK")
    log("=" * 80)
    log(f"Settings: batch_size=16, dim={embed_dim}, num_trials=10")
    log("")
    
    speed_results = {name: [] for name in create_attention_modules(embed_dim).keys()}
    memory_results = {name: [] for name in create_attention_modules(embed_dim).keys()}
    
    for seq_len in seq_lens_speed:
        log(f"\n--- Sequence Length: {seq_len} ---")
        log(f"{'Attention':<25} {'Time (ms)':<12} {'Memory (MB)':<12}")
        log("-" * 50)
        
        attention_modules = create_attention_modules(embed_dim)
        
        for name, attention in attention_modules.items():
            try:
                Q = torch.randn(16, seq_len, embed_dim)
                K = torch.randn(16, seq_len, embed_dim)
                V = torch.randn(16, seq_len, embed_dim)
                
                # Warmup
                reset_memory()
                for _ in range(3):
                    _ = attention(Q, K, V)
                
                # Memory measurement
                reset_memory()
                _ = attention(Q, K, V)
                mem_usage = get_memory_usage()
                
                # Speed measurement
                start = time.time()
                for _ in range(10):
                    _ = attention(Q, K, V)
                elapsed = (time.time() - start) / 10 * 1000  # ms
                
                speed_results[name].append(elapsed)
                memory_results[name].append(mem_usage)
                log(f"{name:<25} {elapsed:<12.2f} {mem_usage:<12.2f}")
                
            except RuntimeError as e:
                # Handle OOM
                speed_results[name].append(None)
                memory_results[name].append(None)
                log(f"{name:<25} {'OOM':<12} {'OOM':<12}")
    
    # Speed & Memory summary table
    log("\n" + "-" * 80)
    log("SPEED RESULTS SUMMARY (ms)")
    log("-" * 80)
    header = f"{'Attention Type':<25}" + "".join(f"L={l:<8}" for l in seq_lens_speed)
    log(header)
    log("-" * 80)
    
    for name, times in speed_results.items():
        row = f"{name:<25}"
        for t in times:
            if t is not None:
                row += f"{t:<8.2f}"
            else:
                row += f"{'OOM':<8}"
        log(row)
    
    log("\n" + "-" * 80)
    log("MEMORY RESULTS SUMMARY (MB)")
    log("-" * 80)
    header = f"{'Attention Type':<25}" + "".join(f"L={l:<8}" for l in seq_lens_speed)
    log(header)
    log("-" * 80)
    
    for name, mems in memory_results.items():
        row = f"{name:<25}"
        for m in mems:
            if m is not None:
                row += f"{m:<8.1f}"
            else:
                row += f"{'OOM':<8}"
        log(row)
    
    # ================================================================
    # PART 3: APPROXIMATION QUALITY
    # ================================================================
    log("\n" + "=" * 80)
    log("PART 3: APPROXIMATION QUALITY")
    log("=" * 80)
    log("Comparing linear approximations to their exact counterparts")
    log("")
    
    approx_results = {
        'Performer vs Standard': {'mse': [], 'cosine': []},
        'Yat-Performer vs Yat': {'mse': [], 'cosine': []},
        'Linear (ELU+1) vs Standard': {'mse': [], 'cosine': []},
        'Linear (ReLU) vs Standard': {'mse': [], 'cosine': []},
    }
    
    for seq_len in seq_lens_approx:
        Q = torch.randn(4, seq_len, embed_dim)
        K = torch.randn(4, seq_len, embed_dim)
        V = torch.randn(4, seq_len, embed_dim)
        
        # Ground truth outputs
        standard = StandardAttention()
        standard_out = standard(Q, K, V)
        
        yat = YatAttention(epsilon=0.1)
        yat_out = yat(Q, K, V)
        
        # Approximations
        performer = PerformerAttention(embed_dim, num_features=num_random_features)
        performer_out = performer(Q, K, V)
        
        yat_perf = SphericalYatPerformer(embed_dim, num_random_features=num_random_features, 
                                          num_quadrature_nodes=num_quadrature_nodes, epsilon=0.1)
        yat_perf_out = yat_perf(Q, K, V)
        
        linear_elu = LinearAttention()
        linear_elu_out = linear_elu(Q, K, V)
        
        linear_relu = CosineLinearAttention()
        linear_relu_out = linear_relu(Q, K, V)
        
        # Compute metrics
        comparisons = [
            ('Performer vs Standard', performer_out, standard_out),
            ('Yat-Performer vs Yat', yat_perf_out, yat_out),
            ('Linear (ELU+1) vs Standard', linear_elu_out, standard_out),
            ('Linear (ReLU) vs Standard', linear_relu_out, standard_out),
        ]
        
        for name, approx_out, exact_out in comparisons:
            mse = F.mse_loss(approx_out, exact_out).item()
            cos_sim = F.cosine_similarity(
                approx_out.flatten().unsqueeze(0),
                exact_out.flatten().unsqueeze(0)
            ).item()
            approx_results[name]['mse'].append(mse)
            approx_results[name]['cosine'].append(cos_sim)
    
    # Approximation summary
    log("\n" + "-" * 80)
    log("APPROXIMATION QUALITY SUMMARY")
    log("-" * 80)
    log(f"{'Comparison':<30} {'Avg MSE':<12} {'Avg Cosine':<12} {'Quality':<15}")
    log("-" * 80)
    
    for name, data in approx_results.items():
        avg_mse = np.mean(data['mse'])
        avg_cos = np.mean(data['cosine'])
        quality = "Excellent" if avg_cos > 0.9 else ("Good" if avg_cos > 0.7 else "Moderate" if avg_cos > 0.5 else "Poor")
        log(f"{name:<30} {avg_mse:<12.6f} {avg_cos:<12.4f} {quality:<15}")
    
    # ================================================================
    # PART 4: COMPLEXITY CLASSIFICATION
    # ================================================================
    log("\n" + "=" * 80)
    log("PART 4: COMPLEXITY CLASSIFICATION")
    log("=" * 80)
    log("")
    log(f"{'Attention Type':<25} {'Time':<12} {'Memory':<12} {'Type':<25}")
    log("-" * 80)
    log(f"{'Standard (Softmax)':<25} {'O(L²d)':<12} {'O(L²)':<12} {'Quadratic (Exact)':<25}")
    log(f"{'Performer (FAVOR+)':<25} {'O(Ld·r)':<12} {'O(Lr)':<12} {'Linear (Approx Softmax)':<25}")
    log(f"{'Linear (ELU+1)':<25} {'O(Ld²)':<12} {'O(Ld)':<12} {'Linear (Kernel)':<25}")
    log(f"{'Linear (ReLU)':<25} {'O(Ld²)':<12} {'O(Ld)':<12} {'Linear (Kernel)':<25}")
    log(f"{'Yat (Exact)':<25} {'O(L²d)':<12} {'O(L²)':<12} {'Quadratic (Exact)':<25}")
    log(f"{'Yat-Performer':<25} {'O(Ld·r)':<12} {'O(Lr)':<12} {'Linear (Approx Yat)':<25}")
    log("")
    log(f"Note: r = {num_random_features} random features, {num_quadrature_nodes} quadrature nodes")
    
    # ================================================================
    # GENERATE PLOTS
    # ================================================================
    log("\n" + "=" * 80)
    log("GENERATING PLOTS")
    log("=" * 80)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Support up to 10 attention types
    
    # Plot 1: Multi-task comparison heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    task_names = list(tasks.keys())
    attention_names = list(create_attention_modules(embed_dim).keys())
    
    data_matrix = []
    for name in attention_names:
        row = [all_task_results[task][name]['final_acc'] for task in task_names]
        data_matrix.append(row)
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_yticks(np.arange(len(attention_names)))
    ax.set_xticklabels(task_names)
    ax.set_yticklabels(attention_names)
    
    # Add text annotations
    for i in range(len(attention_names)):
        for j in range(len(task_names)):
            text = ax.text(j, i, f"{data_matrix[i][j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Multi-Task Performance Comparison (Accuracy)')
    fig.colorbar(im, ax=ax, label='Accuracy')
    plt.tight_layout()
    plt.savefig('all_attention_multitask.png', dpi=150, bbox_inches='tight')
    log("✓ Saved: all_attention_multitask.png")
    
    # Plot 2: Speed comparison
    plt.figure(figsize=(12, 6))
    for i, (name, times) in enumerate(speed_results.items()):
        valid_lens = [l for l, t in zip(seq_lens_speed, times) if t is not None]
        valid_times = [t for t in times if t is not None]
        if valid_times:
            plt.plot(valid_lens, valid_times, marker='o', label=name, linewidth=2, color=colors[i])
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Speed Comparison: All Attention Mechanisms')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.savefig('all_attention_speed.png', dpi=150, bbox_inches='tight')
    log("✓ Saved: all_attention_speed.png")
    
    # Plot 3: Memory comparison
    plt.figure(figsize=(12, 6))
    for i, (name, mems) in enumerate(memory_results.items()):
        valid_lens = [l for l, m in zip(seq_lens_speed, mems) if m is not None]
        valid_mems = [m for m in mems if m is not None]
        if valid_mems:
            plt.plot(valid_lens, valid_mems, marker='s', label=name, linewidth=2, color=colors[i])
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage: All Attention Mechanisms')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig('all_attention_memory.png', dpi=150, bbox_inches='tight')
    log("✓ Saved: all_attention_memory.png")
    
    # Plot 4: Approximation quality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, (name, data) in enumerate(approx_results.items()):
        axes[0].plot(seq_lens_approx, data['mse'], marker='o', label=name, linewidth=2, color=colors[i])
        axes[1].plot(seq_lens_approx, data['cosine'], marker='o', label=name, linewidth=2, color=colors[i])
    
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Approximation Error (MSE)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Approximation Quality (Cosine Similarity)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_attention_approximation.png', dpi=150, bbox_inches='tight')
    log("✓ Saved: all_attention_approximation.png")
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    log("\n" + "=" * 80)
    log("FINAL SUMMARY")
    log("=" * 80)
    log("")
    
    # Best performer per task
    log("Best Performer per Task:")
    log("-" * 40)
    for task_name in tasks.keys():
        best = max(all_task_results[task_name].keys(), 
                   key=lambda k: all_task_results[task_name][k]['final_acc'])
        acc = all_task_results[task_name][best]['final_acc']
        log(f"  • {task_name}: {best} ({acc:.4f})")
    
    log("")
    log("Speed & Memory Leaders (at L=2048):")
    log("-" * 40)
    
    # Find fastest and most memory efficient
    if speed_results and any(speed_results[list(speed_results.keys())[0]]):
        last_valid_speed = {}
        last_valid_memory = {}
        for name in speed_results.keys():
            if speed_results[name][-1] is not None:
                last_valid_speed[name] = speed_results[name][-1]
            if memory_results[name][-1] is not None:
                last_valid_memory[name] = memory_results[name][-1]
        
        if last_valid_speed:
            fastest = min(last_valid_speed.keys(), key=lambda k: last_valid_speed[k])
            log(f"  • Fastest: {fastest} ({last_valid_speed[fastest]:.2f} ms)")
        if last_valid_memory:
            most_efficient = min(last_valid_memory.keys(), key=lambda k: last_valid_memory[k])
            log(f"  • Lowest Memory: {most_efficient} ({last_valid_memory[most_efficient]:.1f} MB)")
    
    log("")
    log("Approximation Quality:")
    log("-" * 40)
    for name, data in approx_results.items():
        avg_cos = np.mean(data['cosine'])
        log(f"  • {name}: cos={avg_cos:.4f}")
    
    log("")
    log("Recommendations:")
    log("-" * 40)
    log("• For short sequences (L < 512): Standard/Yat attention (best quality)")
    log("• For long sequences (L > 1024): Linear (ReLU) or Performer (best speed)")
    log("• For Yat-specific tasks: Yat-Performer with few features is viable")
    log("• Memory-constrained: Linear attention scales best")
    log("")
    
    # Write to file
    log("=" * 80)
    log(f"Results logged to: {log_file}")
    log("=" * 80)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    print(f"\n✓ All results saved to {log_file}")
    
    return {
        'tasks': all_task_results,
        'speed': speed_results,
        'memory': memory_results,
        'approximation': approx_results,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Performer Validation Suite')
    parser.add_argument(
        '--mode',
        type=str,
        default='compare',
        choices=['debug', 'quick', 'full', 'compare'],
        help='Mode to run: debug (implementation tests), quick (fast validation), full (all experiments), compare (all attention types with logging)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs for training experiments (default: 100)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='attention_benchmark_results.txt',
        help='Output file for benchmark results (default: attention_benchmark_results.txt)'
    )
    parser.add_argument(
        '--num-features',
        type=int,
        default=64,
        help='Number of random features for Performer/Yat-Performer (default: 64)'
    )
    parser.add_argument(
        '--num-quadrature',
        type=int,
        default=4,
        help='Number of quadrature nodes for Yat-Performer (default: 4)'
    )
    parser.add_argument(
        '--no-ffn',
        action='store_true',
        help='Disable FFN layer to test pure attention quality'
    )
    parser.add_argument(
        '--attention-only',
        action='store_true',
        help='Run only tasks that truly test attention (Copy, Retrieval, FirstToken, etc.)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'debug':
        run_all_debug_tests()
    elif args.mode == 'quick':
        run_quick_validation()
    elif args.mode == 'full':
        run_full_experiments()
    elif args.mode == 'compare':
        run_complete_attention_benchmark(
            num_epochs=args.epochs,
            log_file=args.log_file,
            num_random_features=args.num_features,
            num_quadrature_nodes=args.num_quadrature,
            use_ffn=not args.no_ffn,
            attention_only=args.attention_only
        )


if __name__ == "__main__":
    main()
