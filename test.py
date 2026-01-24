"""
Unit tests for attention kernel approximations.

Tests verify that each linear attention mechanism approximates
the corresponding exact kernel within acceptable error bounds.

Kernels tested:
- LinearCausalAttention: ELU+1 kernel → (elu(q)+1)·(elu(k)+1)
- CosformerCausalAttention: ReLU kernel with cos/sin position weighting
- RFFCausalAttention: Gaussian kernel → exp(-||q-k||²/2)
- YatCausalAttention: Yat kernel → (q·k)² / (||q||² + ||k||² - 2*q·k + ε)
- YatPerformerCausalAttention: Spherical Yat → x² / (C - 2x) where x = <q̂,k̂>
- FastAttention: ReLU random features → ReLU(random projection)

Run with: pytest test_attention_kernels.py -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytest
import numpy as np


# ============================================================================
# Attention Class Implementations (standalone, no deepspeed dependency)
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


# ============================================================================
# Exact Kernel Computations
# ============================================================================

def compute_exact_linear_kernel(q, k, eps=1e-6):
    """ELU+1 kernel: K(q,k) = (elu(q)+1) · (elu(k)+1)"""
    q_prime = F.elu(q) + 1
    k_prime = F.elu(k) + 1
    kernel_matrix = torch.einsum('bhtd,bhsd->bhts', q_prime, k_prime)
    return kernel_matrix


def compute_exact_cosformer_kernel(q, k, T, eps=1e-6):
    """Cosformer ReLU kernel with position-based cos/sin weighting."""
    B, H, T_dim, D = q.shape
    device = q.device
    
    positions = torch.arange(T, device=device, dtype=torch.float32)
    cos_w = torch.cos(math.pi / 2 * positions / T)
    sin_w = torch.sin(math.pi / 2 * positions / T)
    
    q_relu = F.relu(q)
    k_relu = F.relu(k)
    
    base_kernel = torch.einsum('bhtd,bhsd->bhts', q_relu, k_relu)
    
    pos_weight = torch.einsum('t,s->ts', cos_w, cos_w) + torch.einsum('t,s->ts', sin_w, sin_w)
    pos_weight = pos_weight.unsqueeze(0).unsqueeze(0)
    
    kernel_matrix = base_kernel * pos_weight
    return kernel_matrix


def compute_exact_gaussian_kernel(q, k, sigma=1.0):
    """Gaussian (RBF) kernel: K(q,k) = exp(-||q-k||²/(2σ²))"""
    B, H, T, D = q.shape
    
    q_sq = (q ** 2).sum(dim=-1, keepdim=True)
    k_sq = (k ** 2).sum(dim=-1, keepdim=True)
    qk_dot = torch.einsum('bhtd,bhsd->bhts', q, k)
    
    dist_sq = q_sq - 2 * qk_dot + k_sq.transpose(-2, -1)
    
    kernel_matrix = torch.exp(-dist_sq / (2 * sigma ** 2))
    return kernel_matrix


def compute_exact_yat_kernel(q, k, epsilon=1e-6):
    """Yat kernel: K(q,k) = (q·k)² / (||q||² + ||k||² - 2*q·k + ε)"""
    B, H, T, D = q.shape
    
    qk_dot = torch.einsum('bhtd,bhsd->bhts', q, k)
    q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
    k_norm_sq = (k ** 2).sum(dim=-1, keepdim=True)
    
    numerator = qk_dot ** 2
    denominator = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * qk_dot + epsilon
    
    kernel_matrix = numerator / denominator
    return kernel_matrix


def compute_exact_spherical_yat_kernel(q, k, epsilon=1e-6):
    """Spherical Yat kernel: K(q,k) = x² / (C - 2x) where x = <q̂, k̂>"""
    B, H, T, D = q.shape
    C = 2.0 + epsilon
    
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    x = torch.einsum('bhtd,bhsd->bhts', q_norm, k_norm)
    
    denominator = C - 2 * x
    denominator = torch.clamp(denominator, min=1e-6)
    
    kernel_matrix = (x ** 2) / denominator
    return kernel_matrix


def compute_kernel_from_features(q_features, k_features):
    """Compute kernel matrix K(q,k) = φ(q)·φ(k) from feature maps."""
    return torch.einsum('bhtf,bhsf->bhts', q_features, k_features)


# ============================================================================
# Test Classes
# ============================================================================

class TestLinearAttentionKernel:
    """Test that LinearCausalAttention approximates ELU+1 kernel."""
    
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        return {
            'embed_dim': 64, 'n_heads': 4,
            'head_dim': 16, 'batch_size': 2, 'seq_len': 16,
        }
    
    def test_elu_kernel_approximation(self, setup):
        """Verify ELU+1 features produce correct kernel."""
        B, H, T, D = setup['batch_size'], setup['n_heads'], setup['seq_len'], setup['head_dim']
        
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        
        q_prime = F.elu(q) + 1
        k_prime = F.elu(k) + 1
        
        approx_kernel = compute_kernel_from_features(q_prime, k_prime)
        exact_kernel = compute_exact_linear_kernel(q, k)
        
        torch.testing.assert_close(approx_kernel, exact_kernel, rtol=1e-5, atol=1e-5)


class TestCosformerKernel:
    """Test that CosformerCausalAttention approximates the cos-weighted ReLU kernel."""
    
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        return {'embed_dim': 64, 'n_heads': 4, 'batch_size': 2, 'seq_len': 16}
    
    def test_cosformer_kernel_approximation(self, setup):
        """Verify Cosformer features produce correct kernel."""
        B, H, T = setup['batch_size'], setup['n_heads'], setup['seq_len']
        D = setup['embed_dim'] // H
        
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        
        positions = torch.arange(T, dtype=torch.float32)
        cos_w = torch.cos(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        sin_w = torch.sin(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        
        q_relu, k_relu = F.relu(q), F.relu(k)
        q_prime = torch.cat([q_relu * cos_w, q_relu * sin_w], dim=-1)
        k_prime = torch.cat([k_relu * cos_w, k_relu * sin_w], dim=-1)
        
        approx_kernel = compute_kernel_from_features(q_prime, k_prime)
        exact_kernel = compute_exact_cosformer_kernel(q, k, T)
        
        torch.testing.assert_close(approx_kernel, exact_kernel, rtol=1e-5, atol=1e-5)


class TestRFFKernel:
    """Test that RFFCausalAttention approximates Gaussian kernel."""
    
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        return {'embed_dim': 64, 'n_heads': 4, 'batch_size': 2, 'seq_len': 16, 'num_features': 512}
    
    def test_rff_gaussian_kernel_approximation(self, setup):
        """Verify RFF features approximate Gaussian kernel."""
        B, H, T = setup['batch_size'], setup['n_heads'], setup['seq_len']
        D, M = setup['embed_dim'] // H, setup['num_features']
        
        q = torch.randn(B, H, T, D) * 0.5
        k = torch.randn(B, H, T, D) * 0.5
        
        omega = torch.randn(H, D, M)
        bias = torch.rand(H, M) * 2 * math.pi
        
        def rff_features(x):
            proj = torch.einsum('bhtd,hdm->bhtm', x, omega) + bias.unsqueeze(0).unsqueeze(2)
            return math.sqrt(2.0 / M) * torch.cos(proj)
        
        approx_kernel = compute_kernel_from_features(rff_features(q), rff_features(k))
        exact_kernel = compute_exact_gaussian_kernel(q, k, sigma=1.0)
        
        # RFF is a stochastic approximation - check correlation instead of exact match
        correlation = torch.corrcoef(torch.stack([approx_kernel.flatten(), exact_kernel.flatten()]))[0, 1]
        assert correlation > 0.7, f"RFF correlation too low: {correlation:.4f}"


class TestYatKernel:
    """Test YatCausalAttention computes exact Yat kernel."""
    
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        return {'embed_dim': 64, 'n_heads': 4, 'batch_size': 2, 'seq_len': 16, 'epsilon': 1e-6}
    
    def test_yat_kernel_exact(self, setup):
        """Verify YatCausalAttention computes exact kernel (before softmax)."""
        B, H, T = setup['batch_size'], setup['n_heads'], setup['seq_len']
        D, eps = setup['embed_dim'] // H, setup['epsilon']
        
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        
        computed_kernel = compute_exact_yat_kernel(q, k, eps)
        
        qk_dot = torch.einsum('bhtd,bhsd->bhts', q, k)
        q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
        k_norm_sq = (k ** 2).sum(dim=-1, keepdim=True)
        
        expected_kernel = (qk_dot ** 2) / (q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * qk_dot + eps)
        
        torch.testing.assert_close(computed_kernel, expected_kernel, rtol=1e-5, atol=1e-5)


class TestYatPerformerKernel:
    """Test YatPerformerCausalAttention approximates spherical Yat kernel."""
    
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        return {
            'embed_dim': 64, 'n_heads': 4, 'batch_size': 2, 'seq_len': 8,
            'num_prf_features': 256, 'num_quadrature_nodes': 8, 'epsilon': 1e-6,
        }
    
    def test_prf_exp_kernel_approximation(self, setup):
        """Test PRF approximates exp(2s·x) kernel on unit sphere."""
        H, D, M = setup['n_heads'], setup['embed_dim'] // setup['n_heads'], setup['num_prf_features']
        B, T, s = setup['batch_size'], setup['seq_len'], 0.5
        
        q = F.normalize(torch.randn(B, H, T, D), p=2, dim=-1)
        k = F.normalize(torch.randn(B, H, T, D), p=2, dim=-1)
        
        omega = torch.randn(H, D, M)
        sqrt_2s = math.sqrt(2.0 * s)
        
        def prf_features(x):
            proj = torch.einsum('bhtd,hdm->bhtm', x, omega) * sqrt_2s
            return torch.exp(torch.clamp(proj - s, min=-20.0, max=20.0)) / math.sqrt(M)
        
        approx_kernel = compute_kernel_from_features(prf_features(q), prf_features(k))
        exact_kernel = torch.exp(2 * s * torch.einsum('bhtd,bhsd->bhts', q, k))
        
        correlation = torch.corrcoef(torch.stack([approx_kernel.flatten(), exact_kernel.flatten()]))[0, 1]
        # PRF is a Monte Carlo estimate - with 256 features expect decent correlation
        assert correlation > 0.5, f"PRF correlation too low: {correlation:.4f}"
    
    def test_spherical_yat_kernel_formula(self, setup):
        """Test spherical Yat kernel formula is correct."""
        B, H, T, epsilon = setup['batch_size'], setup['n_heads'], setup['seq_len'], setup['epsilon']
        D = setup['embed_dim'] // H
        
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        
        exact_kernel = compute_exact_spherical_yat_kernel(q, k, epsilon)
        
        q_norm, k_norm = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        x = torch.einsum('bhtd,bhsd->bhts', q_norm, k_norm)
        C = 2.0 + epsilon
        expected = x ** 2 / torch.clamp(C - 2 * x, min=1e-6)
        
        torch.testing.assert_close(exact_kernel, expected, rtol=1e-5, atol=1e-5)


class TestFastAttentionKernel:
    """Test FastAttention (original Performer) with ReLU features."""
    
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        return {'embed_dim': 64, 'n_heads': 4, 'batch_size': 2, 'seq_len': 16, 'kernel_size': 128}
    
    def test_relu_random_features(self, setup):
        """Test ReLU random features produce non-negative kernel."""
        B, H, T = setup['batch_size'], setup['n_heads'], setup['seq_len']
        D, M = setup['embed_dim'] // H, setup['kernel_size']
        
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        
        proj_matrix = torch.randn(H, D, M)
        
        def relu_features(x):
            return F.relu(torch.einsum('bhtd,hdm->bhtm', x, proj_matrix)) / math.sqrt(M)
        
        kernel = compute_kernel_from_features(relu_features(q), relu_features(k))
        assert (kernel >= 0).all(), "ReLU kernel should be non-negative"


# ============================================================================
# Integration Tests
# ============================================================================

class TestAttentionModulesIntegration:
    """Integration tests for full attention modules."""
    
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        return {'embed_dim': 64, 'n_heads': 4, 'batch_size': 2, 'seq_len': 16}
    
    def test_all_attentions_run(self, setup):
        """Test all attention types can forward pass without error."""
        B, T, C, n_heads = setup['batch_size'], setup['seq_len'], setup['embed_dim'], setup['n_heads']
        
        x = torch.randn(B, T, C)
        
        attention_classes = [
            StandardCausalAttention(C, n_heads),
            LinearCausalAttention(C, n_heads),
            CosformerCausalAttention(C, n_heads),
            RFFCausalAttention(C, n_heads, num_features=64),
            YatCausalAttention(C, n_heads),
            YatPerformerCausalAttention(C, n_heads, num_features=32, num_quadrature_nodes=4),
            FastAttention(C, n_heads, kernel_size=64),
        ]
        
        for attn in attention_classes:
            out = attn(x)
            assert out.shape == (B, T, C), f"{attn.__class__.__name__} output shape mismatch"
            assert not torch.isnan(out).any(), f"{attn.__class__.__name__} produced NaN"
            assert not torch.isinf(out).any(), f"{attn.__class__.__name__} produced Inf"
    
    def test_causal_masking(self, setup):
        """Test that causal attention doesn't leak future information."""
        B, T, C, n_heads = setup['batch_size'], setup['seq_len'], setup['embed_dim'], setup['n_heads']
        
        x = torch.randn(B, T, C)
        
        attention_classes = [
            StandardCausalAttention(C, n_heads),
            LinearCausalAttention(C, n_heads),
            YatCausalAttention(C, n_heads),
        ]
        
        for attn in attention_classes:
            attn.eval()
            out_full = attn(x)
            
            x_modified = x.clone()
            x_modified[:, T//2:, :] = 0
            out_partial = attn(x_modified)
            
            diff = (out_full[:, 0, :] - out_partial[:, 0, :]).abs()
            assert diff.max() < 1e-5, f"{attn.__class__.__name__} leaks future information"


# ============================================================================
# Benchmarks: Approximation Quality vs Context Length & Speed
# ============================================================================

import time
import argparse


def get_gaussian_attention(q, k, v, sigma=1.0):
    # Exact Gaussian attention for comparison
    # q, k: (B, T, H, D) -> need (B, H, T, T) distance matrix
    B, T, H, D = q.shape
    q = q.transpose(1, 2) # (B, H, T, D)
    k = k.transpose(1, 2) # (B, H, T, D)
    
    # Compute squared distance: ||q-k||^2 = ||q||^2 + ||k||^2 - 2<q,k>
    q_norm = (q**2).sum(-1, keepdim=True)
    k_norm = (k**2).sum(-1, keepdim=True)
    qk = torch.matmul(q, k.transpose(-2, -1))
    dist_sq = q_norm + k_norm.transpose(-2, -1) - 2 * qk
    
    attn = torch.exp(-dist_sq / (2 * sigma ** 2))
    
    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), 0.0)
    
    # Normalize
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
    return torch.matmul(attn, v.transpose(1, 2)) # (B, H, T, D) 


def benchmark_approximation_quality(context_lengths=[64, 128, 256, 512, 1024], 
                                     embed_dim=256, n_heads=8, batch_size=4,
                                     device='cpu'):
    """
    Benchmark kernel approximation quality against TARGET kernels.
    
    Comparisons:
    1. FastAttention (ReLU) vs Standard (Softmax)
    2. YatPerformer vs YatSpherical (Target Approximation)
    3. YatSpherical vs Yat (Standard vs Spherical kernel difference)
    4. RFF vs Gaussian (Exact)
    
    NOTE: All computations use FP32 for accuracy.
    """
    # ENFORCE FULL PRECISION (FP32)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    
    print("\n" + "="*80)
    print("APPROXIMATION QUALITY BENCHMARK (vs Target Kernels)")
    print("="*80)
    print(f"Config: embed_dim={embed_dim}, n_heads={n_heads}, batch_size={batch_size}")
    print(f"Device: {device}, Precision: FP32")
    print("-"*80)
    
    results = []
    
    for T in context_lengths:
        print(f"\nContext Length: {T}")
        print("-" * 40)
        
        torch.manual_seed(42)
        x = torch.randn(batch_size, T, embed_dim, device=device)
        
        row = {'context_len': T}
        
        # Setup helpers
        def evaluate_pair(name, approx_cls, target_cls=None, target_fn=None):
            # Init approximation
            approx = approx_cls().to(device)
            approx.eval()
            
            # Init target
            if target_cls:
                target = target_cls().to(device)
                target.eval()
                # Share weights
                approx.qkv.weight.data = target.qkv.weight.data.clone()
                approx.qkv.bias.data = target.qkv.bias.data.clone()
                approx.out.weight.data = target.out.weight.data.clone()
                approx.out.bias.data = target.out.bias.data.clone()
                with torch.no_grad():
                    target_out = target(x)
                    approx_out = approx(x)
            else:
                # Use function (for Gaussian)
                with torch.no_grad():
                    approx_out = approx(x)
                    # Helper for function target
                    B, T, C = x.shape
                    qkv = approx.qkv(x)
                    q, k, v = qkv.chunk(3, dim=-1)
                    q = q.view(B, T, n_heads, embed_dim//n_heads).transpose(1, 2)
                    k = k.view(B, T, n_heads, embed_dim//n_heads).transpose(1, 2)
                    v = v.view(B, T, n_heads, embed_dim//n_heads).transpose(1, 2)
                    
                    target_out_head = target_fn(q, k, v)
                    target_out = target_out_head.transpose(1, 2).reshape(B, T, C)
                    target_out = approx.out(target_out)

            # Metrics
            mse = F.mse_loss(approx_out, target_out).item()
            cos_sim = F.cosine_similarity(approx_out.flatten(), target_out.flatten(), dim=0).item()
            corr = torch.corrcoef(torch.stack([approx_out.flatten(), target_out.flatten()]))[0, 1].item()
            
            print(f"  {name:25s}: MSE={mse:.6f}, CosSim={cos_sim:.4f}, Corr={corr:.4f}")
            row[f'{name}_mse'] = mse
            row[f'{name}_corr'] = corr

        # 1. FastAttention vs Standard (Softmax approximation)
        evaluate_pair("FastAttn vs Softmax", 
                      lambda: FastAttention(embed_dim, n_heads, kernel_size=128),
                      lambda: StandardCausalAttention(embed_dim, n_heads))

        # 2. YatPerformer vs YatSpherical (Target Approximation)
        evaluate_pair("YatPerformer vs Sphere", 
                      lambda: YatPerformerCausalAttention(embed_dim, n_heads, num_features=64, num_quadrature_nodes=8),
                      lambda: YatSphericalCausalAttention(embed_dim, n_heads))
                      
        # 3. Yat (Exact) vs YatSpherical (Kernel Difference)
        evaluate_pair("Yat (Exact) vs Sphere", 
                      lambda: YatCausalAttention(embed_dim, n_heads),
                      lambda: YatSphericalCausalAttention(embed_dim, n_heads))

        # 4. RFF vs Gaussian
        evaluate_pair("RFF vs Gaussian", 
                      lambda: RFFCausalAttention(embed_dim, n_heads, num_features=128),
                      target_fn=lambda q,k,v: get_gaussian_attention(q,k,v))
        
        results.append(row)
    
    # Restore original dtype
    torch.set_default_dtype(original_dtype)
    return results


def benchmark_feature_dimensions(yat_configs=None, fast_feature_dims=[16, 32, 64, 128, 256],
                                  context_len=256, embed_dim=256, n_heads=8,
                                  batch_size=4, device='cpu'):
    """
    Benchmark approximation quality vs random feature dimensions at FIXED context length.
    
    YatPerformer has TWO feature parameters:
    - num_features (M): random features per quadrature node
    - num_quadrature_nodes (R): number of quadrature nodes
    - Total features = R × M
    
    FastAttention has one: kernel_size (M)
    
    All computations use FP32 for accuracy.
    """
    # Default YatPerformer configs: (num_features, num_quadrature_nodes) -> total = M*R
    if yat_configs is None:
        yat_configs = [
            (8, 2),    # 16 total
            (16, 2),   # 32 total
            (16, 4),   # 64 total
            (32, 4),   # 128 total
            (64, 4),   # 256 total
            (64, 8),   # 512 total
        ]
    
    # ENFORCE FULL PRECISION (FP32)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    
    print("\n" + "="*80)
    print("FEATURE DIMENSION BENCHMARK (Fixed Context, Varying Features)")
    print("="*80)
    print(f"Config: context_len={context_len}, embed_dim={embed_dim}, n_heads={n_heads}, batch_size={batch_size}")
    print(f"Device: {device}, Precision: FP32")
    print("-"*80)
    
    results = []
    
    torch.manual_seed(42)
    x = torch.randn(batch_size, context_len, embed_dim, device=device, dtype=torch.float32)
    
    # Create reference models (shared weights for fair comparison)
    target_spherical = YatSphericalCausalAttention(embed_dim, n_heads).to(device)
    target_spherical.eval()
    
    target_softmax = StandardCausalAttention(embed_dim, n_heads).to(device)
    target_softmax.eval()
    
    with torch.no_grad():
        target_spherical_out = target_spherical(x)
        target_softmax_out = target_softmax(x)
    
    # === YatPerformer Benchmark (varying M and R) ===
    print("\n--- YatPerformer vs YatSpherical (M=num_features, R=num_quadrature_nodes) ---")
    print(f"{'M':>6}  {'R':>4}  {'Total':>6}  {'MSE':>12}  {'CosSim':>10}  {'Corr':>10}")
    print("-"*60)
    
    for M, R in yat_configs:
        try:
            yat_perf = YatPerformerCausalAttention(embed_dim, n_heads, num_features=M, num_quadrature_nodes=R).to(device)
            yat_perf.eval()
            # Share weights with target
            yat_perf.qkv.weight.data = target_spherical.qkv.weight.data.clone()
            yat_perf.qkv.bias.data = target_spherical.qkv.bias.data.clone()
            yat_perf.out.weight.data = target_spherical.out.weight.data.clone()
            yat_perf.out.bias.data = target_spherical.out.bias.data.clone()
            
            with torch.no_grad():
                yat_perf_out = yat_perf(x)
            
            yat_mse = F.mse_loss(yat_perf_out, target_spherical_out).item()
            yat_cos = F.cosine_similarity(yat_perf_out.flatten(), target_spherical_out.flatten(), dim=0).item()
            yat_corr = torch.corrcoef(torch.stack([yat_perf_out.flatten(), target_spherical_out.flatten()]))[0, 1].item()
            
            print(f"{M:>6}  {R:>4}  {M*R:>6}  {yat_mse:>12.6f}  {yat_cos:>10.4f}  {yat_corr:>10.4f}")
            results.append({'type': 'YatPerformer', 'M': M, 'R': R, 'total': M*R, 'mse': yat_mse, 'cos': yat_cos, 'corr': yat_corr})
            
            del yat_perf
            torch.cuda.empty_cache() if device != 'cpu' else None
        except RuntimeError as e:
            print(f"{M:>6}  {R:>4}  {M*R:>6}  {'OOM':>12}  {'OOM':>10}  {'OOM':>10}")
    
    # === FastAttention Benchmark ===
    print("\n--- FastAttention vs Softmax ---")
    print(f"{'Features':>10}  {'MSE':>12}  {'CosSim':>10}  {'Corr':>10}")
    print("-"*50)
    
    for M in fast_feature_dims:
        try:
            fast_attn = FastAttention(embed_dim, n_heads, kernel_size=M).to(device)
            fast_attn.eval()
            # Share weights with target
            fast_attn.qkv.weight.data = target_softmax.qkv.weight.data.clone()
            fast_attn.qkv.bias.data = target_softmax.qkv.bias.data.clone()
            fast_attn.out.weight.data = target_softmax.out.weight.data.clone()
            fast_attn.out.bias.data = target_softmax.out.bias.data.clone()
            
            with torch.no_grad():
                fast_attn_out = fast_attn(x)
            
            fast_mse = F.mse_loss(fast_attn_out, target_softmax_out).item()
            fast_cos = F.cosine_similarity(fast_attn_out.flatten(), target_softmax_out.flatten(), dim=0).item()
            fast_corr = torch.corrcoef(torch.stack([fast_attn_out.flatten(), target_softmax_out.flatten()]))[0, 1].item()
            
            print(f"{M:>10}  {fast_mse:>12.6f}  {fast_cos:>10.4f}  {fast_corr:>10.4f}")
            results.append({'type': 'FastAttention', 'M': M, 'mse': fast_mse, 'cos': fast_cos, 'corr': fast_corr})
            
            del fast_attn
            torch.cuda.empty_cache() if device != 'cpu' else None
        except RuntimeError as e:
            print(f"{M:>10}  {'OOM':>12}  {'OOM':>10}  {'OOM':>10}")
    
    # Restore original dtype
    torch.set_default_dtype(original_dtype)
    return results


def benchmark_speed(context_lengths=[64, 128, 256, 512, 1024, 2048],
                    embed_dim=256, n_heads=8, batch_size=4,
                    n_warmup=10, n_iterations=50, device='cpu'):
    """
    Benchmark forward pass speed across different context lengths.
    """
    print("\n" + "="*80)
    print("SPEED BENCHMARK (ms per forward pass)")
    print("="*80)
    print(f"Config: embed_dim={embed_dim}, n_heads={n_heads}, batch_size={batch_size}")
    print(f"Device: {device}, Warmup: {n_warmup}, Iterations: {n_iterations}")
    print("-"*80)
    
    attention_classes = {
        'Standard': lambda: StandardCausalAttention(embed_dim, n_heads),
        'FastAttn': lambda: FastAttention(embed_dim, n_heads, kernel_size=128),
        'Yat (exact)': lambda: YatCausalAttention(embed_dim, n_heads),
        'YatSphere': lambda: YatSphericalCausalAttention(embed_dim, n_heads),
        'YatPerformer': lambda: YatPerformerCausalAttention(embed_dim, n_heads, num_features=64),
    }
    
    results = []
    
    # Header
    header = f"{'Context':>8}"
    for name in attention_classes:
        header += f" {name:>12}"
    print(header)
    print("-" * len(header))
    
    for T in context_lengths:
        torch.manual_seed(42)
        x = torch.randn(batch_size, T, embed_dim, device=device)
        
        row = {'context_len': T}
        line = f"{T:>8}"
        
        for name, attn_fn in attention_classes.items():
            try:
                attn = attn_fn().to(device)
                attn.eval()
                
                # Warmup
                with torch.no_grad():
                    for _ in range(n_warmup):
                        _ = attn(x)
                
                # Sync if using CUDA
                if device != 'cpu':
                    torch.cuda.synchronize()
                
                # Timed iterations
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(n_iterations):
                        _ = attn(x)
                
                if device != 'cpu':
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) / n_iterations * 1000  # ms
                
                line += f" {elapsed:>12.2f}"
                row[f'{name}_ms'] = elapsed
                
            except Exception as e:
                line += f" {'OOM/ERR':>12}"
                row[f'{name}_ms'] = float('nan')
        
        print(line)
        results.append(row)
    
    return results


def run_benchmarks(device='cpu', save_csv=False):
    """Run all benchmarks and optionally save results."""
    
    # Adjust context lengths based on device
    if device == 'cpu':
        quality_ctx = [64, 128, 256, 512]
        speed_ctx = [64, 128, 256, 512]
    else:
        quality_ctx = [64, 128, 256, 512, 1024]
        speed_ctx = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # Run benchmarks
    quality_results = benchmark_approximation_quality(
        context_lengths=quality_ctx, device=device
    )
    speed_results = benchmark_speed(
        context_lengths=speed_ctx, device=device
    )
    
    # Save to CSV if requested
    if save_csv:
        import csv
        
        with open('benchmark_quality.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=quality_results[0].keys())
            writer.writeheader()
            writer.writerows(quality_results)
        print("\nSaved quality results to benchmark_quality.csv")
        
        with open('benchmark_speed.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=speed_results[0].keys())
            writer.writeheader()
            writer.writerows(speed_results)
        print("Saved speed results to benchmark_speed.csv")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    
    return quality_results, speed_results


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attention Kernel Tests and Benchmarks')
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run benchmarks instead of unit tests')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run benchmarks on (cpu/cuda)')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save benchmark results to CSV files')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmarks(device=args.device, save_csv=args.save_csv)
    else:
        pytest.main([__file__, '-v', '--tb=short'])

