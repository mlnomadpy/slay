"""
Exact Spherical Yat attention with causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YatSphericalCausalAttention(nn.Module):
    """Exact spherical Yat attention with causal masking.

    Uses kernel:
        K(q,k) = x^2 / (C - 2x),  where x = <q̂, k̂> and C = 2 + ε.

    This module implements *kernel-normalized* attention (linear-attention style):
        Y = (K V) / (K 1)
    under a causal mask, i.e. it does not apply a softmax.

    FP32 upcast added for numerical stability in mixed-precision training.

    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        epsilon: Small constant for numerical stability
        score_scale: Optional multiplicative factor applied to K (cancels under K-normalization)
    """
    
    def __init__(self, embed_dim, n_heads, epsilon=1e-2, score_scale=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = epsilon
        self.C = 2.0 + epsilon  # Larger epsilon = more numerical headroom
        
        # Default scale is sqrt(2), stored as tensor for precision
        if score_scale is None:
            self.register_buffer('score_scale', torch.tensor(1.0))
        else:
            self.register_buffer('score_scale', torch.tensor(float(score_scale)))
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # UPCAST TO FP32 FOR NUMERICAL STABILITY
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # Normalize to unit sphere
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # x = <q̂, k̂> ∈ [-1, 1]
        x_dot = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        # Kernel: x² / (C - 2x)
        denom = torch.clamp(self.C - 2 * x_dot, min=1e-6)
        K = (x_dot ** 2) / denom

        # Optional scaling (cancels under kernel normalization)
        K = K * self.score_scale.to(K.dtype)

        # Causal mask: zero out future contributions
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        K = K.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

        # Kernel-normalized attention: (K V) / (K 1)
        numerator = torch.matmul(K, v)  # (B, H, T, D)
        denominator = K.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        out = numerator / (denominator + 1e-6)
        
        # CAST BACK TO ORIGINAL DTYPE
        out = out.to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.out(out)

