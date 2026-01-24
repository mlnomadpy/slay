"""
Random Fourier Features attention (Gaussian kernel) with causal masking.
"""

import math
import torch
import torch.nn as nn


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
        
        # Random Fourier Features
        self.register_buffer('omega', torch.randn(n_heads, self.head_dim, num_features))
        self.register_buffer('bias', torch.rand(n_heads, num_features) * 2 * math.pi)
        
    def _rff_features(self, x):
        """Compute RFF features: sqrt(2/M) * cos(x @ omega + bias)"""
        # x: (B, H, T, D), omega: (H, D, M), bias: (H, M)
        proj = torch.einsum('bhtd,hdm->bhtm', x, self.omega.float())
        proj = proj + self.bias.float().unsqueeze(0).unsqueeze(2)
        return math.sqrt(2.0 / self.num_features) * torch.cos(proj)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape: (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Upcast for stability
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # Apply RFF
        q_prime = self._rff_features(q)  # (B, H, T, M)
        k_prime = self._rff_features(k)  # (B, H, T, M)
        
        # Causal linear attention via cumsum
        # (B, H, T, M) * (B, H, T, D) -> (B, H, T, M, D)
        kv_prod = torch.einsum('bhtm,bhtd->bhtmd', k_prime, v)
        kv_cumsum = torch.cumsum(kv_prod, dim=2)
        
        # (B, H, T, M) * (B, H, T, M, D) -> (B, H, T, D)
        context = torch.einsum('bhtm,bhtmd->bhtd', q_prime, kv_cumsum)
        
        # Normalization
        k_cumsum = torch.cumsum(k_prime, dim=2)
        norm = torch.einsum('bhtm,bhtm->bht', q_prime, k_cumsum)
        
        out = context / (norm.unsqueeze(-1) + self.eps)
        out = out.to(input_dtype)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
