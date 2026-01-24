"""
Linear attention with ELU+1 kernel and causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCausalAttention(nn.Module):
    """ELU+1 linear attention with causal masking via cumsum."""
    
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
        
        # Reshape: (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Upcast for stability
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # ELU+1 kernel
        q_prime = F.elu(q) + 1
        k_prime = F.elu(k) + 1
        
        # Causal linear attention via cumsum
        # (B, H, T, D) * (B, H, T, D) -> (B, H, T, D, D)
        kv_prod = torch.einsum('bhtd,bhte->bhtde', k_prime, v)
        kv_cumsum = torch.cumsum(kv_prod, dim=2)
        
        # (B, H, T, D) * (B, H, T, D, D) -> (B, H, T, D)
        context = torch.einsum('bhtd,bhtde->bhte', q_prime, kv_cumsum)
        
        # Normalization
        k_cumsum = torch.cumsum(k_prime, dim=2)
        norm = torch.einsum('bhtd,bhtd->bht', q_prime, k_cumsum)
        
        out = context / (norm.unsqueeze(-1) + self.eps)
        out = out.to(input_dtype)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
