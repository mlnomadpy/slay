"""
Performer-style Linear Attention (FAVOR+ approximation).

Uses ReLU-based random feature maps for O(L) complexity.
"""

import math
import torch
import torch.nn as nn


class FastAttention(nn.Module):
    """
    Performer-style Linear Attention (FAVOR+ approximation).
    Uses ReLU-based random feature maps for O(L) complexity.
    """
    
    def __init__(self, embed_dim, n_heads, kernel_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.kernel_size = kernel_size
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        # Frozen Random Projection Matrix (Gaussian)
        self.register_buffer(
            'proj_matrix', 
            torch.randn(n_heads, self.head_dim, kernel_size) / math.sqrt(self.head_dim)
        )
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q = q.float()
        k = k.float()
        v = v.float()
        proj_matrix = self.proj_matrix.float()

        # 1. Kernel Feature Map: phi(x) = ReLU(x @ W)
        q_prime = torch.relu(torch.einsum('bhtd,hdm->bhtm', q, proj_matrix))
        k_prime = torch.relu(torch.einsum('bhtd,hdm->bhtm', k, proj_matrix))
        
        # 2. Causal Linear Attention (Prefix Sums)
        k_cumsum = torch.cumsum(k_prime, dim=2)
        
        kv_prod = torch.einsum('bhtm,bhtd->bhtmd', k_prime, v)
        kv_cumsum = torch.cumsum(kv_prod, dim=2)
        
        context = torch.einsum('bhtm,bhtmd->bhtd', q_prime, kv_cumsum)
        
        norm = torch.einsum('bhtm,bhtm->bht', q_prime, k_cumsum)
        
        out = context / (norm.unsqueeze(-1) + 1e-6)
        out = out.to(input_dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
