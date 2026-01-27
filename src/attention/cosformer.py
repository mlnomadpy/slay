"""
Cosformer attention with cos-based reweighting and causal masking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Reshape: (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Upcast for stability
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # Position weighting
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        cos_w = torch.cos(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        sin_w = torch.sin(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        
        # ReLU kernel with cos/sin weighting
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
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
    
    def forward_triton(self, x):
        """
        Forward pass using Triton-accelerated linear attention.
        
        Processes cos and sin streams using Triton kernel and sums results.
        Falls back to regular forward() if Triton is not available.
        """
        try:
            from .yat_attention_kernel import HAS_TRITON, triton_linear_attention
        except ImportError:
            try:
                from yat_attention_kernel import HAS_TRITON, triton_linear_attention
            except ImportError:
                return self.forward(x)
        
        if not HAS_TRITON or not torch.cuda.is_available():
            return self.forward(x)
        
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # Position weighting
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        cos_w = torch.cos(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        sin_w = torch.sin(math.pi / 2 * positions / T).view(1, 1, -1, 1)
        
        # ReLU features with cos/sin weighting
        q_prime = F.relu(q)
        k_prime = F.relu(k)
        
        q_cos = (q_prime * cos_w).contiguous()
        q_sin = (q_prime * sin_w).contiguous()
        k_cos = (k_prime * cos_w).contiguous()
        k_sin = (k_prime * sin_w).contiguous()
        
        # Triton kernel for cos stream
        out_cos = triton_linear_attention(q_cos, k_cos, v, delta=self.eps)
        
        # Triton kernel for sin stream
        out_sin = triton_linear_attention(q_sin, k_sin, v, delta=self.eps)
        
        # Combine streams
        out = out_cos + out_sin
        
        out = out.to(input_dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
