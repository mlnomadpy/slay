"""
Yat-product attention (exact) with causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YatCausalAttention(nn.Module):
    """Yat-product attention (exact) with causal masking.
    
    Uses kernel: (q·k)² / (||q||² + ||k||² - 2*q·k + eps)
    
    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        epsilon: Small constant for numerical stability
        score_scale: Scale factor for attention scores (default: sqrt(2))
    """
    
    def __init__(self, embed_dim, n_heads, epsilon=1e-6, score_scale=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = epsilon
        
        # Default scale is 1, stored as tensor for precision
        if score_scale is None:
            self.register_buffer('score_scale', torch.tensor(1.0))
        else:
            self.register_buffer('score_scale', torch.tensor(float(score_scale)).sqrt())
        
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
        
        # Compute dot product: q·k
        dot_product = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        
        # Compute squared norms
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        
        # Yat kernel: (q·k)² / (||q||² + ||k||² - 2*q·k + eps)
        # ||q||² + ||k||² broadcasted: (B, H, T, 1) + (B, H, 1, T) -> (B, H, T, T)
        numerator = dot_product ** 2
        denominator = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * dot_product + self.epsilon
        
        scores = numerator / denominator
        
        # Apply score scaling (uses tensor for correct dtype/device)
        scores = scores * self.score_scale.to(scores.dtype)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

