"""
Transformer block implementations.
"""

import torch.nn as nn

from ..attention import FastAttention
from ..activations import novel_activation


class GPT2Block(nn.Module):
    """Standard GPT-2 style transformer block with LayerNorm and GELU."""
    
    def __init__(self, embed_dim, n_heads, attention_class=FastAttention, use_triton=False, dropout=0.1):
        super().__init__()
        self.use_triton = use_triton
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = attention_class(embed_dim, n_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture (GPT-2 style)
        # Use Triton-accelerated attention if enabled and available
        if self.use_triton and hasattr(self.attn, 'forward_triton'):
            x = x + self.attn_dropout(self.attn.forward_triton(self.ln1(x)))
        else:
            x = x + self.attn_dropout(self.attn(self.ln1(x)))
        x = x + self.mlp_dropout(self.mlp(self.ln2(x)))
        return x


class NovelBlock(nn.Module):
    """Novel transformer block with Yat-style activation (for Yat attention variants).
    No LayerNorm - novel activation provides implicit normalization.
    """
    
    def __init__(self, embed_dim, n_heads, attention_class=FastAttention, use_triton=False, dropout=0.1):
        super().__init__()
        self.use_triton = use_triton
        self.attn = attention_class(embed_dim, n_heads)
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Attention residual - use Triton if enabled
        if self.use_triton and hasattr(self.attn, 'forward_triton'):
            x = x + self.attn_dropout(self.attn.forward_triton(x))
        else:
            x = x + self.attn_dropout(self.attn(x))
        
        # MLP with novel activation
        h_proj = self.fc1(x)
        # Novel activation: (x·w)² / (||x - w||² + eps)
        activated = novel_activation(x, h_proj)
        x = x + self.mlp_dropout(self.fc2(activated))
        return x

