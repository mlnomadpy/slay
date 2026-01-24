"""
Spherical Yat-Performer: Linearized Yat attention (FAST).

Optimizations:
1. Hadamard product (element-wise) instead of tensor product: M features vs M×M
2. Vectorized PRF computation
3. Larger chunk size (512)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class YatPerformerCausalAttention(nn.Module):
    """Spherical ⵟ-Performer: Linearized Yat attention (FAST).
    
    Optimizations:
    1. Hadamard product (element-wise) instead of tensor product: M features vs M×M
    2. Vectorized PRF computation
    3. Larger chunk size (512)
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
        nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        nodes = torch.tensor(nodes, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer('quad_nodes', nodes / self.C)
        self.register_buffer('quad_weights', weights / self.C)
        
        # SHARED omega for both poly and PRF features (enables Hadamard product)
        # Shape: (R, H, D, M) - one omega per quadrature node
        self.register_buffer('omega', torch.randn(num_quadrature_nodes, n_heads, self.head_dim, num_features))
    
    def _compute_features_fast(self, x):
        """Compute features using Hadamard product: φ_poly ⊙ φ_PRF.
        
        Instead of tensor product (M×M dims), we use element-wise product (M dims).
        This is valid because we share the random projection ω.
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
