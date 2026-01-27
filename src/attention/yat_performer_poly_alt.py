"""
Spherical Yat-Performer: alternative polynomial kernel approximations.

Implements three variants for the polynomial kernel (x·y)^2:
- Random Maclaurin (RM)
- Nyström approximation
- Anchor (low-rank) features

All variants use the same PRF exponential features and chunked causal
linear attention, and avoid materializing the full tensor-product features except in a gated,
memory-safe vectorized path for short sequences.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _YatPerformerPolyBase(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads,
        num_prf_features=8,
        num_quadrature_nodes=1,
        poly_dim=64,
        epsilon=1e-6,
        chunk_size=256,
        nystrom_reg=1e-3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.num_prf_features = num_prf_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.poly_dim = poly_dim
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.chunk_size = chunk_size
        self.nystrom_reg = nystrom_reg

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        # Gauss-Laguerre quadrature nodes and weights
        nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        nodes = torch.tensor(nodes, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("quad_nodes", nodes / self.C)
        self.register_buffer("quad_weights", weights / self.C)

        # PRF random projections: (R, H, D, M)
        self.register_buffer(
            "omega",
            torch.randn(num_quadrature_nodes, n_heads, self.head_dim, num_prf_features),
        )

    def _prf_features(self, x):
        R = self.num_quadrature_nodes
        M = self.num_prf_features

        omega = self.omega.to(dtype=x.dtype, device=x.device)
        proj = torch.einsum("bhtd,rhdm->rbhtm", x, omega)

        sqrt_2s = torch.sqrt(2.0 * self.quad_nodes.clamp(min=0)).view(R, 1, 1, 1, 1)
        s_vals = self.quad_nodes.view(R, 1, 1, 1, 1)

        exp_arg = torch.clamp(proj * sqrt_2s - s_vals, min=-20.0, max=20.0)
        prf = torch.exp(exp_arg) / math.sqrt(M)
        return prf

    def _compute_chunk_features(self, x_chunk):
        x_norm = F.normalize(x_chunk, p=2, dim=-1)
        poly_feat = self._poly_features(x_norm)
        prf_feat = self._prf_features(x_norm)
        return poly_feat, prf_feat

    def _poly_features(self, x_norm):
        raise NotImplementedError

    def _can_use_vectorized(self, B, T, dtype, device):
            # Conservative safety margin
            SAFETY = 0.4
        
            bytes_per_elem = torch.tensor(0, dtype=dtype).element_size()
        
            R = self.num_quadrature_nodes
            H = self.n_heads
            P = self.poly_dim
            M = self.num_prf_features
            D = self.head_dim
        
            # Size of the largest tensor: kv
            estimated_bytes = (
                R * B * H * T * P * M * D * bytes_per_elem
            )
        
            total_mem = torch.cuda.get_device_properties(device).total_memory
        
            return estimated_bytes < SAFETY * total_mem

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        input_dtype = q.dtype
        
        if q.dtype in (torch.float16, torch.bfloat16):
            q, k, v = q, k, v
        else:
            q, k, v = q.float(), k.float(), v.float()

        R = self.num_quadrature_nodes
        P = self.poly_dim
        M = self.num_prf_features
            
        # --- Vectorized vs Chunked Causal Linear Attention ---
        # For sequence lengths where the full state fits in memory (e.g. T=8192 on A100), 
        # vectorized execution is much faster than chunking.
        if self._can_use_vectorized(B, T, q.dtype, x.device):
            # Helper to get combined features for full sequence
            def get_full_features(x_in):
                x_norm = F.normalize(x_in, p=2, dim=-1)
                poly = self._poly_features(x_norm) # (B, H, T, P)
                prf = self._prf_features(x_norm)   # (R, B, H, T, M)
                # Outer product: (R, B, H, T, P, M)
                return torch.einsum("bhtp,rbhtm->rbhtpm", poly, prf)

            q_outer = get_full_features(q)
            k_outer = get_full_features(k) # (R, B, H, T, P, M)
            
            # KV projection: (R, B, H, T, P, M, D)
            # Memory check: For T=1024, H=12, P=64, M=8, D=64 -> ~1.5GB. Fine.
            kv = torch.einsum("rbhtpm,bhtd->rbhtpmd", k_outer, v)
            
            # Causal cumsum along T (dim 3)
            kv_cumsum = torch.cumsum(kv, dim=3)
            k_cumsum = torch.cumsum(k_outer, dim=3)
            
            # Attention
            context = torch.einsum("rbhtpm,rbhtpmd->bhtd", q_outer, kv_cumsum)
            norm = torch.einsum("rbhtpm,rbhtpm->bht", q_outer, k_cumsum)
            
            norm = torch.clamp(norm, min=1e-6)
            
            out = context / norm.unsqueeze(-1)

        else:
            # Chunked chunked causal linear attention
            chunk_size = self.chunk_size
            num_chunks = (T + chunk_size - 1) // chunk_size
    
            kv_state = torch.zeros(
                R,
                B,
                self.n_heads,
                P,
                M,
                self.head_dim,
                device=x.device,
                dtype=q.dtype,
            )
            k_state = torch.zeros(
                R,
                B,
                self.n_heads,
                P,
                M,
                device=x.device,
                dtype=q.dtype,
            )
    
            out_chunks = []
    
            for i in range(num_chunks):
                st = i * chunk_size
                ed = min(st + chunk_size, T)
    
                q_chunk = q[:, :, st:ed]
                k_chunk = k[:, :, st:ed]
                v_chunk = v[:, :, st:ed]
    
                q_poly, q_prf = self._compute_chunk_features(q_chunk)
                k_poly, k_prf = self._compute_chunk_features(k_chunk)
    
                # Apply quadrature weights (sqrt) to PRF features
                sq_weights = torch.sqrt(self.quad_weights.clamp(min=0)).view(R, 1, 1, 1, 1)
                q_prf = q_prf * sq_weights
                k_prf = k_prf * sq_weights
    
                # Tensor product: (R, B, H, T, P, M)
                q_outer = torch.einsum("bhtp,rbhtm->rbhtpm", q_poly, q_prf)
                k_outer = torch.einsum("bhtp,rbhtm->rbhtpm", k_poly, k_prf)
    
                kv_chunk_prod = torch.einsum("rbhtpm,bhtd->rbhtpmd", k_outer, v_chunk)
                kv_local_cumsum = torch.cumsum(kv_chunk_prod, dim=3)
                k_local_cumsum = torch.cumsum(k_outer, dim=3)
    
                kv_local_cumsum += kv_state.unsqueeze(3)
                kv_current = kv_local_cumsum
                
                k_local_cumsum += k_state.unsqueeze(3)
                k_current = k_local_cumsum
    
                context_chunk = torch.einsum(
                "rbhtpm,rbhtpmd->bhtd", q_outer, kv_current
                )
            
                denom_chunk = torch.einsum(
                    "rbhtpm,rbhtpm->bht", q_outer, k_current
                )
            
                denom_chunk = torch.clamp(denom_chunk, min=1e-6)
                out_chunk = context_chunk / denom_chunk.unsqueeze(-1)
                out_chunks.append(out_chunk)
    
                kv_state = kv_current[:, :, :, -1]
                k_state = k_current[:, :, :, -1]
    
            out = torch.cat(out_chunks, dim=2)
            
        out = out.to(input_dtype)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class YatPerformerRMCausalAttention(_YatPerformerPolyBase):
    """Random Maclaurin polynomial features for (x·y)^2.
    
    Uses the identity: E[(r·x)(r·y)] = x·y for Rademacher r
    For (x·y)^2, we use: phi(x) = (r1·x)(r2·x) / sqrt(D)
    where r1, r2 are independent Rademacher vectors.
    Then E[phi(x)·phi(y)] = (x·y)^2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim

        # Use Gaussian instead of Rademacher for lower variance
        # Scale by 1/sqrt(D) for each random vector
        r1 = torch.randn(P, D) / math.sqrt(D)
        r2 = torch.randn(P, D) / math.sqrt(D)
        self.register_buffer("rm_r1", r1)
        self.register_buffer("rm_r2", r2)

    def _poly_features(self, x_norm):
        # x_norm: (B, H, T, D) - already normalized to unit sphere
        r1 = self.rm_r1.to(x_norm.dtype)
        r2 = self.rm_r2.to(x_norm.dtype)

        # (r1·x) and (r2·x) each have variance ~1 for unit x
        proj1 = torch.einsum("bhtd,pd->bhtp", x_norm, r1)
        proj2 = torch.einsum("bhtd,pd->bhtp", x_norm, r2)
        
        # Product gives features for (x·y)^2
        # Scale by 1/sqrt(P) to normalize the sum
        return (proj1 * proj2) / math.sqrt(self.poly_dim)


class YatPerformerNystromCausalAttention(_YatPerformerPolyBase):
    """Nyström approximation for the polynomial kernel (x·y)^2.
    
    Uses normalized anchor points on the unit sphere to match
    the normalized query/key vectors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim

        # Anchors must be normalized to unit sphere (same as inputs)
        anchors = torch.randn(P, D)
        anchors = F.normalize(anchors, p=2, dim=-1)
        
        # Kernel matrix between anchors: K(a_i, a_j) = (a_i · a_j)^2
        K = (anchors @ anchors.t()) ** 2
        K = K + self.nystrom_reg * torch.eye(P)
        
        # Compute K^{-1/2} for Nyström features
        eigvals, eigvecs = torch.linalg.eigh(K)
        eigvals = torch.clamp(eigvals, min=1e-6)
        W = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.t()

        self.register_buffer("nystrom_anchors", anchors)
        self.register_buffer("nystrom_W", W)

    def _poly_features(self, x_norm):
        # x_norm: (B, H, T, D) - normalized to unit sphere
        anchors = self.nystrom_anchors.to(x_norm.dtype)
        W = self.nystrom_W.to(x_norm.dtype)
        
        # K(x, anchors) = (x · a_i)^2
        K_xA = (torch.einsum("bhtd,pd->bhtp", x_norm, anchors)) ** 2
        
        # Nyström feature: phi(x) = K(x, A) @ W
        return torch.einsum("bhtp,pq->bhtq", K_xA, W) / math.sqrt(self.poly_dim)


class YatPerformerAnchorCausalAttention(_YatPerformerPolyBase):
    """Low-rank anchor features for the polynomial kernel (x·y)^2.
    
    Uses normalized anchor vectors on the unit sphere. The feature map is:
        phi(x) = [(a_1·x)^2, (a_2·x)^2, ..., (a_P·x)^2] / sqrt(P)
    
    For normalized x, y: phi(x)·phi(y) approximates a function of (x·y).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim
        
        # Normalize anchors to unit sphere for consistency with inputs
        anchors = torch.randn(P, D)
        anchors = F.normalize(anchors, p=2, dim=-1)
        self.register_buffer("anchor_vectors", anchors)

    def _poly_features(self, x_norm):
        # x_norm: (B, H, T, D) - normalized to unit sphere
        anchors = self.anchor_vectors.to(dtype=x_norm.dtype, device=x_norm.device)
        
        # Feature: (a_p · x)^2 for each anchor
        return (torch.einsum("bhtd,pd->bhtp", x_norm, anchors) ** 2) / math.sqrt(
            self.poly_dim
        )
