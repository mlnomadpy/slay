"""
Spherical Yat-Performer (Laplace-only): Linearized Yat attention using
only exponential PRF features under the Laplace-mixture proposition.

This variant tests the claim that the polynomial factor can be absorbed
into the Laplace measure, yielding a pure exponential-kernel mixture.
It uses modified quadrature weights and does NOT construct polynomial
features. This is an experimental baseline for validation.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YatPerformerLaplaceCausalAttention(nn.Module):
    """Experimental Laplace-only Yat attention (causal).

    Uses only PRF features with modified quadrature weights:
        w_r = (C^2 / 4) * alpha_r / C = (C / 4) * alpha_r

    This directly tests the Laplace-only proposition and should be
    compared empirically against tensor-product and Hadamard baselines.
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        num_features=32,
        num_quadrature_nodes=2,
        epsilon=1e-6,
        chunk_size=512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.num_features = num_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        # Gauss-Laguerre quadrature nodes and weights
        nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        nodes = torch.tensor(nodes, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("quad_nodes", nodes / self.C)

        # Laplace-only weights: (C^2/4) * (alpha_r / C) = (C/4) * alpha_r
        laplace_weights = weights * (self.C / 4.0)
        self.register_buffer("quad_weights", laplace_weights)

        # PRF random projections: (R, H, D, M)
        self.register_buffer(
            "omega",
            torch.randn(num_quadrature_nodes, n_heads, self.head_dim, num_features),
        )

    def _compute_features(self, x):
        """Compute Laplace-only PRF features.

        Returns:
            features: (B, H, T, R*M)
        """
        B, H, T, D = x.shape
        R = self.num_quadrature_nodes
        M = self.num_features

        x_norm = F.normalize(x, p=2, dim=-1)
        omega = self.omega.float()

        proj = torch.einsum("bhtd,rhdm->rbhtm", x_norm, omega)

        sqrt_2s = torch.sqrt(2.0 * self.quad_nodes.clamp(min=0)).view(R, 1, 1, 1, 1)
        s_vals = self.quad_nodes.view(R, 1, 1, 1, 1)

        exp_arg = torch.clamp(proj * sqrt_2s - s_vals, min=-20.0, max=20.0)
        prf_feat = torch.exp(exp_arg) / math.sqrt(M)

        sq_weights = torch.sqrt(self.quad_weights.clamp(min=0)).view(R, 1, 1, 1, 1)
        prf_feat = prf_feat * sq_weights

        fused = prf_feat.permute(1, 2, 3, 0, 4).reshape(B, H, T, R * M)
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

        q_features = self._compute_features(q)
        k_features = self._compute_features(k)

        feat_dim = q_features.shape[-1]
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size

        kv_state = torch.zeros(B, self.n_heads, feat_dim, self.head_dim, device=x.device, dtype=q.dtype)
        k_state = torch.zeros(B, self.n_heads, feat_dim, device=x.device, dtype=q.dtype)

        out_chunks = []

        for i in range(num_chunks):
            st = i * chunk_size
            ed = min(st + chunk_size, T)

            q_chunk = q_features[:, :, st:ed]
            k_chunk = k_features[:, :, st:ed]
            v_chunk = v[:, :, st:ed]

            kv_chunk_prod = torch.einsum("bhtf,bhtd->bhtfd", k_chunk, v_chunk)
            kv_local_cumsum = torch.cumsum(kv_chunk_prod, dim=2)
            k_local_cumsum = torch.cumsum(k_chunk, dim=2)

            kv_current = kv_local_cumsum + kv_state.unsqueeze(2)
            k_current = k_local_cumsum + k_state.unsqueeze(2)

            context_chunk = torch.einsum("bhtf,bhtfd->bhtd", q_chunk, kv_current)
            denom_chunk = torch.einsum("bhtf,bhtf->bht", q_chunk, k_current)

            denom_chunk = torch.clamp(denom_chunk, min=1e-6)
            out_chunk = context_chunk / denom_chunk.unsqueeze(-1)
            out_chunks.append(out_chunk)

            kv_state = kv_current[:, :, -1]
            k_state = k_current[:, :, -1]

        out = torch.cat(out_chunks, dim=2)
        out = out.to(input_dtype)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
