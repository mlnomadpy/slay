"""
Spherical Yat-Performer: alternative polynomial kernel approximations.

Implements three variants for the polynomial kernel (x·y)^2:
- Random Maclaurin (RM)
- Nyström approximation
- Anchor (low-rank) features

All variants use the same PRF exponential features and chunked causal
linear attention, and avoid materializing the full tensor-product features.
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

        omega = self.omega.float()
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

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()

        # Chunked causal linear attention
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size

        R = self.num_quadrature_nodes
        P = self.poly_dim
        M = self.num_prf_features

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

            kv_current = kv_local_cumsum + kv_state.unsqueeze(3)
            k_current = k_local_cumsum + k_state.unsqueeze(3)

            context_chunk = torch.einsum("rbhtpm,rbhtpmd->rbhtd", q_outer, kv_current)
            denom_chunk = torch.einsum("rbhtpm,rbhtpm->rbht", q_outer, k_current)

            context_chunk = context_chunk.sum(dim=0)
            denom_chunk = denom_chunk.sum(dim=0)

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
    """Random Maclaurin polynomial features for (x·y)^2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim

        r1 = torch.randint(0, 2, (P, D), dtype=torch.int64) * 2 - 1
        r2 = torch.randint(0, 2, (P, D), dtype=torch.int64) * 2 - 1
        self.register_buffer("rm_r1", r1)
        self.register_buffer("rm_r2", r2)

    def _poly_features(self, x_norm):
        # x_norm: (B, H, T, D)
        r1 = self.rm_r1.to(x_norm.dtype)
        r2 = self.rm_r2.to(x_norm.dtype)

        proj1 = torch.einsum("bhtd,pd->bhtp", x_norm, r1)
        proj2 = torch.einsum("bhtd,pd->bhtp", x_norm, r2)
        return (proj1 * proj2) / math.sqrt(self.poly_dim)


class YatPerformerNystromCausalAttention(_YatPerformerPolyBase):
    """Nyström approximation for the polynomial kernel (x·y)^2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim

        anchors = torch.randn(P, D)
        K = (anchors @ anchors.t()) ** 2
        K = K + self.nystrom_reg * torch.eye(P)
        eigvals, eigvecs = torch.linalg.eigh(K)
        eigvals = torch.clamp(eigvals, min=1e-8)
        W = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.t()

        self.register_buffer("nystrom_anchors", anchors)
        self.register_buffer("nystrom_W", W)

    def _poly_features(self, x_norm):
        anchors = self.nystrom_anchors.to(x_norm.dtype)
        W = self.nystrom_W.to(x_norm.dtype)
        K_xA = (torch.einsum("bhtd,pd->bhtp", x_norm, anchors)) ** 2
        return torch.einsum("bhtp,pq->bhtq", K_xA, W) / math.sqrt(self.poly_dim)


class YatPerformerAnchorCausalAttention(_YatPerformerPolyBase):
    """Low-rank anchor features for the polynomial kernel (x·y)^2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim
        anchors = torch.randn(P, D)
        self.register_buffer("anchor_vectors", anchors)

    def _poly_features(self, x_norm):
        anchors = self.anchor_vectors.to(x_norm.dtype)
        return (torch.einsum("bhtd,pd->bhtp", x_norm, anchors) ** 2) / math.sqrt(
            self.poly_dim
        )
