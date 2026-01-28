"""
Spherical Yat-Performer (TensorSketch): Linearized Yat attention using
TensorSketch for the polynomial kernel and a tensor-product feature map.

This implementation follows the paper's tensor-product construction:
  phi(u) = sqrt(w_r) * (phi_poly(u) ⊗ phi_prf(u; s_r))

Notes:
- TensorSketch approximates (u·v)^2 with a low-dimensional sketch.
- The tensor product can still be large (P * M per quadrature node).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YatPerformerTensorCausalAttention(nn.Module):
    """Spherical Yat attention with TensorSketch polynomial features (causal).

    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        num_prf_features: Number of PRF features (M)
        num_quadrature_nodes: Number of Gauss-Laguerre nodes (R)
        poly_sketch_dim: TensorSketch dimension for polynomial kernel (P)
        epsilon: Small constant for numerical stability
        chunk_size: Chunk size for causal prefix-sum computation
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        num_prf_features=8,
        num_quadrature_nodes=1,
        poly_sketch_dim=64,
        epsilon=1e-6,
        chunk_size=256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.num_prf_features = num_prf_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.poly_sketch_dim = poly_sketch_dim
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

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

        # TensorSketch hash/sign for polynomial kernel (degree 2)
        # Need TWO independent hash/sign pairs for degree-2 TensorSketch
        h1 = torch.randint(0, poly_sketch_dim, (self.head_dim,), dtype=torch.int64)
        s1 = torch.randint(0, 2, (self.head_dim,), dtype=torch.int64) * 2 - 1
        h2 = torch.randint(0, poly_sketch_dim, (self.head_dim,), dtype=torch.int64)
        s2 = torch.randint(0, 2, (self.head_dim,), dtype=torch.int64) * 2 - 1
        self.register_buffer("ts_hash1", h1)
        self.register_buffer("ts_sign1", s1)
        self.register_buffer("ts_hash2", h2)
        self.register_buffer("ts_sign2", s2)

    def _count_sketch(self, x, h, s):
        """CountSketch of x over last dimension.

        x: (..., D)
        h: hash indices (D,)
        s: sign flips (D,)
        returns: (..., P)
        """
        P = self.poly_sketch_dim

        # Broadcast hash/sign to x shape
        view_shape = (1,) * (x.dim() - 1) + (x.shape[-1],)
        h_exp = h.view(view_shape).expand_as(x)
        s_exp = s.view(view_shape).expand_as(x).to(x.dtype)

        out = torch.zeros(*x.shape[:-1], P, device=x.device, dtype=x.dtype)
        out.scatter_add_(-1, h_exp, x * s_exp)
        return out

    def _poly_tensor_sketch(self, x):
        """TensorSketch for the degree-2 polynomial kernel.

        Approximates (x·y)^2 using convolution of two independent CountSketches.
        For degree-2: phi(x) = iFFT(FFT(CS1(x)) * FFT(CS2(x)))
        Then phi(x)·phi(y) ≈ (x·y)^2
        """
        cs1 = self._count_sketch(x, self.ts_hash1, self.ts_sign1)
        cs2 = self._count_sketch(x, self.ts_hash2, self.ts_sign2)
        
        # FFT-based convolution of two independent sketches
        fft_cs1 = torch.fft.rfft(cs1, dim=-1)
        fft_cs2 = torch.fft.rfft(cs2, dim=-1)
        ts = torch.fft.irfft(fft_cs1 * fft_cs2, n=self.poly_sketch_dim, dim=-1)
        return ts / math.sqrt(self.poly_sketch_dim)

    def _prf_features(self, x):
        """Positive random features for the exponential kernel.

        x: (B, H, T, D)
        returns: (R, B, H, T, M)
        """
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
        """Compute per-chunk tensor-product components.

        Returns:
            poly_feat: (B, H, T, P)
            prf_feat:  (R, B, H, T, M)
        """
        # Normalize input
        x_norm = F.normalize(x_chunk, p=2, dim=-1)

        # Polynomial sketch (B, H, T, P)
        poly_feat = self._poly_tensor_sketch(x_norm)

        # PRF features (R, B, H, T, M)
        prf_feat = self._prf_features(x_norm)

        return poly_feat, prf_feat

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()

        # Chunked causal linear attention (avoid materializing full tensor-product features)
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size

        R = self.num_quadrature_nodes
        P = self.poly_sketch_dim
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

            # Build tensor-product components for the chunk
            q_outer = torch.einsum("bhtp,rbhtm->rbhtpm", q_poly, q_prf)
            k_outer = torch.einsum("bhtp,rbhtm->rbhtpm", k_poly, k_prf)

            kv_chunk_prod = torch.einsum("rbhtpm,bhtd->rbhtpmd", k_outer, v_chunk)
            kv_local_cumsum = torch.cumsum(kv_chunk_prod, dim=3)
            k_local_cumsum = torch.cumsum(k_outer, dim=3)

            kv_current = kv_local_cumsum + kv_state.unsqueeze(3)
            k_current = k_local_cumsum + k_state.unsqueeze(3)

            context_chunk = torch.einsum("rbhtpm,rbhtpmd->rbhtd", q_outer, kv_current)
            denom_chunk = torch.einsum("rbhtpm,rbhtpm->rbht", q_outer, k_current)

            # Sum over quadrature nodes (concatenation equivalence)
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
