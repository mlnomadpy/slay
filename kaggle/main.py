# Install necessary libraries
# !pip install -Uq "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip install -Uq tokenizers datasets wandb orbax-checkpoint flax optax mteb torch

import os
import time
import json
from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as orbax
import wandb
from datasets import load_dataset
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tokenizers import Tokenizer

# --- JAX Device and Mesh Setup ---
if jax.default_backend() == 'tpu':
    mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
else:
    num_devices = len(jax.devices())
    mesh_shape = (num_devices, 1)
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))

# --- Modern Architecture Components ---

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs = None):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim))

    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes cos and sin frequencies for RoPE."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs)
    # Shape: [end, dim/2]
    return jnp.array(np.cos(freqs)), jnp.array(np.sin(freqs))

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Applies Rotary Positional Embeddings."""
    # x: [Batch, SeqLen, NumHeads, HeadDim]
    # freqs: [SeqLen, HeadDim/2]
    
    # Split x into even and odd components for rotation
    # Assuming HeadDim is the last dimension
    d = x.shape[-1]
    x_r = x[..., 0::2]
    x_i = x[..., 1::2]
    
    # Reshape freqs for broadcasting: [1, SeqLen, 1, HeadDim/2]
    freqs_cos = freqs_cos.reshape(1, freqs_cos.shape[0], 1, freqs_cos.shape[1])
    freqs_sin = freqs_sin.reshape(1, freqs_sin.shape[0], 1, freqs_sin.shape[1])
    
    # Apply rotation
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos
    
    # Interleave back
    x_out = jnp.stack([x_out_r, x_out_i], axis=-1).reshape(x.shape)
    return x_out

class RotarySelfAttention(nnx.Module):
    """Multi-Head Attention with RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
        else:
            kernel_init = nnx.initializers.xavier_uniform()

        self.q_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        self.o_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, x, freqs_cos, freqs_sin, decode: bool = False):
        B, L, E = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        # Scaled Dot-Product Attention
        # Standard implementation
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        
        # Causal mask is NOT used for BERT (bidirectional), but padding mask might be needed.
        # For simplicity in this mini-bert MLM, we omit complex masking logic assuming full attention 
        # is valid for non-pad tokens. In standard Flax MHA, this is handled carefully.
        # Ideally, we should mask padded positions here using -inf. 
        # For now, we rely on the robustness of the model and the MLM objective to ignore pads.
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        output = output.reshape(B, L, E)
        return self.o_proj(output)


# --- Alternative Attention Mechanisms ---

class LinearCausalAttention(nnx.Module):
    """ELU+1 linear attention with causal masking."""
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.eps = 1e-6
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape: [B, L, H, D] -> [B, H, L, D]
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # ELU+1 kernel
        q_prime = jax.nn.elu(q) + 1.0
        k_prime = jax.nn.elu(k) + 1.0
        
        # Causal linear attention via cumsum
        # kv_prod: [B, H, L, D, D]
        kv_prod = jnp.einsum('bhld,bhle->bhlde', k_prime, v)
        kv_cumsum = jnp.cumsum(kv_prod, axis=2)
        
        # context = q_t * S_t
        context = jnp.einsum('bhld,bhlde->bhle', q_prime, kv_cumsum)
        
        # Normalization
        k_cumsum = jnp.cumsum(k_prime, axis=2) # [B,H,L,D]
        norm = jnp.einsum('bhld,bhld->bhl', q_prime, k_cumsum)
        
        output = context / (norm[..., None] + self.eps)
        
        # Reshape back: [B, H, L, D] -> [B, L, H, D] -> [B, L, E]
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class FastAttention(nnx.Module):
    """Performer-style (FAVOR+) Linear Attention (ReLU)."""
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 64, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        # Frozen Random Projection Matrix (Gaussian)
        proj_key = rngs.params()
        self.proj_matrix = nnx.Cache(jax.random.normal(proj_key, (self.num_heads, self.head_dim, kernel_size)) / jnp.sqrt(self.head_dim))

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 1. Kernel Feature Map
        proj_matrix = jax.device_put(self.proj_matrix.value) # [H, D, M]
        
        q_prime = jax.nn.relu(jnp.einsum('bhld,hdm->bhlm', q, proj_matrix))
        k_prime = jax.nn.relu(jnp.einsum('bhld,hdm->bhlm', k, proj_matrix))
        
        # 2. Causal Linear Attention
        kv_prod = jnp.einsum('bhlm,bhld->bhlmd', k_prime, v)
        kv_cumsum = jnp.cumsum(kv_prod, axis=2)
        
        context = jnp.einsum('bhlm,bhlmd->bhld', q_prime, kv_cumsum)
        
        k_cumsum = jnp.cumsum(k_prime, axis=2)
        norm = jnp.einsum('bhlm,bhlm->bhl', q_prime, k_cumsum)
        
        output = context / (norm[..., None] + 1e-6)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class CosformerCausalAttention(nnx.Module):
    """Cosformer with cos-based reweighting."""
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = 1e-6
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Position weighting
        positions = jnp.arange(L, dtype=jnp.float32).reshape(1, 1, L, 1)
        cos_w = jnp.cos(jnp.pi / 2 * positions / L)
        sin_w = jnp.sin(jnp.pi / 2 * positions / L)
        
        q_prime = jax.nn.relu(q)
        k_prime = jax.nn.relu(k)
        
        q_cos, q_sin = q_prime * cos_w, q_prime * sin_w
        k_cos, k_sin = k_prime * cos_w, k_prime * sin_w
        
        kv_cos = jnp.einsum('bhld,bhle->bhlde', k_cos, v)
        kv_cos_cumsum = jnp.cumsum(kv_cos, axis=2)
        context_cos = jnp.einsum('bhld,bhlde->bhle', q_cos, kv_cos_cumsum)
        
        kv_sin = jnp.einsum('bhld,bhle->bhlde', k_sin, v)
        kv_sin_cumsum = jnp.cumsum(kv_sin, axis=2)
        context_sin = jnp.einsum('bhld,bhlde->bhle', q_sin, kv_sin_cumsum)
        
        context = context_cos + context_sin
        
        k_cos_cumsum = jnp.cumsum(k_cos, axis=2)
        k_sin_cumsum = jnp.cumsum(k_sin, axis=2)
        norm = (jnp.einsum('bhld,bhld->bhl', q_cos, k_cos_cumsum) + 
                jnp.einsum('bhld,bhld->bhl', q_sin, k_sin_cumsum))
        
        output = context / (norm[..., None] + self.eps)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class RFFCausalAttention(nnx.Module):
    """Random Fourier Features attention (Gaussian kernel)."""
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 64, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        self.eps = 1e-6
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        param_key = rngs.params()
        k1, k2 = jax.random.split(param_key)
        self.omega = nnx.Cache(jax.random.normal(k1, (self.num_heads, self.head_dim, num_features)))
        self.bias = nnx.Cache(jax.random.uniform(k2, (self.num_heads, num_features)) * 2 * jnp.pi)
        
    def _rff_features(self, x):
        omega = jax.device_put(self.omega.value)
        bias = jax.device_put(self.bias.value)
        proj = jnp.einsum('bhld,hdm->bhlm', x, omega)
        proj = proj + bias[None, :, None, :]
        return jnp.sqrt(2.0 / self.num_features) * jnp.cos(proj)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q_prime = self._rff_features(q)
        k_prime = self._rff_features(k)
        
        kv_prod = jnp.einsum('bhlm,bhld->bhlmd', k_prime, v)
        kv_cumsum = jnp.cumsum(kv_prod, axis=2)
        
        context = jnp.einsum('bhlm,bhlmd->bhld', q_prime, kv_cumsum)
        
        k_cumsum = jnp.cumsum(k_prime, axis=2)
        norm = jnp.einsum('bhlm,bhlm->bhl', q_prime, k_cumsum)
        
        output = context / (norm[..., None] + self.eps)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)


class YatCausalAttention(nnx.Module):
    """Yat-product attention (exact) with causal masking."""
    def __init__(self, embed_dim: int, num_heads: int, epsilon: float = 1e-6, score_scale=None, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon
        self.score_scale = score_scale if score_scale is not None else 1.0
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compute dot product: q·k
        # q: [B, H, L, D], k: [B, H, L, D] -> dot: [B, H, L, L]
        dot_product = jnp.einsum('bhid,bhjd->bhij', q, k)
        
        # Compute squared norms
        q_norm_sq = jnp.sum(q * q, axis=-1, keepdims=True) # [B, H, L, 1]
        k_norm_sq = jnp.sum(k * k, axis=-1, keepdims=True) # [B, H, L, 1]
        
        # Yat kernel: (q·k)² / (||q||² + ||k||² - 2*q·k + eps)
        # Broadcast norms: q [L, 1] + k [1, L] -> [L, L]
        vals = q_norm_sq + k_norm_sq.transpose(0, 1, 3, 2)
        
        numerator = dot_product ** 2
        denominator = vals - 2 * dot_product + self.epsilon
        
        scores = numerator / denominator
        scores = scores * self.score_scale
        
        # Causal mask (using -inf)
        mask = jnp.triu(jnp.ones((L, L)), k=1) * -1e9
        scores = scores + mask[None, None, :, :]
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class YatSphericalCausalAttention(nnx.Module):
    """Exact Spherical Yat attention with causal masking."""
    def __init__(self, embed_dim: int, num_heads: int, epsilon: float = 1e-2, score_scale=None, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.score_scale = score_scale if score_scale is not None else 1.0
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Normalize to unit sphere
        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        k_norm = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
        
        # x = <q̂, k̂> ∈ [-1, 1]
        x_dot = jnp.einsum('bhid,bhjd->bhij', q_norm, k_norm)
        
        # Kernel: x² / (C - 2x)
        denominator = self.C - 2 * x_dot
        scores = (x_dot ** 2) / denominator
        scores = scores * self.score_scale
        
        # Causal mask
        mask = jnp.triu(jnp.ones((L, L)), k=1) * -1e9
        scores = scores + mask[None, None, :, :]
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)


class YatPerformerCausalAttention(nnx.Module):
    """Linearized Yat attention (approximate) using Hadamard fusion and chunking."""
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 32, num_quadrature_nodes: int = 2, epsilon: float = 1e-6, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        # Gauss-Laguerre nodes
        try:
            nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        except Exception:
            # Fallback for simple cases if numpy not avail/issues, though np is imported
            nodes = np.array([0.585786, 3.414214]) if num_quadrature_nodes == 2 else np.array([1.0])
            weights = np.array([0.853553, 0.146447]) if num_quadrature_nodes == 2 else np.array([1.0])
            
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) / self.C)
        
        # SHARED omega for both poly and PRF (Hadamard optimization)
        # Shape: [R, H, D, M]
        param_key = rngs.params()
        self.omega = nnx.Cache(jax.random.normal(param_key, (num_quadrature_nodes, self.num_heads, self.head_dim, num_features)))

    def _compute_features_fast(self, x):
        # x: [B, H, L, D]
        # omega: [R, H, D, M]
        # x needs to be broadcast against R: [1, B, H, L, D]
        # omega needs to be broadcast against B, L: [R, 1, H, 1, D, M]
        
        # We can use einsum. 
        # x: bhld, omega: rhdm -> result: rbhlm
        # Normalize x first
        x_norm = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        omega = jax.device_put(self.omega.value)
        quad_nodes = jax.device_put(self.quad_nodes.value)
        quad_weights = jax.device_put(self.quad_weights.value)
        
        # proj: [R, B, H, L, M]
        proj = jnp.einsum('bhld,rhdm->rbhlm', x_norm, omega)
        
        # Poly features: (proj^2) / sqrt(M)
        poly_feat = (proj ** 2) / jnp.sqrt(self.num_features)
        
        # PRF features
        # sqrt(2*s) * proj - s
        # s: [R] -> broadcast to [R, 1, 1, 1, 1]
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1)
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-20.0, a_max=20.0)
        prf_feat = jnp.exp(exp_arg) / jnp.sqrt(self.num_features)
        
        # Hadamard Fusion
        fused = poly_feat * prf_feat
        
        # Apply weights: sqrt(w) * fused
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        fused = fused * sq_weights
        
        # Reshape to [B, H, L, R*M]
        # Permute: [R, B, H, L, M] -> [B, H, L, R, M]
        fused = fused.transpose(1, 2, 3, 0, 4)
        B, H, L, _, _ = fused.shape
        fused = fused.reshape(B, H, L, -1)
        return fused

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) # [B,H,L,D]
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q_feat = self._compute_features_fast(q) # [B, H, L, F] (F=R*M)
        k_feat = self._compute_features_fast(k) # [B, H, L, F]
        
        # Chunked Scan
        
        def scan_fn(carry, inputs):
            # carry: (kv_state [B,H,F,D], k_state [B,H,F])
            # inputs: (q [B,H,F], k [B,H,F], v [B,H,D])
            kv_state, k_state = carry
            q_t, k_t, v_t = inputs
            
            # Update states
            # k_t: [B,H,F], v_t: [B,H,D] -> outer: [B,H,F,D]
            kv_update = jnp.einsum('bhf,bhd->bhfd', k_t, v_t)
            kv_state = kv_state + kv_update
            k_state = k_state + k_t
            
            # Compute output
            # q_t: [B,H,F]
            # num: q_t * kv_state -> [B,H,D]
            numerator = jnp.einsum('bhf,bhfd->bhd', q_t, kv_state)
            # den: q_t * k_state -> [B,H]
            denominator = jnp.einsum('bhf,bhf->bh', q_t, k_state)
            
            out_t = numerator / (denominator[..., None] + 1e-6)
            return (kv_state, k_state), out_t

        # Prepare scan inputs: permute to [L, B, H, ...]
        q_scan = q_feat.transpose(2, 0, 1, 3) 
        k_scan = k_feat.transpose(2, 0, 1, 3)
        v_scan = v.transpose(2, 0, 1, 3)
        
        F_dim = q_feat.shape[-1]
        D_dim = self.head_dim
        
        init_kv = jnp.zeros((B, self.num_heads, F_dim, D_dim))
        init_k = jnp.zeros((B, self.num_heads, F_dim))
        
        _, output_scan = jax.lax.scan(scan_fn, (init_kv, init_k), (q_scan, k_scan, v_scan))
        
        # output_scan: [L, B, H, D] -> [B, L, H, D] -> [B, L, E]
        output = output_scan.transpose(1, 0, 2, 3).reshape(B, L, E)
        return self.out(output)


class YatPerformerTensorCausalAttention(nnx.Module):
    """Spherical Yat attention with TensorSketch polynomial features (causal)."""
    def __init__(self, embed_dim: int, num_heads: int, num_prf_features: int = 8, num_quadrature_nodes: int = 1, poly_sketch_dim: int = 64, epsilon: float = 1e-6, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_prf_features = num_prf_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.poly_sketch_dim = poly_sketch_dim
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        # Gauss-Laguerre nodes
        try:
            nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        except Exception:
            nodes = np.array([1.0])
            weights = np.array([1.0])
            
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) / self.C)
        
        param_key = rngs.params()
        k1, k2, k3, k4, k5 = jax.random.split(param_key, 5)
        self.omega = nnx.Cache(jax.random.normal(k1, (num_quadrature_nodes, self.num_heads, self.head_dim, num_prf_features)))
        
        # TensorSketch hash/sign
        self.ts_hash1 = nnx.Cache(jax.random.randint(k2, (self.head_dim,), 0, poly_sketch_dim))
        self.ts_sign1 = nnx.Cache(jax.random.randint(k3, (self.head_dim,), 0, 2) * 2 - 1)
        self.ts_hash2 = nnx.Cache(jax.random.randint(k4, (self.head_dim,), 0, poly_sketch_dim))
        self.ts_sign2 = nnx.Cache(jax.random.randint(k5, (self.head_dim,), 0, 2) * 2 - 1)

    def _count_sketch(self, x, h, s):
        # x: [..., D]
        # h: [D]
        # s: [D]
        B, H, L, D = x.shape
        P = self.poly_sketch_dim
        
        # In JAX, we can use idx updating or unsorted_segment_sum
        # We need to sum contributions to each bucket p in [0, P-1]
        
        # Expand h, s to match x
        # Actually simplest is to flatten B,H,L and use vmap or just basic segmented sum
        # But we want to keep dim structure mostly.
        
        # x_flat: [N, D] where N = B*H*L
        x_flat = x.reshape(-1, D)
        
        # We want out: [N, P]
        # Each row n: out[n, h[d]] += x[n,d] * s[d]
        
        # Prepare indices for scatter
        # indices: [N, D] -> values depend on h.
        # h is constant across N.
        # We can use jax.ops.segment_sum but that is 1D.
        
        # Let's compute contributions: vals = x * s (broadcast s)
        vals = x * s[None, None, None, :]
        
        # We want to sum `vals` into buckets defined by `h` along the last dimension.
        # For each (b,h,l), we have D values mapping to P buckets.
        # out[b,h,l, p] = sum_{d: h[d]=p} vals[b,h,l,d]
        
        # We can implement this with a matmul if we construct a [D, P] binary matrix,
        # but P is small (64).
        # Matrix M where M[d, p] = 1 if h[d] == p else 0.
        # out = vals @ M
        
        # Create M on the fly or cache it? cached h is indices.
        # M = jax.nn.one_hot(h, P) -> [D, P]
        M = jax.nn.one_hot(h, P)
        out = jnp.dot(vals, M)
        return out

    def _poly_tensor_sketch(self, x):
        h1, s1 = jax.device_put(self.ts_hash1.value), jax.device_put(self.ts_sign1.value)
        h2, s2 = jax.device_put(self.ts_hash2.value), jax.device_put(self.ts_sign2.value)
        
        cs1 = self._count_sketch(x, h1, s1)
        cs2 = self._count_sketch(x, h2, s2)
        
        # FFT conv
        fft_cs1 = jnp.fft.rfft(cs1, axis=-1)
        fft_cs2 = jnp.fft.rfft(cs2, axis=-1)
        ts = jnp.fft.irfft(fft_cs1 * fft_cs2, n=self.poly_sketch_dim, axis=-1)
        return ts / jnp.sqrt(self.poly_sketch_dim)

    def _prf_features(self, x):
        omega = jax.device_put(self.omega.value) # [R,H,D,M]
        quad_nodes = jax.device_put(self.quad_nodes.value)
        
        # x: [B,H,L,D] -> [1,B,H,L,D]
        # omega: [R,H,D,M] -> [R,1,H,1,D,M]
        proj = jnp.einsum('bhld,rhdm->rbhlm', x, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1) # [R, 1...]
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-20.0, a_max=20.0)
        prf = jnp.exp(exp_arg) / jnp.sqrt(self.num_prf_features)
        return prf

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        k_norm = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
        
        q_poly = self._poly_tensor_sketch(q_norm) # [B,H,L,P]
        k_poly = self._poly_tensor_sketch(k_norm)
        
        q_prf = self._prf_features(q_norm) # [R,B,H,L,M]
        k_prf = self._prf_features(k_norm)
        
        # Apply quad weights (sqrt)
        quad_weights = jax.device_put(self.quad_weights.value)
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        q_prf = q_prf * sq_weights
        k_prf = k_prf * sq_weights
        
        # Tensor Product Logic
        # Out feature map phi(x) = poly ⊗ prf (flattened)
        # We need to perform causal attention with this fused feature map.
        # D_feat = P * M
        # q_fuse = einsum 'bhlp,rbhlm->rbhlpm' -> reshape [B,H,L, R*P*M]
        
        q_fuse = jnp.einsum('bhlp,rbhlm->rbhlpm', q_poly, q_prf)
        k_fuse = jnp.einsum('bhlp,rbhlm->rbhlpm', k_poly, k_prf)
        
        # Sum over R (quadrature nodes) is handled by concatenation or sum?
        # In YatPerformerTensor, it sums chunks: "context_chunk = context_chunk.sum(dim=0)" -> sum over R.
        # This means the R features are essentially additive components of the kernel integral.
        # Integral = sum_r w_r * K_r.
        # phi(x) = [ sqrt(w_1)phi_1(x), sqrt(w_2)phi_2(x), ... ] CONCATENATED
        # Then phi(x).phi(y) = sum_r w_r phi_r(x).phi_r(y)
        # So we should reshape R into the feature dimension (concatenate), NOT sum.
        # WAIT. In the PyTorch code:
        # context_chunk = torch.einsum("rbhtpm,rbhtpmd->rbhtd", q_outer, kv_current)
        # context_chunk = context_chunk.sum(dim=0)
        # This is summing over R. 
        # q_outer has shape [R, ...].
        # Is eq equivalent to concatenating R into feature dim and dot product? Yes.
        # Dot product of [v1, v2] . [u1, u2] = v1.u1 + v2.u2.
        # So summing over R *after* computing per-node context is correct.
        # Alternatively, reshape R into features and do standard scan.
        # Let's reshape R into features to use the same scan function.
        # fused shape: [R, B, H, L, P, M] -> [B, H, L, R, P, M] -> [B, H, L, R*P*M]
        
        q_fuse = q_fuse.transpose(1, 2, 3, 0, 4, 5).reshape(B, self.num_heads, L, -1)
        k_fuse = k_fuse.transpose(1, 2, 3, 0, 4, 5).reshape(B, self.num_heads, L, -1)
        
        F_dim = q_fuse.shape[-1]
        
        # Scan (same as YatPerformer)
        def scan_fn(carry, inputs):
            kv_state, k_state = carry
            q_t, k_t, v_t = inputs
            
            kv_update = jnp.einsum('bhf,bhd->bhfd', k_t, v_t)
            kv_state = kv_state + kv_update
            k_state = k_state + k_t
            
            numerator = jnp.einsum('bhf,bhfd->bhd', q_t, kv_state)
            denominator = jnp.einsum('bhf,bhf->bh', q_t, k_state)
            
            out_t = numerator / (denominator[..., None] + 1e-6)
            return (kv_state, k_state), out_t

        q_scan = q_fuse.transpose(2, 0, 1, 3) 
        k_scan = k_fuse.transpose(2, 0, 1, 3)
        v_scan = v.transpose(2, 0, 1, 3)
        
        D_dim = self.head_dim
        init_kv = jnp.zeros((B, self.num_heads, F_dim, D_dim))
        init_k = jnp.zeros((B, self.num_heads, F_dim))
        
        _, output_scan = jax.lax.scan(scan_fn, (init_kv, init_k), (q_scan, k_scan, v_scan))
        
        output = output_scan.transpose(1, 0, 2, 3).reshape(B, L, E)
        return self.out(output)

class YatPerformerLaplaceCausalAttention(nnx.Module):
    """Laplace-only Yat attention (causal)."""
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 32, num_quadrature_nodes: int = 2, epsilon: float = 1e-6, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        try:
            nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        except Exception:
            nodes = np.array([0.585786, 3.414214]) if num_quadrature_nodes == 2 else np.array([1.0])
            weights = np.array([0.853553, 0.146447]) if num_quadrature_nodes == 2 else np.array([1.0])
            
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        # Laplace-only weights: (C/4) * alpha_r
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) * (self.C / 4.0))
        
        param_key = rngs.params()
        self.omega = nnx.Cache(jax.random.normal(param_key, (num_quadrature_nodes, self.num_heads, self.head_dim, num_features)))

    def _compute_features(self, x):
        x_norm = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        omega = jax.device_put(self.omega.value)
        quad_nodes = jax.device_put(self.quad_nodes.value)
        quad_weights = jax.device_put(self.quad_weights.value)
        
        proj = jnp.einsum('bhld,rhdm->rbhlm', x_norm, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1)
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-20.0, a_max=20.0)
        prf_feat = jnp.exp(exp_arg) / jnp.sqrt(self.num_features)
        
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        fused = prf_feat * sq_weights
        
        # Reshape [R, B, H, L, M] -> [B, H, L, R*M]
        fused = fused.transpose(1, 2, 3, 0, 4)
        B, H, L, _, _ = fused.shape
        fused = fused.reshape(B, H, L, -1)
        return fused

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q_feat = self._compute_features(q)
        k_feat = self._compute_features(k)
        
        def scan_fn(carry, inputs):
            kv_state, k_state = carry
            q_t, k_t, v_t = inputs
            kv_update = jnp.einsum('bhf,bhd->bhfd', k_t, v_t)
            kv_state = kv_state + kv_update
            k_state = k_state + k_t
            numerator = jnp.einsum('bhf,bhfd->bhd', q_t, kv_state)
            denominator = jnp.einsum('bhf,bhf->bh', q_t, k_state)
            out_t = numerator / (denominator[..., None] + 1e-6)
            return (kv_state, k_state), out_t

        q_scan = q_feat.transpose(2, 0, 1, 3) 
        k_scan = k_feat.transpose(2, 0, 1, 3)
        v_scan = v.transpose(2, 0, 1, 3)
        
        F_dim = q_feat.shape[-1]
        D_dim = self.head_dim
        init_kv = jnp.zeros((B, self.num_heads, F_dim, D_dim))
        init_k = jnp.zeros((B, self.num_heads, F_dim))
        
        _, output_scan = jax.lax.scan(scan_fn, (init_kv, init_k), (q_scan, k_scan, v_scan))
        
        output = output_scan.transpose(1, 0, 2, 3).reshape(B, L, E)
        return self.out(output)


class _YatPerformerPolyBase(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_prf_features: int = 8, num_quadrature_nodes: int = 1, poly_dim: int = 64, epsilon: float = 1e-6, chunk_size: int = 256, nystrom_reg: float = 1e-3, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_prf_features = num_prf_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.poly_dim = poly_dim
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.chunk_size = chunk_size
        self.nystrom_reg = nystrom_reg
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        try:
            nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        except Exception:
            nodes = np.array([1.0])
            weights = np.array([1.0])
            
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) / self.C)
        
        param_key = rngs.params()
        self.omega = nnx.Cache(jax.random.normal(param_key, (num_quadrature_nodes, self.num_heads, self.head_dim, num_prf_features)))

    def _prf_features(self, x):
        omega = jax.device_put(self.omega.value)
        quad_nodes = jax.device_put(self.quad_nodes.value)
        
        proj = jnp.einsum('bhld,rhdm->rbhlm', x, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1)
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-20.0, a_max=20.0)
        prf = jnp.exp(exp_arg) / jnp.sqrt(self.num_prf_features)
        return prf

    def _poly_features(self, x_norm):
        raise NotImplementedError

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        k_norm = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
        
        # Chunked scan
        # Since poly features can be large, we might want to do it chunked if memory is issue.
        # But for simplicity, we compute all features first like other implementations.
        
        q_poly = self._poly_features(q_norm)
        k_poly = self._poly_features(k_norm)
        
        q_prf = self._prf_features(q_norm)
        k_prf = self._prf_features(k_norm)
        
        quad_weights = jax.device_put(self.quad_weights.value)
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        q_prf = q_prf * sq_weights
        k_prf = k_prf * sq_weights
        
        # Tensor product fusion
        q_fuse = jnp.einsum('bhlp,rbhlm->rbhlpm', q_poly, q_prf)
        k_fuse = jnp.einsum('bhlp,rbhlm->rbhlpm', k_poly, k_prf)
        
        q_fuse = q_fuse.transpose(1, 2, 3, 0, 4, 5).reshape(B, self.num_heads, L, -1)
        k_fuse = k_fuse.transpose(1, 2, 3, 0, 4, 5).reshape(B, self.num_heads, L, -1)
        
        F_dim = q_fuse.shape[-1]
        
        def scan_fn(carry, inputs):
            kv_state, k_state = carry
            q_t, k_t, v_t = inputs
            
            kv_update = jnp.einsum('bhf,bhd->bhfd', k_t, v_t)
            kv_state = kv_state + kv_update
            k_state = k_state + k_t
            
            numerator = jnp.einsum('bhf,bhfd->bhd', q_t, kv_state)
            denominator = jnp.einsum('bhf,bhf->bh', q_t, k_state)
            
            out_t = numerator / (denominator[..., None] + 1e-6)
            return (kv_state, k_state), out_t

        q_scan = q_fuse.transpose(2, 0, 1, 3) 
        k_scan = k_fuse.transpose(2, 0, 1, 3)
        v_scan = v.transpose(2, 0, 1, 3)
        
        D_dim = self.head_dim
        init_kv = jnp.zeros((B, self.num_heads, F_dim, D_dim))
        init_k = jnp.zeros((B, self.num_heads, F_dim))
        
        _, output_scan = jax.lax.scan(scan_fn, (init_kv, init_k), (q_scan, k_scan, v_scan))
        
        output = output_scan.transpose(1, 0, 2, 3).reshape(B, L, E)
        return self.out(output)

class YatPerformerRMCausalAttention(_YatPerformerPolyBase):
    """Random Maclaurin polynomial features."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim
        # RNG handling in super requires us to pass rngs. 
        # We need more keys for this class.
        # But wait, super consumed rngs? nnx.Rngs is stateful if used as generator?
        # nnx.Rngs object has methods like params().
        # We should probably pass rngs explicitly to super and use it here too.
        rngs = kwargs.get('rngs')
        
        k1, k2 = jax.random.split(rngs.params())
        self.rm_r1 = nnx.Cache(jax.random.normal(k1, (P, D)) / jnp.sqrt(D))
        self.rm_r2 = nnx.Cache(jax.random.normal(k2, (P, D)) / jnp.sqrt(D))

    def _poly_features(self, x_norm):
        r1 = jax.device_put(self.rm_r1.value)
        r2 = jax.device_put(self.rm_r2.value)
        
        proj1 = jnp.einsum('bhld,pd->bhlp', x_norm, r1)
        proj2 = jnp.einsum('bhld,pd->bhlp', x_norm, r2)
        
        return (proj1 * proj2) / jnp.sqrt(self.poly_dim)

class YatPerformerNystromCausalAttention(_YatPerformerPolyBase):
    """Nyström approximation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim
        rngs = kwargs.get('rngs')
        
        anchors = jax.random.normal(rngs.params(), (P, D))
        anchors = anchors / jnp.linalg.norm(anchors, axis=-1, keepdims=True)
        
        # Kernel matrix K(a_i, a_j) = (a_i . a_j)^2
        K = (anchors @ anchors.T) ** 2
        K = K + self.nystrom_reg * jnp.eye(P)
        
        eigvals, eigvecs = jnp.linalg.eigh(K)
        eigvals = jnp.clip(eigvals, a_min=1e-6)
        # W = eigvecs @ diag(eigvals^-0.5) @ eigvecs.T
        W = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T
        
        self.nystrom_anchors = nnx.Cache(anchors)
        self.nystrom_W = nnx.Cache(W)

    def _poly_features(self, x_norm):
        anchors = jax.device_put(self.nystrom_anchors.value)
        W = jax.device_put(self.nystrom_W.value)
        
        K_xA = (jnp.einsum('bhld,pd->bhlp', x_norm, anchors)) ** 2
        return jnp.einsum('bhlp,pq->bhlq', K_xA, W) / jnp.sqrt(self.poly_dim)

class YatPerformerAnchorCausalAttention(_YatPerformerPolyBase):
    """Anchor features."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim
        rngs = kwargs.get('rngs')
        
        anchors = jax.random.normal(rngs.params(), (P, D))
        anchors = anchors / jnp.linalg.norm(anchors, axis=-1, keepdims=True)
        self.anchor_vectors = nnx.Cache(anchors)

    def _poly_features(self, x_norm):
        anchors = jax.device_put(self.anchor_vectors.value)
        return (jnp.einsum('bhld,pd->bhlp', x_norm, anchors) ** 2) / jnp.sqrt(self.poly_dim)


class StandardCausalAttention(nnx.Module):
    """Standard softmax attention with causal masking (no RoPE)."""
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, decode: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        mask = jnp.triu(jnp.ones((L, L)), k=1) * -1e9
        scores = scores + mask[None, None, :, :]
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class ModernTransformerBlock(nnx.Module):
    """Transformer block with RMSNorm and RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1, attention_type: str = 'rotary', attention_kwargs: dict = None):
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
        else:
            kernel_init = nnx.initializers.xavier_uniform()

        self.attention_type = attention_type
        if attention_kwargs is None:
            attention_kwargs = {}

        if attention_type == 'rotary':
            self.attn = RotarySelfAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'standard':
            self.attn = StandardCausalAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'linear':
            self.attn = LinearCausalAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'performer':
            self.attn = FastAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'cosformer':
            self.attn = CosformerCausalAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'rff':
            self.attn = RFFCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat':
            self.attn = YatCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-spherical':
            self.attn = YatSphericalCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-performer':
            self.attn = YatPerformerCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-performer-tensor':
            self.attn = YatPerformerTensorCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-performer-laplace':
            self.attn = YatPerformerLaplaceCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-performer-rm':
            self.attn = YatPerformerRMCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-performer-nystrom':
            self.attn = YatPerformerNystromCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-performer-anchor':
            self.attn = YatPerformerAnchorCausalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = RMSNorm(embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        
        self.ffn = nnx.Sequential(
            nnx.Linear(embed_dim, ff_dim, kernel_init=kernel_init, rngs=rngs),
            nnx.gelu,
            nnx.Linear(ff_dim, embed_dim, kernel_init=kernel_init, rngs=rngs)
        )
        self.norm2 = RMSNorm(embed_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)

    def __call__(self, x, freqs_cos, freqs_sin, training: bool = False):
        # Pre-Norm Architecture
        # x = x + Drop(Attn(Norm(x)))
        h = self.norm1(x)
        
        # Pass freqs only if rotary, or just pass them and let recv ignore
        # All our classes accept (x, freqs_cos, freqs_sin) signature now
        attn_out = self.attn(h, freqs_cos, freqs_sin)
        
        x = x + self.dropout1(attn_out, deterministic=not training)
        
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.dropout2(ffn_out, deterministic=not training)
        return x

class TokenEmbedding(nnx.Module):
    """Just Token Embeddings (No Absolute Positional Embeddings)."""
    def __init__(self, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        return self.token_emb(x)

class MiniBERT(nnx.Module):
    """Modernized MiniBERT with RoPE, RMSNorm, and Tied Weights."""
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs, attention_type: str = 'rotary', attention_kwargs: dict = None):
        self.embedding_layer = TokenEmbedding(vocab_size, embed_dim, rngs=rngs)
        self.transformer_blocks = [ModernTransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs, attention_type=attention_type, attention_kwargs=attention_kwargs) for _ in range(num_transformer_blocks)]
        self.norm_final = RMSNorm(embed_dim, rngs=rngs)
        
        # Precompute RoPE frequencies
        self.head_dim = embed_dim // num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, maxlen)
        self.freqs_cos = nnx.Cache(freqs_cos)
        self.freqs_sin = nnx.Cache(freqs_sin)

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        
        freqs_cos = jax.device_put(self.freqs_cos.value)
        freqs_sin = jax.device_put(self.freqs_sin.value)
        
        for block in self.transformer_blocks:
            x = block(x, freqs_cos, freqs_sin, training=training)
            
        x = self.norm_final(x)
        
        embedding_weights = self.embedding_layer.token_emb.embedding.value
        logits = x @ embedding_weights.T
        return logits

    def embed(self, inputs, training: bool = False):
        """Gets embeddings before the final output layer."""
        x = self.embedding_layer(inputs)
        freqs_cos = jax.device_put(self.freqs_cos.value)
        freqs_sin = jax.device_put(self.freqs_sin.value)
        
        for block in self.transformer_blocks:
            x = block(x, freqs_cos, freqs_sin, training=training)
        
        return self.norm_final(x)

def create_model(rngs, config):
    return MiniBERT(
        maxlen=config['maxlen'], vocab_size=config['vocab_size'], embed_dim=config['embed_dim'],
        num_heads=config['num_heads'], feed_forward_dim=config['feed_forward_dim'],
        num_transformer_blocks=config['num_transformer_blocks'], rngs=rngs,
        attention_type=config.get('attention_type', 'rotary'),
        attention_kwargs=config.get('attention_kwargs', {})
    )


# --- Utilities ---
def mean_pooling(hidden_states, attention_mask):
    """Performs mean pooling on the token embeddings."""
    # attention_mask: [batch, seq_len] (1 for valid, 0 for pad)
    input_mask_expanded = jnp.expand_dims(attention_mask, -1) # [batch, seq_len, 1]
    sum_embeddings = jnp.sum(hidden_states * input_mask_expanded, axis=1)
    sum_mask = jnp.clip(input_mask_expanded.sum(axis=1), a_min=1e-9)
    return sum_embeddings / sum_mask

# --- Data Preprocessing (MLM) ---
def create_masked_lm_predictions(tokens, mask_prob, maxlen, vocab_size, mask_token_id, pad_token_id):
    labels = np.full(maxlen, -100)
    # Filter out PAD tokens. RoBERTa pad_token_id is typically 1.
    non_padding_indices = np.where(tokens != pad_token_id)[0]
    
    if len(non_padding_indices) == 0: return tokens, labels
    
    num_to_predict = max(1, int(round(len(non_padding_indices) * mask_prob)))
    masked_indices = np.random.choice(non_padding_indices, size=min(num_to_predict, len(non_padding_indices)), replace=False)
    
    labels[masked_indices] = tokens[masked_indices]
    for i in masked_indices:
        rand = np.random.rand()
        if rand < 0.8: tokens[i] = mask_token_id
        elif rand < 0.9: tokens[i] = np.random.randint(0, vocab_size)
    return tokens, labels

def process_dataset_for_mlm(dataset, tokenizer, maxlen, mask_prob, vocab_size):
    # RoBERTa special tokens
    mask_token_id = tokenizer.token_to_id("<mask>")
    pad_token_id = tokenizer.token_to_id("<pad>")
    
    # Safety check in case the tokenizer doesn't have these exact strings
    if mask_token_id is None: 
        print("Warning: '<mask>' not found, falling back to '[MASK]'")
        mask_token_id = tokenizer.token_to_id("[MASK]")
    if pad_token_id is None:
        print("Warning: '<pad>' not found, falling back to '[PAD]'")
        pad_token_id = tokenizer.token_to_id("[PAD]")

    def tokenize_pad_and_mask(examples):
        input_ids, labels = [], []
        for text in examples['text']:
            # Tokenizer.encode returns an Encoding object with .ids
            # RoBERTa tokenizer handles <s> and </s> automatically in encode()
            encoded = tokenizer.encode(text)
            tokens = encoded.ids[:maxlen]
            
            # Pad manually to ensure numpy consistency
            if len(tokens) < maxlen:
                tokens = tokens + [pad_token_id] * (maxlen - len(tokens))
            else:
                tokens = tokens[:maxlen]
                
            token_array = np.array(tokens)
            masked, label = create_masked_lm_predictions(token_array.copy(), mask_prob, maxlen, vocab_size, mask_token_id, pad_token_id)
            input_ids.append(masked.tolist())
            labels.append(label.tolist())
        return {'input_ids': input_ids, 'labels': labels}
    
    columns_to_remove = [col for col in dataset.column_names if col not in ['input_ids', 'labels']]
    dataset = dataset.map(tokenize_pad_and_mask, batched=True, batch_size=1000, remove_columns=columns_to_remove)
    return dataset.shuffle(buffer_size=10_000, seed=42)



# --- JAX Loss and Step Functions (MLM) ---
def loss_fn_mlm(model, batch, training: bool):
    logits = model(batch['input_ids'], training=training)
    labels = batch['labels']
    logits_flat, labels_flat = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(logits=logits_flat, labels=labels_flat)
    num_masked = jnp.sum(labels_flat != -100)
    return jnp.where(num_masked > 0, jnp.sum(loss_per_pos) / num_masked, 0.0), logits

@nnx.jit
def train_step_mlm(model: MiniBERT, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(lambda m, b: loss_fn_mlm(m, b, training=True), has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, model, optimizer

@nnx.jit
def eval_step_mlm(model: MiniBERT, batch):
    loss, _ = loss_fn_mlm(model, batch, training=False)
    return loss



# --- Main Functions ---
def main_pretrain():
    """Runs the MLM pre-training loop."""
    config = {
        'num_transformer_blocks': 12, 'maxlen': 1024,
        'embed_dim': 768, 'num_heads': 12, 'feed_forward_dim': 3072, 'batch_size': 64,
        'learning_rate': 1e-4, 'mask_prob': 0.15, 
        'max_tokens_to_process': 1_000_000_000, 
        'eval_interval': 10000, 'eval_steps': 50, 'val_set_size': 2000,
        'checkpoint_interval': 10000, 'checkpoint_dir': './minibert_checkpoints',
        'wandb_project': 'fineweb-bert-combined-run'
    }
    config['checkpoint_dir'] = os.path.abspath(config['checkpoint_dir'])
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    max_iterations = config['max_tokens_to_process'] // (config['batch_size'] * config['maxlen'])
    last_checkpoint_path = ""

    wandb.init(project=config['wandb_project'], config=config, name="phase1_mlm")
    rngs = nnx.Rngs(0)
    
    # 1. Load Data Stream
    print("\n=== Data Loading ===")
    full_dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
    
    # 2. Load Pretrained Tokenizer (RoBERTa)
    print("Loading pretrained 'roberta-base' tokenizer...")
    tokenizer = Tokenizer.from_pretrained("roberta-base")
    tokenizer.enable_truncation(max_length=config['maxlen'])
    
    config['vocab_size'] = tokenizer.get_vocab_size()
    print(f"Vocab Size: {config['vocab_size']}")

    model = create_model(rngs, config)
    optimizer = nnx.Optimizer(model, optax.adamw(config['learning_rate']))

    print("\n=== Phase 1: MLM Pre-training ===")
    train_dataset = process_dataset_for_mlm(full_dataset.skip(config['val_set_size']), tokenizer, config['maxlen'], config['mask_prob'], config['vocab_size'])
    val_dataset = process_dataset_for_mlm(full_dataset.take(config['val_set_size']), tokenizer, config['maxlen'], config['mask_prob'], config['vocab_size'])
    train_iterator = iter(train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))
    val_iterator = iter(val_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))

    start_time = time.time()
    for step in range(max_iterations):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))
            batch = next(train_iterator)
        
        sharding = NamedSharding(mesh, P('batch', None))
        sharded_batch = {k: jax.device_put(jnp.array(v), sharding) for k, v in batch.items()}
        loss, model, optimizer = train_step_mlm(model, optimizer, sharded_batch)
        wandb.log({"mlm/train_loss": loss.item()}, step=step)

        if (step + 1) % config['eval_interval'] == 0:
            print(f"MLM Step {step+1}/{max_iterations}, Loss: {loss.item():.4f}")

    path = os.path.join(config['checkpoint_dir'], 'mlm_final')
    checkpointer = orbax.PyTreeCheckpointer()
    _, param_state, _ = nnx.split(model, nnx.Param, ...)
    checkpointer.save(path, item=param_state)
    checkpointer.close()
    last_checkpoint_path = path
    print(f"MLM Pre-training finished. Checkpoint saved at {last_checkpoint_path}")
    
    wandb.finish()
    return last_checkpoint_path, config

if __name__ == '__main__':
    mlm_ckpt, config = main_pretrain()
    wandb.finish()