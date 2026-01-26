# Install necessary libraries
# !pip install -Uq "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip install -Uq tokenizers datasets wandb orbax-checkpoint flax optax mteb torch

import os
import time
import json
import argparse
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

# Clear any existing JAX caches to free memory
jax.clear_caches()

# --- Numerical Stability Helpers ---
def safe_normalize(x, axis=-1, eps=1e-6):
    """Safely normalize vectors to unit length, avoiding division by zero."""
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.clip(norm, a_min=eps)

# --- JAX Device and Mesh Setup ---
if jax.default_backend() == 'tpu':
    mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))
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
    # freqs: [MaxLen, HeadDim/2] - need to slice to current SeqLen
    
    seq_len = x.shape[1]
    freqs_cos = freqs_cos[:seq_len]
    freqs_sin = freqs_sin[:seq_len]
    
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
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), P(None, 'model'))
        else:
            kernel_init = nnx.initializers.xavier_uniform()

        self.q_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        self.o_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def __call__(self, x, freqs_cos, freqs_sin, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        # Scaled Dot-Product Attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [B, L] with 1=valid, 0=pad
            # Expand to [B, 1, 1, L] for broadcasting
            pad_mask = attention_mask[:, None, None, :]  # [B, 1, 1, L]
            attn_weights = jnp.where(pad_mask == 1, attn_weights, jnp.finfo(attn_weights.dtype).min)
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Apply attention dropout
        if self.attn_dropout > 0 and training:
            attn_weights = self.dropout(attn_weights, deterministic=False)
        
        output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        output = output.reshape(B, L, E)
        return self.o_proj(output)


# --- Alternative Attention Mechanisms ---

class LinearAttention(nnx.Module):
    """ELU+1 linear attention (bidirectional) with optional RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, use_rope: bool = False, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.eps = 1e-4
        self.use_rope = use_rope
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape: [B, L, E] -> [B, L, H, D]
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE before feature mapping if enabled
        if self.use_rope and freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        # Transpose to [B, H, L, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # ELU+1 kernel
        q_prime = jax.nn.elu(q) + 1.0
        k_prime = jax.nn.elu(k) + 1.0
        
        # Apply attention mask to k_prime (zero out pad positions)
        if attention_mask is not None:
            # attention_mask: [B, L] -> [B, 1, L, 1]
            mask = attention_mask[:, None, :, None]
            k_prime = k_prime * mask
            v = v * mask
        
        # Bidirectional linear attention: compute global KV once
        kv = jnp.einsum('bhld,bhle->bhde', k_prime, v)
        context = jnp.einsum('bhld,bhde->bhle', q_prime, kv)
        
        # Normalization
        k_sum = k_prime.sum(axis=2, keepdims=True)
        norm = jnp.einsum('bhld,bhld->bhl', q_prime, jnp.broadcast_to(k_sum, q_prime.shape))
        
        output = context / (norm[..., None] + self.eps)
        
        # Apply dropout
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class FastAttention(nnx.Module):
    """Performer-style (FAVOR+) Linear Attention (ReLU) - bidirectional with optional RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 64, use_rope: bool = False, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.use_rope = use_rope
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        proj_key = rngs.params()
        self.proj_matrix = nnx.Cache(jax.random.normal(proj_key, (self.num_heads, self.head_dim, kernel_size)) / jnp.sqrt(self.head_dim))
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE before feature mapping if enabled
        if self.use_rope and freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        proj_matrix = self.proj_matrix[...]
        
        q_prime = jax.nn.relu(jnp.einsum('bhld,hdm->bhlm', q, proj_matrix))
        k_prime = jax.nn.relu(jnp.einsum('bhld,hdm->bhlm', k, proj_matrix))
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask[:, None, :, None]
            k_prime = k_prime * mask
            v = v * mask
        
        kv = jnp.einsum('bhlm,bhld->bhmd', k_prime, v)
        context = jnp.einsum('bhlm,bhmd->bhld', q_prime, kv)
        
        k_sum = k_prime.sum(axis=2, keepdims=True)
        norm = jnp.einsum('bhlm,bhlm->bhl', q_prime, jnp.broadcast_to(k_sum, q_prime.shape))
        
        output = context / (norm[..., None] + 1e-4)
        
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class CosformerAttention(nnx.Module):
    """Cosformer with cos-based reweighting (bidirectional) with optional RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, use_rope: bool = False, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.eps = 1e-4
        self.use_rope = use_rope
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        if self.use_rope and freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Position weighting
        positions = jnp.arange(L, dtype=jnp.float32).reshape(1, 1, L, 1)
        cos_w = jnp.cos(jnp.pi / 2 * positions / L)
        sin_w = jnp.sin(jnp.pi / 2 * positions / L)
        
        q_prime = jax.nn.relu(q)
        k_prime = jax.nn.relu(k)
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask[:, None, :, None]
            k_prime = k_prime * mask
            v = v * mask
        
        q_cos, q_sin = q_prime * cos_w, q_prime * sin_w
        k_cos, k_sin = k_prime * cos_w, k_prime * sin_w
        
        kv_cos = jnp.einsum('bhld,bhle->bhde', k_cos, v)
        context_cos = jnp.einsum('bhld,bhde->bhle', q_cos, kv_cos)
        
        kv_sin = jnp.einsum('bhld,bhle->bhde', k_sin, v)
        context_sin = jnp.einsum('bhld,bhde->bhle', q_sin, kv_sin)
        
        context = context_cos + context_sin
        
        k_cos_sum = k_cos.sum(axis=2, keepdims=True)
        k_sin_sum = k_sin.sum(axis=2, keepdims=True)
        norm = (jnp.einsum('bhld,bhld->bhl', q_cos, jnp.broadcast_to(k_cos_sum, q_cos.shape)) + 
                jnp.einsum('bhld,bhld->bhl', q_sin, jnp.broadcast_to(k_sin_sum, q_sin.shape)))
        
        output = context / (norm[..., None] + self.eps)
        
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class RFFAttention(nnx.Module):
    """Random Fourier Features attention (Gaussian kernel) - bidirectional with optional RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 64, use_rope: bool = False, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.eps = 1e-4
        self.use_rope = use_rope
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        param_key = rngs.params()
        k1, k2 = jax.random.split(param_key)
        self.omega = nnx.Cache(jax.random.normal(k1, (self.num_heads, self.head_dim, num_features)))
        self.bias = nnx.Cache(jax.random.uniform(k2, (self.num_heads, num_features)) * 2 * jnp.pi)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)
        
    def _rff_features(self, x):
        omega = self.omega[...]
        bias = self.bias[...]
        proj = jnp.einsum('bhld,hdm->bhlm', x, omega)
        proj = proj + bias[None, :, None, :]
        return jnp.sqrt(2.0 / self.num_features) * jnp.cos(proj)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        if self.use_rope and freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        q_prime = self._rff_features(q)
        k_prime = self._rff_features(k)
        
        if attention_mask is not None:
            mask = attention_mask[:, None, :, None]
            k_prime = k_prime * mask
            v = v * mask
        
        kv = jnp.einsum('bhlm,bhld->bhmd', k_prime, v)
        context = jnp.einsum('bhlm,bhmd->bhld', q_prime, kv)
        
        k_sum = k_prime.sum(axis=2, keepdims=True)
        norm = jnp.einsum('bhlm,bhlm->bhl', q_prime, jnp.broadcast_to(k_sum, q_prime.shape))
        
        output = context / (norm[..., None] + self.eps)
        
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)


class YatAttention(nnx.Module):
    """Yat-product attention (exact) - bidirectional."""
    def __init__(self, embed_dim: int, num_heads: int, epsilon: float = 1e-6, score_scale=None, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon
        self.score_scale = score_scale if score_scale is not None else 1.0
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        dot_product = jnp.einsum('bhid,bhjd->bhij', q, k)
        
        q_norm_sq = jnp.sum(q * q, axis=-1, keepdims=True)
        k_norm_sq = jnp.sum(k * k, axis=-1, keepdims=True)
        
        vals = q_norm_sq + k_norm_sq.transpose(0, 1, 3, 2)
        numerator = dot_product ** 2
        denominator = vals - 2 * dot_product + self.epsilon
        
        scores = numerator / denominator
        scores = scores * self.score_scale
        
        # Apply padding mask
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :]
            scores = jnp.where(pad_mask == 1, scores, jnp.finfo(scores.dtype).min)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if self.attn_dropout > 0 and training:
            attn_weights = self.dropout(attn_weights, deterministic=False)
        
        output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class YatSphericalAttention(nnx.Module):
    """Exact Spherical Yat attention - bidirectional."""
    def __init__(self, embed_dim: int, num_heads: int, epsilon: float = 1e-2, score_scale=None, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.score_scale = score_scale if score_scale is not None else 1.0
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q_norm = safe_normalize(q, axis=-1)
        k_norm = safe_normalize(k, axis=-1)
        
        x_dot = jnp.einsum('bhid,bhjd->bhij', q_norm, k_norm)
        
        denominator = jnp.clip(self.C - 2 * x_dot, a_min=1e-4)
        scores = (x_dot ** 2) / denominator
        scores = scores * self.score_scale
        
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :]
            scores = jnp.where(pad_mask == 1, scores, jnp.finfo(scores.dtype).min)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if self.attn_dropout > 0 and training:
            attn_weights = self.dropout(attn_weights, deterministic=False)
        
        output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)


class SLAYAttention(nnx.Module):
    """SLAY: Spherical Linearized Attention with Yat-Kernel (bidirectional)."""
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 32, num_quadrature_nodes: int = 2, epsilon: float = 1e-6, use_rope: bool = False, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.use_rope = use_rope
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        
        try:
            nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        except Exception:
            nodes = np.array([0.585786, 3.414214]) if num_quadrature_nodes == 2 else np.array([1.0])
            weights = np.array([0.853553, 0.146447]) if num_quadrature_nodes == 2 else np.array([1.0])
            
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) / self.C)
        
        param_key = rngs.params()
        self.omega = nnx.Cache(jax.random.normal(param_key, (num_quadrature_nodes, self.num_heads, self.head_dim, num_features)))
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def _compute_features_fast(self, x):
        x_norm = safe_normalize(x, axis=-1)
        
        omega = self.omega[...]
        quad_nodes = self.quad_nodes[...]
        quad_weights = self.quad_weights[...]
        
        proj = jnp.einsum('bhld,rhdm->rbhlm', x_norm, omega)
        poly_feat = (proj ** 2) / jnp.sqrt(self.num_features)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1)
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-10.0, a_max=10.0)
        prf_feat = jnp.exp(exp_arg) / jnp.sqrt(self.num_features)
        
        fused = poly_feat * prf_feat
        
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        fused = fused * sq_weights
        
        fused = fused.transpose(1, 2, 3, 0, 4)
        B, H, L, _, _ = fused.shape
        fused = fused.reshape(B, H, L, -1)
        return fused

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        if self.use_rope and freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        q_feat = self._compute_features_fast(q)
        k_feat = self._compute_features_fast(k)
        
        if attention_mask is not None:
            mask = attention_mask[:, None, :, None]
            k_feat = k_feat * mask
            v = v * mask
        
        kv = jnp.einsum('bhlf,bhld->bhfd', k_feat, v)
        context = jnp.einsum('bhlf,bhfd->bhld', q_feat, kv)
        
        k_sum = k_feat.sum(axis=2, keepdims=True)
        norm = jnp.einsum('bhlf,bhlf->bhl', q_feat, jnp.broadcast_to(k_sum, q_feat.shape))
        
        output = context / (jnp.clip(norm, a_min=1e-4)[..., None])
        
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)


class SLAYTensorAttention(nnx.Module):
    """SLAY with TensorSketch polynomial features (bidirectional)."""
    def __init__(self, embed_dim: int, num_heads: int, num_prf_features: int = 8, num_quadrature_nodes: int = 1, poly_sketch_dim: int = 16, epsilon: float = 1e-6, attn_dropout: float = 0.0, use_rope: bool = False, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_prf_features = num_prf_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.poly_sketch_dim = poly_sketch_dim
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.attn_dropout = attn_dropout
        self.use_rope = use_rope
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)
        
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
        h1, s1 = self.ts_hash1[...]
        h2, s2 = self.ts_hash2[...]
        ts_sign1 = self.ts_sign1[...]
        ts_sign2 = self.ts_sign2[...]
        
        cs1 = self._count_sketch(x, h1, ts_sign1)
        cs2 = self._count_sketch(x, h2, ts_sign2)
        
        # FFT conv
        fft_cs1 = jnp.fft.rfft(cs1, axis=-1)
        fft_cs2 = jnp.fft.rfft(cs2, axis=-1)
        ts = jnp.fft.irfft(fft_cs1 * fft_cs2, n=self.poly_sketch_dim, axis=-1)
        return ts / jnp.sqrt(self.poly_sketch_dim)

    def _prf_features(self, x):
        omega = self.omega[...] # [R,H,D,M]
        quad_nodes = self.quad_nodes[...]
        
        # x: [B,H,L,D] -> [1,B,H,L,D]
        # omega: [R,H,D,M] -> [R,1,H,1,D,M]
        proj = jnp.einsum('bhld,rhdm->rbhlm', x, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1) # [R, 1...]
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-10.0, a_max=10.0)
        prf = jnp.exp(exp_arg) / jnp.sqrt(self.num_prf_features)
        return prf

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Optional RoPE before normalization
        if self.use_rope and freqs_cos is not None and freqs_sin is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q_norm = safe_normalize(q, axis=-1)
        k_norm = safe_normalize(k, axis=-1)
        
        q_poly = self._poly_tensor_sketch(q_norm) # [B,H,L,P]
        k_poly = self._poly_tensor_sketch(k_norm)
        
        q_prf = self._prf_features(q_norm) # [R,B,H,L,M]
        k_prf = self._prf_features(k_norm)
        
        # Apply quad weights (sqrt)
        quad_weights = self.quad_weights[...]
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        q_prf = q_prf * sq_weights
        k_prf = k_prf * sq_weights
        
        # Fuse poly and PRF features: outer product then flatten
        # q_poly: [B,H,L,P], q_prf: [R,B,H,L,M] -> [B,H,L,R,P,M] -> [B,H,L,F]
        q_fuse = jnp.einsum('bhlp,rbhlm->bhlrpm', q_poly, q_prf)
        q_fuse = q_fuse.reshape(B, self.num_heads, L, -1)
        
        k_fuse = jnp.einsum('bhlp,rbhlm->bhlrpm', k_poly, k_prf)
        k_fuse = k_fuse.reshape(B, self.num_heads, L, -1)
        
        # Apply attention mask for linear attention (mask out padding in k,v)
        if attention_mask is not None:
            mask = attention_mask[:, None, :, None]  # [B, 1, L, 1]
            k_fuse = k_fuse * mask
            v = v * mask
        
        # Bidirectional: compute global KV once
        kv = jnp.einsum('bhlf,bhld->bhfd', k_fuse, v)
        context = jnp.einsum('bhlf,bhfd->bhld', q_fuse, kv)
        
        k_sum = k_fuse.sum(axis=2, keepdims=True)
        norm = jnp.einsum('bhlf,bhlf->bhl', q_fuse, jnp.broadcast_to(k_sum, q_fuse.shape))
        
        output = context / (jnp.clip(norm, a_min=1e-4)[..., None])
        
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class SLAYLaplaceAttention(nnx.Module):
    """SLAY Laplace-only attention (bidirectional, no polynomial factor)."""
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 32, num_quadrature_nodes: int = 2, epsilon: float = 1e-6, attn_dropout: float = 0.0, use_rope: bool = False, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        self.attn_dropout = attn_dropout
        self.use_rope = use_rope
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)
        
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
        x_norm = safe_normalize(x, axis=-1)
        omega = self.omega[...]
        quad_nodes = self.quad_nodes[...]
        quad_weights = self.quad_weights[...]
        
        proj = jnp.einsum('bhld,rhdm->rbhlm', x_norm, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1)
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-10.0, a_max=10.0)
        prf_feat = jnp.exp(exp_arg) / jnp.sqrt(self.num_features)
        
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        fused = prf_feat * sq_weights
        
        # Reshape [R, B, H, L, M] -> [B, H, L, R*M]
        fused = fused.transpose(1, 2, 3, 0, 4)
        B, H, L, _, _ = fused.shape
        fused = fused.reshape(B, H, L, -1)
        return fused

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Optional RoPE before feature mapping
        if self.use_rope and freqs_cos is not None and freqs_sin is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q_feat = self._compute_features(q)
        k_feat = self._compute_features(k)
        
        # Apply attention mask for linear attention (mask out padding in k,v)
        if attention_mask is not None:
            mask = attention_mask[:, None, :, None]  # [B, 1, L, 1]
            k_feat = k_feat * mask
            v = v * mask
        
        # Bidirectional: compute global KV once
        kv = jnp.einsum('bhlf,bhld->bhfd', k_feat, v)
        context = jnp.einsum('bhlf,bhfd->bhld', q_feat, kv)
        
        k_sum = k_feat.sum(axis=2, keepdims=True)
        norm = jnp.einsum('bhlf,bhlf->bhl', q_feat, jnp.broadcast_to(k_sum, q_feat.shape))
        
        output = context / (jnp.clip(norm, a_min=1e-4)[..., None])
        
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)


class _SLAYPolyBase(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_prf_features: int = 8, num_quadrature_nodes: int = 1, poly_dim: int = 16, epsilon: float = 1e-6, chunk_size: int = 256, nystrom_reg: float = 1e-3, attn_dropout: float = 0.0, use_rope: bool = False, *, rngs: nnx.Rngs):
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
        self.attn_dropout = attn_dropout
        self.use_rope = use_rope
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)
        
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
        omega = self.omega[...]
        quad_nodes = self.quad_nodes[...]
        
        proj = jnp.einsum('bhld,rhdm->rbhlm', x, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1, 1)
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(proj * sqrt_2s - s_vals, a_min=-10.0, a_max=10.0)
        prf = jnp.exp(exp_arg) / jnp.sqrt(self.num_prf_features)
        return prf

    def _poly_features(self, x_norm):
        raise NotImplementedError

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Optional RoPE before normalization
        if self.use_rope and freqs_cos is not None and freqs_sin is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q_norm = safe_normalize(q, axis=-1)
        k_norm = safe_normalize(k, axis=-1)
        
        # Compute component features [B,H,L,P] and [R,B,H,L,M]
        q_poly = self._poly_features(q_norm)
        k_poly = self._poly_features(k_norm)
        
        q_prf = self._prf_features(q_norm)
        k_prf = self._prf_features(k_norm)
        
        # Weights
        quad_weights = self.quad_weights[...]
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1, 1), a_min=0))
        q_prf = q_prf * sq_weights
        k_prf = k_prf * sq_weights
        
        # Fuse poly and PRF features: outer product then flatten
        # q_poly: [B,H,L,P], q_prf: [R,B,H,L,M] -> [B,H,L,R,P,M] -> [B,H,L,F]
        q_fuse = jnp.einsum('bhlp,rbhlm->bhlrpm', q_poly, q_prf)
        q_fuse = q_fuse.reshape(B, self.num_heads, L, -1)
        
        k_fuse = jnp.einsum('bhlp,rbhlm->bhlrpm', k_poly, k_prf)
        k_fuse = k_fuse.reshape(B, self.num_heads, L, -1)
        
        # Apply attention mask for linear attention (mask out padding in k,v)
        if attention_mask is not None:
            mask = attention_mask[:, None, :, None]  # [B, 1, L, 1]
            k_fuse = k_fuse * mask
            v = v * mask
        
        # Bidirectional: compute global KV once
        kv = jnp.einsum('bhlf,bhld->bhfd', k_fuse, v)
        context = jnp.einsum('bhlf,bhfd->bhld', q_fuse, kv)
        
        k_sum = k_fuse.sum(axis=2, keepdims=True)
        norm = jnp.einsum('bhlf,bhlf->bhl', q_fuse, jnp.broadcast_to(k_sum, q_fuse.shape))
        
        output = context / (jnp.clip(norm, a_min=1e-4)[..., None])
        
        if self.attn_dropout > 0 and training:
            output = self.dropout(output, deterministic=False)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class SLAYRMAttention(_SLAYPolyBase):
    """SLAY with Random Maclaurin polynomial features (bidirectional)."""
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
        r1 = self.rm_r1[...]
        r2 = self.rm_r2[...]
        
        proj1 = jnp.einsum('bhld,pd->bhlp', x_norm, r1)
        proj2 = jnp.einsum('bhld,pd->bhlp', x_norm, r2)
        
        return (proj1 * proj2) / jnp.sqrt(self.poly_dim)

class SLAYNystromAttention(_SLAYPolyBase):
    """SLAY with Nyström approximation (bidirectional)."""
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
        anchors = self.nystrom_anchors[...]
        W = self.nystrom_W[...]
        
        K_xA = (jnp.einsum('bhld,pd->bhlp', x_norm, anchors)) ** 2
        return jnp.einsum('bhlp,pq->bhlq', K_xA, W) / jnp.sqrt(self.poly_dim)

class SLAYAnchorAttention(_SLAYPolyBase):
    """SLAY with Anchor features (bidirectional)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.poly_dim
        D = self.head_dim
        rngs = kwargs.get('rngs')
        
        anchors = jax.random.normal(rngs.params(), (P, D))
        anchors = anchors / jnp.linalg.norm(anchors, axis=-1, keepdims=True)
        self.anchor_vectors = nnx.Cache(anchors)

    def _poly_features(self, x_norm):
        anchors = self.anchor_vectors[...]
        return (jnp.einsum('bhld,pd->bhlp', x_norm, anchors) ** 2) / jnp.sqrt(self.poly_dim)


class StandardAttention(nnx.Module):
    """Standard softmax attention - bidirectional (no causal mask)."""
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        if attn_dropout > 0:
            self.dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def __call__(self, x, freqs_cos=None, freqs_sin=None, attention_mask=None, training: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :]
            scores = jnp.where(pad_mask == 1, scores, jnp.finfo(scores.dtype).min)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if self.attn_dropout > 0 and training:
            attn_weights = self.dropout(attn_weights, deterministic=False)
        
        output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, E)
        return self.out(output)

class ModernTransformerBlock(nnx.Module):
    """Transformer block with RMSNorm and RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1, attention_type: str = 'rotary', attention_kwargs: dict = None):
        if mesh is not None:
             # FIX: Pass PartitionSpec directly, not NamedSharding
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), P(None, 'model'))
        else:
            kernel_init = nnx.initializers.xavier_uniform()

        self.attention_type = attention_type
        if attention_kwargs is None:
            attention_kwargs = {}

        if attention_type == 'rotary':
            self.attn = RotarySelfAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'standard':
            self.attn = StandardAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'linear':
            self.attn = LinearAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'performer':
            self.attn = FastAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'cosformer':
            self.attn = CosformerAttention(embed_dim, num_heads, rngs=rngs)
        elif attention_type == 'rff':
            self.attn = RFFAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat':
            self.attn = YatAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'yat-spherical':
            self.attn = YatSphericalAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'slay':
            self.attn = SLAYAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'slay-tensor':
            self.attn = SLAYTensorAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'slay-laplace':
            self.attn = SLAYLaplaceAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'slay-rm':
            self.attn = SLAYRMAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'slay-nystrom':
            self.attn = SLAYNystromAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
        elif attention_type == 'slay-anchor':
            self.attn = SLAYAnchorAttention(embed_dim, num_heads, rngs=rngs, **attention_kwargs)
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

    @nnx.remat(static_argnums=(5,))
    def __call__(self, x, freqs_cos, freqs_sin, attention_mask=None, training: bool = False):
        # Pre-Norm Architecture
        h = self.norm1(x)
        
        # All attention classes now accept attention_mask and training
        attn_out = self.attn(h, freqs_cos, freqs_sin, attention_mask=attention_mask, training=training)
        
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
        # Use nnx.List to store a list of Modules
        self.transformer_blocks = nnx.List([ModernTransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs, attention_type=attention_type, attention_kwargs=attention_kwargs) for _ in range(num_transformer_blocks)])
        self.norm_final = RMSNorm(embed_dim, rngs=rngs)
        
        # Precompute RoPE frequencies
        self.head_dim = embed_dim // num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, maxlen)
        self.freqs_cos = nnx.Cache(freqs_cos)
        self.freqs_sin = nnx.Cache(freqs_sin)

    def __call__(self, inputs, attention_mask=None, training: bool = False):
        x = self.embedding_layer(inputs)
        
        # FIX: Access variables using [...] to get the Tracer in JIT without creating new DevicePut ops
        freqs_cos = self.freqs_cos[...]
        freqs_sin = self.freqs_sin[...]
        
        for block in self.transformer_blocks:
            x = block(x, freqs_cos, freqs_sin, attention_mask, training)
            
        x = self.norm_final(x)
        
        embedding_weights = self.embedding_layer.token_emb.embedding[...]
        logits = x @ embedding_weights.T
        return logits

    def embed(self, inputs, attention_mask=None, training: bool = False):
        """Gets embeddings before the final output layer."""
        x = self.embedding_layer(inputs)
        freqs_cos = self.freqs_cos[...]
        freqs_sin = self.freqs_sin[...]
        
        for block in self.transformer_blocks:
            x = block(x, freqs_cos, freqs_sin, attention_mask, training)
        
        return self.norm_final(x)

def create_model(rngs, config):
    model = MiniBERT(
        maxlen=config['maxlen'], vocab_size=config['vocab_size'], embed_dim=config['embed_dim'],
        num_heads=config['num_heads'], feed_forward_dim=config['feed_forward_dim'],
        num_transformer_blocks=config['num_transformer_blocks'], rngs=rngs,
        attention_type=config.get('attention_type', 'rotary'),
        attention_kwargs=config.get('attention_kwargs', {})
    )
    
    # Apply positive weight initialization if requested
    if config.get('positive_weights', False):
        model = apply_positive_weights(model, rngs)
    
    return model


def apply_positive_weights(model, rngs):
    """Reinitialize all model weights to be strictly positive using |N(0, sigma)|."""
    key = rngs.params()
    
    # Iterate through all parameters and make them positive
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    
    def init_positive(path, param):
        nonlocal key
        key, subkey = jax.random.split(key)
        # param is the array value
        old_val = param
        # Use fan_in based scaling for better initialization
        fan_in = old_val.shape[0] if len(old_val.shape) > 0 else 1
        scale = 1.0 / jnp.sqrt(fan_in)
        new_val = jnp.abs(jax.random.normal(subkey, old_val.shape) * scale)
        return new_val
    
    # Apply to all params using tree_map_with_path
    new_params = jax.tree_util.tree_map_with_path(init_positive, params)
    
    # Merge back into model
    model = nnx.merge(graphdef, new_params, rest)
    
    print("Applied positive weight initialization to all parameters.")
    return model


# --- Utilities ---
def mean_pooling(hidden_states, attention_mask):
    """Performs mean pooling on the token embeddings."""
    # attention_mask: [batch, seq_len] (1 for valid, 0 for pad)
    input_mask_expanded = jnp.expand_dims(attention_mask, -1) # [batch, seq_len, 1]
    sum_embeddings = jnp.sum(hidden_states * input_mask_expanded, axis=1)
    sum_mask = jnp.clip(input_mask_expanded.sum(axis=1), a_min=1e-9)
    return sum_embeddings / sum_mask

# --- Data Preprocessing (MLM) ---
def create_masked_lm_predictions(tokens, mask_prob, maxlen, vocab_size, mask_token_id, pad_token_id, bos_token_id=None, eos_token_id=None):
    labels = np.full(maxlen, -100)
    # Filter out PAD tokens and special tokens (BOS/EOS)
    special_tokens = {pad_token_id}
    if bos_token_id is not None:
        special_tokens.add(bos_token_id)
    if eos_token_id is not None:
        special_tokens.add(eos_token_id)
    
    non_special_indices = np.array([i for i, t in enumerate(tokens) if t not in special_tokens])
    
    if len(non_special_indices) == 0: return tokens, labels
    
    num_to_predict = max(1, int(round(len(non_special_indices) * mask_prob)))
    masked_indices = np.random.choice(non_special_indices, size=min(num_to_predict, len(non_special_indices)), replace=False)
    
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
        input_ids, labels, attention_masks = [], [], []
        for text in examples['text']:
            # Tokenizer.encode returns an Encoding object with .ids
            # RoBERTa tokenizer handles <s> and </s> automatically in encode()
            encoded = tokenizer.encode(text)
            tokens = encoded.ids[:maxlen]
            
            # Create attention mask before padding (1 for real tokens, 0 for padding)
            attn_mask = [1] * len(tokens)
            
            # Pad manually to ensure numpy consistency
            if len(tokens) < maxlen:
                pad_len = maxlen - len(tokens)
                tokens = tokens + [pad_token_id] * pad_len
                attn_mask = attn_mask + [0] * pad_len
            else:
                tokens = tokens[:maxlen]
                attn_mask = attn_mask[:maxlen]
                
            token_array = np.array(tokens)
            masked, label = create_masked_lm_predictions(token_array.copy(), mask_prob, maxlen, vocab_size, mask_token_id, pad_token_id)
            input_ids.append(masked.tolist())
            labels.append(label.tolist())
            attention_masks.append(attn_mask)
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_masks}
    
    columns_to_remove = [col for col in dataset.column_names if col not in ['input_ids', 'labels', 'attention_mask']]
    dataset = dataset.map(tokenize_pad_and_mask, batched=True, batch_size=1000, remove_columns=columns_to_remove)
    return dataset.shuffle(buffer_size=10_000, seed=42)



# --- JAX Loss and Step Functions (MLM) ---
def loss_fn_mlm(model, batch, training: bool):
    attention_mask = batch.get('attention_mask', None)
    logits = model(batch['input_ids'], attention_mask=attention_mask, training=training)
    labels = batch['labels']
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    
    # Create mask for valid (masked) positions - labels != -100
    mask = labels_flat != -100
    num_masked = jnp.sum(mask)
    
    # Replace -100 with 0 for the loss computation (will be masked out anyway)
    # This prevents indexing errors in softmax_cross_entropy
    safe_labels = jnp.where(mask, labels_flat, 0)
    
    # Compute loss per position
    loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_flat, labels=safe_labels
    )
    
    # Only sum loss for masked positions
    masked_loss = jnp.where(mask, loss_per_pos, 0.0)
    total_loss = jnp.sum(masked_loss)
    
    # Average over number of masked tokens (avoid div by zero)
    loss = jnp.where(num_masked > 0, total_loss / num_masked, 0.0)
    
    return loss, logits

@nnx.jit
def train_step_mlm(model: MiniBERT, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(lambda m, b: loss_fn_mlm(m, b, training=True), has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    
    # Calculate Gradient Norm
    grad_norm = optax.global_norm(grads)
    
    optimizer.update(model, grads)
    return loss, model, optimizer, grad_norm

@nnx.jit
def eval_step_mlm(model: MiniBERT, batch):
    loss, _ = loss_fn_mlm(model, batch, training=False)
    return loss



# --- Main Functions ---
def main_pretrain(**kwargs):
    """Runs the MLM pre-training loop."""
    config = {
        'num_transformer_blocks': 12, 'maxlen': 1024,
        'embed_dim': 768, 'num_heads': 12, 'feed_forward_dim': 3072, 'batch_size': 16,
        'learning_rate': 1e-4, 'mask_prob': 0.15, 
        'warmup_steps': 2000, 'weight_decay': 0.01, 'max_grad_norm': 1.0,
        'max_tokens_to_process': 1_000_000_000, 
        'eval_interval': 10000, 'eval_steps': 50, 'val_set_size': 2000,
        'checkpoint_interval': 10000, 'checkpoint_dir': './minibert_checkpoints',
        'wandb_project': 'fineweb-bert-attention-benchmark',
        'attention_type': 'rotary', # Default
        'attention_kwargs': {},
        'positive_weights': False,  # Optional positive weight initialization
    }
    # Update config with any overrides
    config.update(kwargs)
    
    print(f"Configuration: {json.dumps(config, indent=2)}")
    config['checkpoint_dir'] = os.path.abspath(config['checkpoint_dir'])
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    max_iterations = config['max_tokens_to_process'] // (config['batch_size'] * config['maxlen'])
    last_checkpoint_path = ""

    run_name = f"mlm_{config['attention_type']}"
    wandb.init(project=config['wandb_project'], config=config, name=run_name)
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
    
    with jax.set_mesh(mesh):
        model = create_model(rngs, config)
    
        # Create learning rate schedule with warmup and cosine decay
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config['learning_rate'],
            warmup_steps=config['warmup_steps'],
            decay_steps=max_iterations,
            end_value=config['learning_rate'] * 0.01
        )
        
        # Optimizer with gradient clipping and weight decay
        optimizer = nnx.Optimizer(model, optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.adamw(schedule, weight_decay=config['weight_decay'])
        ), wrt=nnx.Param)

        # Throughput tracking
        total_tokens = 0
        tokens_per_step = config['batch_size'] * config['maxlen']

        # --- Warmup Step ---
        print("\n=== Warmup Step ===")
        # Create dummy batch for compilation
        dummy_input_ids = jnp.ones((config['batch_size'], config['maxlen']), dtype=jnp.int32)
        dummy_labels = jnp.ones((config['batch_size'], config['maxlen']), dtype=jnp.int32)
        dummy_batch = {'input_ids': dummy_input_ids, 'labels': dummy_labels}
        
        # Shard dummy batch
        sharding = NamedSharding(mesh, P('batch', None))
        sharded_dummy_batch = {k: jax.device_put(v, sharding) for k, v in dummy_batch.items()}
        
        print("Compiling training step...")
        warmup_start = time.time()
        # Run one step and block until ready
        warmup_loss, _, _, _ = train_step_mlm(model, optimizer, sharded_dummy_batch)
        jax.block_until_ready(warmup_loss)
        print(f"Warmup complete. Time taken: {time.time() - warmup_start:.2f}s")

    print("\n=== Phase 1: MLM Pre-training ===")
    train_dataset = process_dataset_for_mlm(full_dataset.skip(config['val_set_size']), tokenizer, config['maxlen'], config['mask_prob'], config['vocab_size'])
    val_dataset = process_dataset_for_mlm(full_dataset.take(config['val_set_size']), tokenizer, config['maxlen'], config['mask_prob'], config['vocab_size'])
    train_iterator = iter(train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))
    val_iterator = iter(val_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))

    start_time = time.time()
    step_start_time = time.time()
    
    for step in range(max_iterations):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))
            batch = next(train_iterator)
        
        sharding = NamedSharding(mesh, P('batch', None))
        sharded_batch = {k: jax.device_put(jnp.array(v), sharding) for k, v in batch.items()}
        loss, model, optimizer, grad_norm = train_step_mlm(model, optimizer, sharded_batch)
        
        # Metrics
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        tokens_processed = tokens_per_step
        total_tokens += tokens_processed
        throughput = tokens_processed / step_duration if step_duration > 0 else 0
        
        # Get current learning rate from schedule
        current_lr = schedule(step)
        
        wandb.log({
            "mlm/train_loss": loss.item(),
            "mlm/grad_norm": grad_norm.item(),
            "mlm/learning_rate": float(current_lr),
            "throughput/tokens_per_sec": throughput,
            "throughput/step_duration": step_duration,
            "throughput/total_tokens": total_tokens
        }, step=step)
        
        step_start_time = step_end_time # Reset for next step

        if (step + 1) % config['eval_interval'] == 0:
            print(f"MLM Step {step+1}/{max_iterations}, Loss: {loss.item():.4f}, Tokens/sec: {throughput:.2f}")

    # --- Final Validation ---
    print("\n=== Final Validation ===")
    val_losses = []
    val_iterator = iter(val_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))
    for val_step in range(config['eval_steps']):
        try:
            val_batch = next(val_iterator)
        except StopIteration:
            break
        sharded_val_batch = {k: jax.device_put(jnp.array(v), sharding) for k, v in val_batch.items()}
        val_loss = eval_step_mlm(model, sharded_val_batch)
        val_losses.append(val_loss.item())
    
    if val_losses:
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_perplexity = float(jnp.exp(avg_val_loss))
        print(f"Final Validation Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        wandb.log({
            "mlm/final_val_loss": avg_val_loss,
            "mlm/final_val_perplexity": val_perplexity,
        })
    
    # --- Save Checkpoint ---
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
    parser = argparse.ArgumentParser(description="Train MiniBERT with various attention mechanisms.")
    
    # Attention Selection
    parser.add_argument('--attention_type', type=str, default='rotary', 
                        choices=[
                            'rotary', 'standard', 'linear', 'performer', 'cosformer', 'rff',
                            'yat', 'yat-spherical', 'slay', 'slay-tensor', 
                            'slay-laplace', 'slay-rm', 'slay-nystrom', 
                            'slay-anchor'
                        ],
                        help="Type of attention mechanism to use.")
    
    # --- Model Architecture ---
    parser.add_argument('--num_transformer_blocks', type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument('--embed_dim', type=int, default=768, help="Embedding dimension.")
    parser.add_argument('--num_heads', type=int, default=12, help="Number of attention heads.")
    parser.add_argument('--feed_forward_dim', type=int, default=3072, help="Feed-forward hidden dimension.")
    parser.add_argument('--positive_weights', action='store_true', help="Initialize all weights to be strictly positive.")
    
    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--maxlen', type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--mask_prob', type=float, default=0.15, help="MLM mask probability.")
    parser.add_argument('--warmup_steps', type=int, default=2000, help="Number of warmup steps for learning rate schedule.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument('--max_tokens_to_process', type=int, default=2_500_000_000, help="Total tokens to process during training.")
    
    # --- Evaluation & Checkpointing ---
    parser.add_argument('--eval_interval', type=int, default=10000, help="Steps between evaluations.")
    parser.add_argument('--eval_steps', type=int, default=50, help="Number of eval steps per evaluation.")
    parser.add_argument('--val_set_size', type=int, default=10000, help="Validation set size.")
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help="Steps between checkpoints.")
    parser.add_argument('--checkpoint_dir', type=str, default='./minibert_checkpoints', help="Checkpoint directory.")
    
    # --- Logging ---
    parser.add_argument('--wandb_project', type=str, default='fineweb-bert-attention-benchmark', help="Weights & Biases project name.")
    
    # --- Attention Regularization ---
    parser.add_argument('--attn_dropout', type=float, default=0.0, help="Attention dropout rate (0.0 to disable).")
    parser.add_argument('--use_rope', action='store_true', help="Use RoPE for linear attention variants (before feature mapping).")
    
    # --- Performer / RFF attention ---
    parser.add_argument('--performer_kernel_size', type=int, default=64, help="Kernel size for Performer (FAVOR+).")
    parser.add_argument('--rff_num_features', type=int, default=64, help="Number of RFF features for RFF attention.")
    
    # --- Yat exact attention ---
    parser.add_argument('--yat_epsilon', type=float, default=1e-6, help="Epsilon for Yat kernel denominator.")
    parser.add_argument('--yat_score_scale', type=float, default=1.0, help="Score scaling for Yat attention.")
    
    # --- Yat-Spherical exact attention ---
    parser.add_argument('--yat_spherical_epsilon', type=float, default=1e-2, help="Epsilon for Spherical Yat (C = 2 + eps).")
    
    # --- SLAY (Hadamard fusion) ---
    parser.add_argument('--slay_num_features', type=int, default=32, help="PRF features for SLAY.")
    parser.add_argument('--slay_num_quad', type=int, default=2, help="Quadrature nodes for SLAY.")
    
    # --- SLAY-Tensor (TensorSketch) ---
    parser.add_argument('--slay_tensor_num_prf', type=int, default=8, help="PRF features for SLAY-Tensor.")
    parser.add_argument('--slay_tensor_num_quad', type=int, default=1, help="Quadrature nodes for SLAY-Tensor.")
    parser.add_argument('--slay_tensor_poly_sketch_dim', type=int, default=16, help="TensorSketch dim for polynomial.")
    
    # --- SLAY-Laplace (Laplace-only, no poly factor) ---
    parser.add_argument('--slay_laplace_num_features', type=int, default=32, help="PRF features for SLAY-Laplace.")
    parser.add_argument('--slay_laplace_num_quad', type=int, default=2, help="Quadrature nodes for SLAY-Laplace.")
    
    # --- SLAY-RM (Random Maclaurin polynomial) ---
    parser.add_argument('--slay_rm_num_prf', type=int, default=8, help="PRF features for SLAY-RM.")
    parser.add_argument('--slay_rm_num_quad', type=int, default=1, help="Quadrature nodes for SLAY-RM.")
    parser.add_argument('--slay_rm_poly_dim', type=int, default=16, help="Random Maclaurin polynomial dim.")
    
    # --- SLAY-Nystrom ---
    parser.add_argument('--slay_nystrom_num_prf', type=int, default=8, help="PRF features for SLAY-Nystrom.")
    parser.add_argument('--slay_nystrom_num_quad', type=int, default=1, help="Quadrature nodes for SLAY-Nystrom.")
    parser.add_argument('--slay_nystrom_poly_dim', type=int, default=16, help="Number of Nystrom anchors.")
    parser.add_argument('--slay_nystrom_reg', type=float, default=1e-3, help="Regularization for Nystrom kernel inversion.")
    
    # --- SLAY-Anchor ---
    parser.add_argument('--slay_anchor_num_prf', type=int, default=8, help="PRF features for SLAY-Anchor.")
    parser.add_argument('--slay_anchor_num_quad', type=int, default=1, help="Quadrature nodes for SLAY-Anchor.")
    parser.add_argument('--slay_anchor_poly_dim', type=int, default=16, help="Number of anchor vectors.")
    
    args = parser.parse_args()
    
    # Prepare attention_kwargs based on attention type
    attn_kwargs = {}
    
    if args.attention_type == 'performer':
        attn_kwargs['kernel_size'] = args.performer_kernel_size
        
    elif args.attention_type == 'rff':
        attn_kwargs['num_features'] = args.rff_num_features
        attn_kwargs['use_rope'] = args.use_rope
        
    elif args.attention_type == 'yat':
        attn_kwargs['epsilon'] = args.yat_epsilon
        attn_kwargs['score_scale'] = args.yat_score_scale
        
    elif args.attention_type == 'yat-spherical':
        attn_kwargs['epsilon'] = args.yat_spherical_epsilon
        attn_kwargs['score_scale'] = args.yat_score_scale
        
    elif args.attention_type == 'slay':
        attn_kwargs['num_features'] = args.slay_num_features
        attn_kwargs['num_quadrature_nodes'] = args.slay_num_quad
        attn_kwargs['use_rope'] = args.use_rope
        
    elif args.attention_type == 'slay-tensor':
        attn_kwargs['num_prf_features'] = args.slay_tensor_num_prf
        attn_kwargs['num_quadrature_nodes'] = args.slay_tensor_num_quad
        attn_kwargs['poly_sketch_dim'] = args.slay_tensor_poly_sketch_dim
        attn_kwargs['use_rope'] = args.use_rope
        
    elif args.attention_type == 'slay-laplace':
        attn_kwargs['num_features'] = args.slay_laplace_num_features
        attn_kwargs['num_quadrature_nodes'] = args.slay_laplace_num_quad
        attn_kwargs['use_rope'] = args.use_rope
        
    elif args.attention_type == 'slay-rm':
        attn_kwargs['num_prf_features'] = args.slay_rm_num_prf
        attn_kwargs['num_quadrature_nodes'] = args.slay_rm_num_quad
        attn_kwargs['poly_dim'] = args.slay_rm_poly_dim
        attn_kwargs['use_rope'] = args.use_rope
        
    elif args.attention_type == 'slay-nystrom':
        attn_kwargs['num_prf_features'] = args.slay_nystrom_num_prf
        attn_kwargs['num_quadrature_nodes'] = args.slay_nystrom_num_quad
        attn_kwargs['poly_dim'] = args.slay_nystrom_poly_dim
        attn_kwargs['nystrom_reg'] = args.slay_nystrom_reg
        attn_kwargs['use_rope'] = args.use_rope
        
    elif args.attention_type == 'slay-anchor':
        attn_kwargs['num_prf_features'] = args.slay_anchor_num_prf
        attn_kwargs['num_quadrature_nodes'] = args.slay_anchor_num_quad
        attn_kwargs['poly_dim'] = args.slay_anchor_poly_dim
        attn_kwargs['use_rope'] = args.use_rope
    
    elif args.attention_type in ['linear', 'performer', 'cosformer']:
        attn_kwargs['use_rope'] = args.use_rope
    
    # Add attn_dropout to all attention types that support it
    if args.attn_dropout > 0:
        attn_kwargs['attn_dropout'] = args.attn_dropout

    print(f"Selected Attention: {args.attention_type}")
    print(f"Attention kwargs: {attn_kwargs}")
    
    mlm_ckpt, config = main_pretrain(
        attention_type=args.attention_type,
        attention_kwargs=attn_kwargs,
        # Model architecture
        num_transformer_blocks=args.num_transformer_blocks,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        feed_forward_dim=args.feed_forward_dim,
        positive_weights=args.positive_weights,
        # Training
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        learning_rate=args.learning_rate,
        mask_prob=args.mask_prob,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_tokens_to_process=args.max_tokens_to_process,
        # Eval & Checkpointing
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        val_set_size=args.val_set_size,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        # Logging
        wandb_project=args.wandb_project,
    )
    wandb.finish()