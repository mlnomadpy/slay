#!/usr/bin/env python3
import os
import sys
import argparse
import time
import json
import zipfile
import math
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# --- Helpers ---

def safe_normalize(x, axis=-1, eps=1e-6):
    """Safely normalize vectors to unit length, avoiding division by zero."""
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.clip(norm, a_min=eps)

# --- Feature Maps (Approximations) ---

class PerformerFeatures(nnx.Module):
    """
    Performer (FAVOR+) feature map: ReLU(x @ W).
    Approximates Softmax kernel.
    """
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 64, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size
        
        proj_key = rngs.params()
        # Random projection matrix for Performer (Fixed, so use Cache)
        self.proj_matrix = nnx.Cache(jax.random.normal(proj_key, (self.num_heads, self.head_dim, kernel_size)) / jnp.sqrt(self.head_dim))

    def __call__(self, x):
        # x: [Batch, Dim]
        orig_shape = x.shape
        x_reshaped = x.reshape(x.shape[0], self.num_heads, self.head_dim)
        
        proj_matrix = self.proj_matrix[...] # [H, D, M]
        
        # [B, H, D] @ [H, D, M] -> [B, H, M]
        proj = jnp.einsum('bhd,hdm->bhm', x_reshaped, proj_matrix)
        
        features = jax.nn.relu(proj)
        
        # Flatten back to [B, Features_Flat]
        return features.reshape(orig_shape[0], -1)

class SLAYFeatures(nnx.Module):
    """
    SLAY feature map: Tensor product of Anchor-based Polynomial and PRF features.
    Approximates Yat kernel (Spherical) with Anchor approximation for polynomial part.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 32, poly_dim: int = 16, num_quadrature_nodes: int = 2, epsilon: float = 1e-6, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features # PRF features (M)
        self.poly_dim = poly_dim # Anchor/Poly features (P)
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        try:
            nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        except Exception:
            nodes = np.array([0.585786, 3.414214]) if num_quadrature_nodes == 2 else np.array([1.0])
            weights = np.array([0.853553, 0.146447]) if num_quadrature_nodes == 2 else np.array([1.0])
            
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) / self.C)
        
        param_key = rngs.params()
        k1, k2 = jax.random.split(param_key)
        
        # PRF Projections
        self.omega = nnx.Cache(jax.random.normal(k1, (num_quadrature_nodes, self.num_heads, self.head_dim, num_features)))
        
        # Anchors for polynomial part
        anchors = jax.random.normal(k2, (poly_dim, self.head_dim))
        anchors = anchors / jnp.linalg.norm(anchors, axis=-1, keepdims=True)
        # Replicate for heads or shared? Generally shared anchors across heads or per-head?
        # SLAYAnchorAttention in main.py uses (P, D) where D is head_dim. 
        # So anchors are shared across heads relative to the subspace D? 
        # Wait, in main.py: anchors = jax.random.normal(..., (P, D)). 
        # In _poly_features: einsum('bhld,pd->bhlp', x_norm, anchors).
        # Yes, shared anchors across heads.
        self.anchor_vectors = nnx.Cache(anchors)

    def __call__(self, x):
        # x: [Batch, Dim]
        B = x.shape[0]
        x_reshaped = x.reshape(B, self.num_heads, self.head_dim)
        
        # Normalize
        x_norm = safe_normalize(x_reshaped, axis=-1)
        
        # 1. Polynomial Features (Anchors)
        anchors = self.anchor_vectors[...] # [P, D]
        # [B, H, D] @ [P, D].T -> [B, H, P]
        poly_proj = jnp.einsum('bhd,pd->bhp', x_norm, anchors)
        poly_feat = (poly_proj ** 2) / jnp.sqrt(self.poly_dim)
        
        # 2. PRF Features
        omega = self.omega[...] # [R, H, D, M]
        quad_nodes = self.quad_nodes[...]
        quad_weights = self.quad_weights[...]
        
        # proj: x [B,H,D], omega [R,H,D,M] -> [R, B, H, M]
        prf_proj = jnp.einsum('bhd,rhdm->rbhm', x_norm, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1) # [R, 1, 1, 1]
        sqrt_2s = jnp.sqrt(2.0 * jnp.clip(s_vals, a_min=0))
        
        exp_arg = jnp.clip(prf_proj * sqrt_2s - s_vals, a_min=-10.0, a_max=10.0)
        prf_feat = jnp.exp(exp_arg) / jnp.sqrt(self.num_features) # [R, B, H, M]
        
        # Apply quad weights
        sq_weights = jnp.sqrt(jnp.clip(quad_weights.reshape(-1, 1, 1, 1), a_min=0))
        prf_feat = prf_feat * sq_weights
        
        # 3. Fusion: Tensor Product
        # poly: b h p
        # prf:  r b h m
        # Out:  b h (r p m)
        
        fused = jnp.einsum('bhp,rbhm->brhpm', poly_feat, prf_feat)
        # Flatten: [B, R, H, P, M] -> [B, -1]
        output = fused.reshape(B, -1)
        return output
        
        # Flatten
        output = fused.reshape(B, -1)
        return output

# --- CONFIGURATION ---
# --- CONFIGURATION ---
DATASETS = {
    'Eurlex-4K': {
        'id': '0B3lPMIHmG6vGU0VTR1pCejFpWjg',
        'train': 'Eurlex/eurlex_train.txt',
        'test': 'Eurlex/eurlex_test.txt'
    },
    'LF-AmazonTitles-131K': {
        'id': '1VlfcdJKJA99223fLEawRmrXhXpwjwJKn',
        'train': 'LF-AmazonTitles-131K/train.txt',
        'test': 'LF-AmazonTitles-131K/test.txt'
    },
    'LF-Amazon-131K': {
        'id': '1YNGEifTHu4qWBmCaLEBfjx07qRqw9DVW',
        'train': 'LF-Amazon-131K/train.txt',
        'test': 'LF-Amazon-131K/test.txt'
    },
    'LF-WikiSeeAlsoTitles-320K': {
        'id': '1edWtizAFBbUzxo9Z2wipGSEA9bfy5mdX',
        'train': 'LF-WikiSeeAlsoTitles-320K/train.txt',
        'test': 'LF-WikiSeeAlsoTitles-320K/test.txt'
    },
    'LF-Wikipedia-500K': {
        'id': '0B3lPMIHmG6vGRmEzVDVkNjBMR3c',
        'train': 'LF-Wikipedia-500K/train.txt',
        'test': 'LF-Wikipedia-500K/test.txt'
    },
    'LF-AmazonTitles-1.3M': {
        'id': '1Davc6BIfoTIAS3mP1mUY5EGcGr2zN2pO',
        'train': 'LF-AmazonTitles-1.3M/train.txt',
        'test': 'LF-AmazonTitles-1.3M/test.txt'
    },
}
TRAIN_FILE = "Eurlex/eurlex_train.txt"
TEST_FILE = "Eurlex/eurlex_test.txt"

# --- DATA DOWNLOADER ---
# --- DATA DOWNLOADER ---
def download_data(dataset_name):
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return

    info = DATASETS[dataset_name]
    train_file = info['train']
    
    # Check if extracted dir exists (heuristic: dirname of train file)
    dataset_dir = os.path.dirname(train_file)
    if os.path.exists(dataset_dir):
        print(f"Dataset {dataset_name} already exists at {dataset_dir}")
        return

    print(f"Downloading {dataset_name} from Google Drive...")
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    file_id = info['id']
    output = f"{dataset_name}.zip"
    
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        # Resource key handling if needed (simplified for now as most new links don't seem to need it explicitly or gdown handles it)
        gdown.download(url, output, quiet=False)

    print(f"Extracting {dataset_name}...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Done.")

# --- DATA LOADER ---
def load_xml_data(file_path):
    print(f"Loading {file_path}...")
    
    # Log head of file before loading
    print(f"--- Head of {file_path} (Raw) ---")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(5):
                line = f.readline()
                if not line: break
                print(line.strip())
    except Exception as e:
        print(f"Could not read raw file head: {e}")
    print("-----------------------------------")

    with open(file_path, 'rb') as f:
        header_line = f.readline()
        header = header_line.decode('utf-8').strip().split()
        num_samples, num_features, num_labels = map(int, header)
        offset = len(header_line)
        
    data = load_svmlight_file(file_path, multilabel=True, n_features=num_features, offset=offset)
    # Convert to list of lists for labels
    labels = [np.array(l, dtype=np.int32) for l in data[1]]

    # Log head of data after loading
    print(f"--- Head of Loaded Data ({file_path}) ---")
    print(f"Num Samples: {num_samples}, Num Features: {num_features}, Num Labels: {num_labels}")
    print("First 5 samples X (indices):")
    for i in range(min(5, data[0].shape[0])):
        print(f"  Sample {i}: {data[0][i].indices}")
    print("First 5 samples Y (labels):")
    for i in range(min(5, len(labels))):
        print(f"  Sample {i}: {labels[i]}")
    print("-----------------------------------------")

    return data[0], labels, num_features, num_labels

class XMLDataset:
    def __init__(self, X: csr_matrix, Y: List[np.ndarray], num_labels: int, batch_size: int, shuffle: bool = True):
        self.X = X
        self.Y = Y
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(X.shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self.indices):
            self.on_epoch_end()
            raise StopIteration
            
        batch_indices = self.indices[self.n: min(self.n + self.batch_size, len(self.indices))]
        self.n += self.batch_size
        
        # Prepare batch
        batch_X = self.X[batch_indices]
        
        max_feats = max((len(batch_X[i].indices) for i in range(len(batch_indices))), default=0)
        padded_feats = np.zeros((len(batch_indices), max_feats), dtype=np.int32)
        masks = np.zeros((len(batch_indices), max_feats), dtype=np.float32)
        
        for i, row_idx in enumerate(range(len(batch_indices))):
             row = batch_X[i] 
             indices = row.indices
             length = len(indices)
             padded_feats[i, :length] = indices
             masks[i, :length] = 1.0
             
        # Y labels: Pad to max length in batch
        max_labels = max((len(self.Y[i]) for i in batch_indices), default=0)
        # We need a fixed shape? No, batch-dynamic shape is fine for JIT if we don't recompile too often.
        # But for stability let's ensure at least 1.
        max_labels = max(max_labels, 1)
        
        padded_labels = np.full((len(batch_indices), max_labels), -1, dtype=np.int32)
        label_masks = np.zeros((len(batch_indices), max_labels), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            lbls = self.Y[idx]
            if len(lbls) > 0:
                padded_labels[i, :len(lbls)] = lbls
                label_masks[i, :len(lbls)] = 1.0
        
        return {
            'features': jnp.array(padded_feats),
            'masks': jnp.array(masks),
            'labels': jnp.array(padded_labels),
            'label_masks': jnp.array(label_masks)
        }

# --- MODELS ---

class MeanEmbedding(nnx.Module):
    def __init__(self, num_features: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(num_features, embed_dim, rngs=rngs)
    
    def __call__(self, indices, mask):
        # indices: [B, L]
        # mask: [B, L]
        embeds = self.embedding(indices) # [B, L, D]
        sum_embeds = jnp.sum(embeds * mask[:, :, None], axis=1) # [B, D]
        sum_mask = jnp.sum(mask, axis=1, keepdims=True) # [B, 1]
        return sum_embeds / jnp.clip(sum_mask, a_min=1e-9)


class FullSoftmaxXML(nnx.Module):
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        # Use bias=False to match KernelXML (dot product similarity)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        
    def __call__(self, indices, mask):
        embeds = self.encoder(indices, mask)
        # Normalize embeddings to match the "Spherical" nature of SLAY/Performer benchmarks
        # and to correct for the small norm of mean-pooled vectors.
        embeds = safe_normalize(embeds, axis=-1)
        return self.classifier(embeds)
        
    def loss(self, indices, mask, labels, label_mask):
        logits = self(indices, mask) # [B, NumLabels]
        log_probs = nnx.log_softmax(logits, axis=-1)
        
        B = logits.shape[0]
        
        # Create multi-hot targets efficiently
        # labels: [B, K] containing indices, -1 for padding
        # We can use scatter
        targets = jnp.zeros_like(logits) # [B, NumLabels]
        
        # We need batch indices for scatter: [B, K] -> [0,0,..0, 1,1,..1, ...]
        batch_indices = jnp.arange(B)[:, None]
        
        # Only set 1 where label_mask is 1
        # Set invalid indices (padding -1) to 0 temporarily (won't matter due to mask but cleaner for scatter)
        safe_labels = jnp.where(label_mask > 0, labels, 0)
        
        # Scatter add? Or just set. XML is usually binary.
        # We rely on max(label_mask) to set 1s only where mask is valid.
        # safe_labels has 0 for padding. mask has 0.0 for padding.
        # So padding entries will do .max(0.0) on index 0, keeping it 0 (unless real label 0 exists).
        targets = targets.at[batch_indices, safe_labels].max(label_mask)
                
        multilabel_loss = -jnp.sum(targets * log_probs) / B
        return multilabel_loss

    def predict(self, indices, mask, k=5):
        logits = self(indices, mask)
        return jax.lax.top_k(logits, k)


class KernelXML(nnx.Module):
    """
    Approximation-based XML using Attention Feature Maps.
    Z ~ Phi(q) . Sum(Phi(W))
    P(y|q) ~ Phi(q) . Phi(w_y) / Z
    """
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, attention_type: str, attention_kwargs: dict = {}, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs) # W matrix
        
        self.attention_type = attention_type
        
        # num_heads=4 default to break vector down? 
        # But we want to approximate a full vector dot product <x, y>.
        # If we split into heads, we are approximating sum_<h> <x_h, y_h>.
        # Which is <x, y>. So splitting into heads is valid way to reduce dim per feature map.
        
        num_heads = attention_kwargs.get('num_heads', 4)
        if embed_dim % num_heads != 0:
            num_heads = 1 # Fallback
        
        if attention_type in ['slay', 'yat', 'yat-spherical']:
            self.feature_map = SLAYFeatures(embed_dim, num_heads, **attention_kwargs, rngs=rngs)
        elif attention_type == 'performer':
            self.feature_map = PerformerFeatures(embed_dim, num_heads, **attention_kwargs, rngs=rngs)
        else:
             raise ValueError(f"Unsupported attention type for KernelXML (only decomposable kernels allowed): {attention_type}")

    def get_features(self, x):
        return self.feature_map(x)

        # Denominator: phi(q) . Z_vec
        denom = jnp.dot(phi_query, Z_approx_vec) + 1e-6
        # log_Z = jnp.log(denom) # This was part of an incomplete method

        
    def loss(self, indices, mask, labels, label_mask):
        B = indices.shape[0]
        # 1. Encode Query
        query = self.encoder(indices, mask) # [B, D]
        query = safe_normalize(query, axis=-1)
        phi_query = self.get_features(query) # [B, M]
        
        # 2. Compute Global Denominator Sum(phi(W))
        W_vecs = self.classifier.kernel[...] # [D, L] -- Wait, Linear kernel is [In, Out] i.e. [D, L]?
        # Earlier I saw transpose logic. `nnx.Linear` kernel is [In, Out] -> [D, Labels].
        # So W_vecs should be [Labels, D] for get_features input?
        # get_features expects [Batch, D].
        # So transposing W to [Labels, D] is correct.
        
        W_vecs = W_vecs.T # [L, D]
        
        phi_W = self.get_features(W_vecs) # [L, M]
        
        Z_approx_vec = jnp.sum(phi_W, axis=0) # [M]
        
        # Denominator: phi(q) . Z_vec
        denom = jnp.dot(phi_query, Z_approx_vec) + 1e-6
        log_Z = jnp.log(denom) # [B]
        
        # Numerator Vectorization
        # labels: [B, K]
        # We need phi(W_pos) for all B, K.
        # W_pos: Gather from W_vecs using labels.
        # OPTIMIZATION: We already computed phi_W for all labels [L, M].
        # We can just gather from phi_W instead of re-projecting W_pos.
        
        # Handle padding in labels (-1) by clamping to 0 (will face mask later)
        safe_labels = jnp.maximum(labels, 0)
        
        # Gather from phi_W: [L, M] -> [B, K, M]
        phi_w_pos = phi_W[safe_labels]
        
        # Dot product with phi_query [B, M] -> expand to [B, 1, M]
        # [B, K, M] * [B, 1, M] -> [B, K, M] -> sum over M -> [B, K]
        nums = jnp.sum(phi_w_pos * phi_query[:, None, :], axis=-1) + 1e-6
        log_nums = jnp.log(nums) # [B, K]
        
        # Loss per positive: -(log_num - log_Z)
        # log_Z is [B], broadcast to [B, K]
        log_probs = log_nums - log_Z[:, None]
        
        # Mask out padding
        masked_log_probs = log_probs * label_mask # [B, K]
        
        # Sum over K positives, then average over Batch
        loss_total = -jnp.sum(masked_log_probs)
        return loss_total / B
        
    def predict(self, indices, mask, k=5):
        query = self.encoder(indices, mask)
        query = safe_normalize(query, axis=-1)
        
        phi_query = self.get_features(query) # [B, M]
        W_vecs = self.classifier.kernel[...]
        W_vecs = W_vecs.T
        phi_W = self.get_features(W_vecs) # [L, M]
        
        scores = phi_query @ phi_W.T
        return jax.lax.top_k(scores, k) # values, indices


class ExactSphericalYatXML(nnx.Module):
    """
    Exact Spherical Yat Kernel: K(q, k) = (q.k)^2 / (2 + eps - 2 q.k)
    Used when q, k are on unit sphere.
    """
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, epsilon: float = 1e-6, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.C = 2.0 + epsilon

    def kernel_fn(self, q, k_vecs):
        # q, k must be normalized
        dot = jnp.dot(q, k_vecs.T)
        raw_kernel = (dot ** 2) / (self.C - 2 * dot)
        return raw_kernel

    def loss(self, indices, mask, labels, label_mask):
        B = indices.shape[0]
        query = self.encoder(indices, mask) 
        query = safe_normalize(query, axis=-1)
        
        W_vecs = self.classifier.kernel[...] 
        W_vecs = W_vecs.T 
        W_vecs = safe_normalize(W_vecs, axis=-1)
        
        sf_scores = self.kernel_fn(query, W_vecs) # [B, L]
        
        Z_vec = jnp.sum(sf_scores, axis=1) # [B]
        log_Z = jnp.log(Z_vec + 1e-9)
        
        # Numerator
        safe_labels = jnp.maximum(labels, 0)
        pos_scores = jnp.take_along_axis(sf_scores, safe_labels, axis=1) # [B, K]
        log_pos = jnp.log(pos_scores + 1e-9)
        
        log_probs = log_pos - log_Z[:, None]
        masked_log_probs = log_probs * label_mask
        loss_total = -jnp.sum(masked_log_probs)
        return loss_total / B
        
    def predict(self, indices, mask, k=5):
        query = self.encoder(indices, mask)
        query = safe_normalize(query, axis=-1)
        
        W_vecs = self.classifier.kernel[...]
        W_vecs = safe_normalize(W_vecs.T, axis=-1)
        
        scores = self.kernel_fn(query, W_vecs)
        return jax.lax.top_k(scores, k)

class ExactYatXML(nnx.Module):
    """
    Exact (General) Yat Kernel: K(q, k) = (q.k)^2 / (||q-k||^2 + eps)
    Does NOT enforce spherical constraint.
    """
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, epsilon: float = 1e-6, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.epsilon = epsilon

    def kernel_fn(self, q, k_vecs):
        # q: [B, D]
        # k_vecs: [L, D]
        
        dot = jnp.dot(q, k_vecs.T) # [B, L]
        
        q_norm2 = jnp.sum(q**2, axis=-1, keepdims=True) # [B, 1]
        k_norm2 = jnp.sum(k_vecs**2, axis=-1) # [L]
        
        # ||q-k||^2 = ||q||^2 + ||k||^2 - 2 q.k
        dist2 = q_norm2 + k_norm2[None, :] - 2 * dot # [B, L]
        dist2 = jnp.maximum(dist2, 0.0) # Numerical safety
        
        denom = dist2 + self.epsilon
        raw_kernel = (dot ** 2) / denom
        return raw_kernel

    def loss(self, indices, mask, labels, label_mask):
        B = indices.shape[0]
        query = self.encoder(indices, mask) # [B, D] (No normalization)
        
        W_vecs = self.classifier.kernel[...] 
        W_vecs = W_vecs.T # [L, D] (No normalization)
        
        sf_scores = self.kernel_fn(query, W_vecs) # [B, L]
        
        Z_vec = jnp.sum(sf_scores, axis=1) # [B]
        log_Z = jnp.log(Z_vec + 1e-9)
        
        safe_labels = jnp.maximum(labels, 0)
        pos_scores = jnp.take_along_axis(sf_scores, safe_labels, axis=1) # [B, K]
        log_pos = jnp.log(pos_scores + 1e-9)
        
        log_probs = log_pos - log_Z[:, None]
        masked_log_probs = log_probs * label_mask
        loss_total = -jnp.sum(masked_log_probs)
        return loss_total / B
        
    def predict(self, indices, mask, k=5):
        query = self.encoder(indices, mask)
        
        W_vecs = self.classifier.kernel[...]
        W_vecs = W_vecs.T
        
        scores = self.kernel_fn(query, W_vecs)
        return jax.lax.top_k(scores, k)

# --- METRICS ---
# --- METRICS ---

def get_propensity_scores(labels, num_labels, A=0.55, B=1.5):
    """
    Calculate propensity scores based on Jain et al. 2016.
    p_l = 1 / (1 + C * (N_l + B)^-A)
    C = (log N - 1) * (B + 1)^A
    """
    N = len(labels) # Number of samples
    
    freqs = np.zeros(num_labels)
    for i in range(N):
        freqs[labels[i]] += 1
        
    C = (np.log(N) - 1) * ((B + 1) ** A)
    
    p = 1.0 / (1.0 + C * ((freqs + B) ** (-A)))
    p[freqs == 0] = 0.0 # Handle missing labels if any (though usually not an issue for prop calculation, just ensures no division by zero later if p used in denominator)
    return p

def precision_at_k(targets_list, pred_indices, k=5):
    p_k_sum = np.zeros(k) # P@1...P@k
    n_samples = len(targets_list)
    pred_indices = np.array(pred_indices)

    for i in range(n_samples):
        # targets_list[i] is likely a JAX array or numpy array with padding -1
        t_row = np.array(targets_list[i]) 
        # Filter out padding (-1)
        true_labels = set(t_row[t_row != -1])
        
        if len(true_labels) == 0: continue
        
        preds = pred_indices[i]
        hits = 0
        for j in range(k):
            if j < len(preds) and preds[j] in true_labels:
                hits += 1
            p_k_sum[j] += hits / (j + 1)
            
    return p_k_sum / n_samples

def psp_at_k(targets_list, pred_indices, propensity_scores, k=5):
    """
    Propensity Scored Precision @ k
    PSP@k = (1/k) * sum_{l in top_k} (y_l / p_l)
    """
    psp_sum = np.zeros(k)
    n_samples = len(targets_list)
    pred_indices = np.array(pred_indices)
    
    # Pre-compute 1/p to avoid division inside loop
    inv_p = np.zeros_like(propensity_scores)
    inv_p[propensity_scores > 0] = 1.0 / propensity_scores[propensity_scores > 0]

    for i in range(n_samples):
        t_row = np.array(targets_list[i])
        true_labels = set(t_row[t_row != -1])
        
        if len(true_labels) == 0: continue
        
        preds = pred_indices[i]
        score = 0.0
        for j in range(k):
            if j < len(preds) and preds[j] in true_labels:
                score += inv_p[preds[j]]
            psp_sum[j] += score / (j + 1)
            
    return psp_sum / n_samples

# --- RUNNER ---
# --- RUNNER ---
# --- RUNNER ---
def run_benchmark(dataset_name: str = "all", method_filter: str = "all"):
    # JAX Device Check
    print(f"JAX Devices: {jax.devices()}")

    # Determine datasets to run
    if dataset_name == "all":
        datasets_to_run = list(DATASETS.keys())
    elif dataset_name in DATASETS:
        datasets_to_run = [dataset_name]
    else:
        print(f"Dataset {dataset_name} not found.")
        return {}

    final_results = {}

    for ds_name in datasets_to_run:
        print(f"\n=== Benchmarking on {ds_name} ===")
        download_data(ds_name)
        
        info = DATASETS[ds_name]
        try:
            X_train, Y_train, n_feat, n_lab = load_xml_data(info['train'])
            X_test, Y_test, _, _ = load_xml_data(info['test'])
        except Exception as e:
            print(f"Failed to load data for {ds_name}: {e}")
            continue
            
        n_feat = max(n_feat, X_test.shape[1])
        
        # Calculate Propensity Scores
        print("Calculating propensity scores...")
        propensity = get_propensity_scores(Y_train, n_lab)
        
        # Config
        BATCH_SIZE = 1024
        # Adjust embedding dim based on dataset size if needed? 
        # Keeping constant 256 for now.
        EMBED_DIM = 256 
        LR = 1e-3
        EPOCHS = 3 # Keep low for speed as per user preference in previous turns
        
        train_ds = XMLDataset(X_train, Y_train, n_lab, BATCH_SIZE, shuffle=True)
        test_ds = XMLDataset(X_test, Y_test, n_lab, BATCH_SIZE, shuffle=False)
        
        ds_results = {}
        
        # Models to test
        all_configs = [
            ('Yat (Exact)', {'exact': True, 'spherical': False}),
            ('Yat (Spherical)', {'exact': True, 'spherical': True}),
            ('FullSoftmax', {}),
            #('Performer', {'attention_type': 'performer', 'attention_kwargs': {'kernel_size': 64}}),
            ('SLAY (Approx)', {'attention_type': 'slay', 'attention_kwargs': {'num_features': 32, 'num_quadrature_nodes': 2}}),
        ]
        
        if method_filter == "all":
            configs = all_configs
        else:
            configs = [c for c in all_configs if method_filter.lower() in c[0].lower()]
            if not configs:
                print(f"No method matched filter '{method_filter}'. Available: {[c[0] for c in all_configs]}")
                continue

        for name, args in configs:
            print(f"\nTraining {name} on {ds_name}...")
            rngs = nnx.Rngs(0)
            
            if name == 'FullSoftmax':
                model = FullSoftmaxXML(n_feat, n_lab, EMBED_DIM, rngs=rngs)
            elif args.get('exact'):
                 if args.get('spherical'):
                     model = ExactSphericalYatXML(n_feat, n_lab, EMBED_DIM, rngs=rngs)
                 else:
                     model = ExactYatXML(n_feat, n_lab, EMBED_DIM, rngs=rngs)
            else:
                try:
                    model = KernelXML(n_feat, n_lab, EMBED_DIM, **args, rngs=rngs)
                except ValueError as e:
                    print(f"Skipping {name}: {e}")
                    continue
                
            optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
            
            @nnx.jit
            def train_step(model, optimizer, indices, mask, labels, label_mask):
                 def loss_fn(m):
                     return m.loss(indices, mask, labels, label_mask)
                 
                 loss, grads = nnx.value_and_grad(loss_fn)(model)
                 optimizer.update(model, grads)
                 return loss
    
            def train_epoch(model, optimizer):
                total_loss = 0
                count = 0
                for batch in train_ds:
                    indices = batch['features']
                    mask = batch['masks']
                    labels = batch['labels']
                    label_mask = batch['label_masks']
                    
                    loss = train_step(model, optimizer, indices, mask, labels, label_mask)
                    
                    total_loss += loss
                    count += 1
                    
                return total_loss / count
    
            for ep in range(EPOCHS):
                t0 = time.time()
                loss = train_epoch(model, optimizer)
                print(f"Ep {ep+1} | Loss: {loss:.4f} | Time: {time.time()-t0:.2f}s")
                
            # Eval
            all_preds = []
            all_targets = []
            for batch in test_ds:
                indices = batch['features']
                mask = batch['masks']
                # Predict top 5
                _, top_k = model.predict(indices, mask, k=5)
                all_preds.extend(top_k)
                all_targets.extend(batch['labels'])
                
            pk = precision_at_k(all_targets, all_preds, k=5)
            pspk = psp_at_k(all_targets, all_preds, propensity, k=5)
            
            print(f"Results for {name}:")
            print(f"  P@1: {pk[0]:.4f}, P@3: {pk[2]:.4f}, P@5: {pk[4]:.4f}")
            print(f"  PSP@1: {pspk[0]:.4f}, PSP@3: {pspk[2]:.4f}, PSP@5: {pspk[4]:.4f}")
            
            ds_results[name] = {
                'P@1': pk[0], 'P@3': pk[2], 'P@5': pk[4],
                'PSP@1': pspk[0], 'PSP@3': pspk[2], 'PSP@5': pspk[4]
            }
            
        final_results[ds_name] = ds_results
        
    return final_results

def generate_latex(results):
    print("\nGenerating LaTeX Table...")
    path = "extreme_results.tex"
    with open(path, "w") as f:
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tabular}{l l cccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Dataset & Method & P@1 & P@3 & P@5 & PSP@1 & PSP@3 & PSP@5 \\" + "\n")
        f.write(r"\midrule" + "\n")
        
        for ds_name, methods in results.items():
            first = True
            for method, metrics in methods.items():
                ds_label = ds_name if first else ""
                f.write(f"{ds_label} & {method} & {metrics['P@1']:.4f} & {metrics['P@3']:.4f} & {metrics['P@5']:.4f} & {metrics['PSP@1']:.4f} & {metrics['PSP@3']:.4f} & {metrics['PSP@5']:.4f} \\\\" + "\n")
                first = False
            f.write(r"\midrule" + "\n")
            
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Extreme Classification Benchmark Results}" + "\n")
        f.write(r"\label{tab:extreme_results}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Table saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="LF-AmazonTitles-131K", help="Dataset to run (or 'all')")
    parser.add_argument("--method", type=str, default="all", help="Method to run (filter)")
    args = parser.parse_args()
    
    results = run_benchmark(args.dataset, args.method)
    generate_latex(results)
