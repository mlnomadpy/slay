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
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx
import optax
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
import queue

# Enable TensorFloat32 for Ampere+ GPUs
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# --- Helpers ---

def safe_normalize(x, axis=-1, eps=1e-4):
    """Safely normalize vectors to unit length, avoiding division by zero."""
    # Robust normalization to avoid nan gradients at zero
    sq_sum = jnp.sum(x**2, axis=axis, keepdims=True)
    # Avoid sqrt(0) by adding epsilon inside sqrt or using maximum
    norm = jnp.sqrt(jnp.maximum(sq_sum, 1e-12))
    return x / (norm + eps)

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
    
    Improvements:
    - Modularized feature computation (Poly, PRF, Fusion)
    - CountSketch approximation for high-dimensional tensor products
    - Improved numerical stability
    """
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 32, poly_dim: int = 16, 
                 num_quadrature_nodes: int = 2, epsilon: float = 1e-6, 
                 use_sketching: bool = False, sketch_dim: Optional[int] = None, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features # PRF features (M)
        self.poly_dim = poly_dim # Anchor/Poly features (P)
        self.num_quadrature_nodes = num_quadrature_nodes
        self.epsilon = epsilon
        self.C = 2.0 + epsilon
        
        self.use_sketching = use_sketching
        self.sketch_dim = sketch_dim if sketch_dim is not None else (poly_dim * num_features)
        
        try:
            nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        except Exception:
            nodes = np.array([0.585786, 3.414214]) if num_quadrature_nodes == 2 else np.array([1.0])
            weights = np.array([0.853553, 0.146447]) if num_quadrature_nodes == 2 else np.array([1.0])
            
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) / self.C)
        
        param_key = rngs.params()
        k1, k2, k3, k4 = jax.random.split(param_key, 4)
        
        # PRF Projections: ω ~ N(0, I)
        self.omega = nnx.Cache(jax.random.normal(k1, (num_quadrature_nodes, self.num_heads, self.head_dim, num_features)))
        
        # Anchors for polynomial part
        # "Anchor features are computationally simplest... and empirically most stable at small P"
        anchors = jax.random.normal(k2, (poly_dim, self.head_dim))
        anchors = safe_normalize(anchors, axis=-1)
        self.anchor_vectors = nnx.Cache(anchors)
        
        # CountSketch Parameters
        if self.use_sketching:
            total_dim = poly_dim * num_features
            # Hash functions for CountSketch: h \in [0, D_t-1], s \in {-1, 1}
            self.sketch_hash = nnx.Cache(jax.random.randint(k3, (total_dim,), 0, self.sketch_dim))
            self.sketch_sign = nnx.Cache(jax.random.choice(k4, jnp.array([-1.0, 1.0]), (total_dim,)))

    def _poly_features(self, x_norm):
        """Compute Anchor features for (x^T y)^2."""
        anchors = self.anchor_vectors[...] # [P, D]
        # [B, H, D] @ [P, D].T -> [B, H, P]
        poly_proj = jnp.einsum('bhd,pd->bhp', x_norm, anchors)
        # φ_anc(x) = (a_i^T x)^2 / sqrt(P)
        poly_feat = (poly_proj ** 2) / jnp.sqrt(self.poly_dim)
        return poly_feat

    def _prf_features(self, x_norm):
        """Compute PRF features for exp(2s x^T y)."""
        # [B, H, D]
        omega = self.omega[...] # [R, H, D, M]
        quad_nodes = self.quad_nodes[...] # [R]
        
        # proj: x [B,H,D], omega [R,H,D,M] -> [R, B, H, M]
        prf_proj = jnp.einsum('bhd,rhdm->rbhm', x_norm, omega)
        
        s_vals = quad_nodes.reshape(-1, 1, 1, 1) # [R, 1, 1, 1]
        sqrt_2s = jnp.sqrt(2.0 * jnp.maximum(s_vals, 0.0))
        
        # exp(sqrt(2s) w^T x - s)
        # Expanded clipping range [-20, 20] typical for float32 exp stability
        exp_arg = jnp.clip(prf_proj * sqrt_2s - s_vals, a_min=-20.0, a_max=20.0)
        prf_feat = jnp.exp(exp_arg) / jnp.sqrt(self.num_features) # [R, B, H, M]
        
        return prf_feat

    def _apply_quadrature_weights(self, prf_feat):
        quad_weights = self.quad_weights[...]
        sq_weights = jnp.sqrt(jnp.maximum(quad_weights.reshape(-1, 1, 1, 1), 0.0))
        return prf_feat * sq_weights

    def _fuse_features(self, poly_feat, prf_feat):
        """
        Fuse Poly and PRF features.
        poly: [B, H, P]
        prf:  [R, B, H, M]
        Returns: [R, B, H, FeatureDim] where FeatureDim = P*M or SketchDim
        """
        R, B, H, M = prf_feat.shape
        P = poly_feat.shape[-1]
        
        # Outer product: [R, B, H, P, M]
        # Equivalent to e.g. poly[:,:,:,None] * prf[:,:,:,None,:] broadcasted
        # poly: b h p -> r b h p 1
        # prf: r b h m -> r b h 1 m
        outer = poly_feat[None, :, :, :, None] * prf_feat[:, :, :, None, :]
        
        if self.use_sketching:
             # Flatten P, M -> [R, B, H, P*M]
             outer_flat = outer.reshape(R, B, H, P * M)
             
             # Apply CountSketch
             # y_j = sum_{i: h(i)=j} s(i) x_i
             hash_idxs = self.sketch_hash[...] # [P*M]
             signs = self.sketch_sign[...]     # [P*M]
             
             # Pre-multiply by signs
             weighted_input = outer_flat * signs[None, None, None, :]
             
             # Aggregate into buckets
             # We want to sum over the last dimension based on hash_idxs
             # JAX: target.at[..., idxs].add(values) works for fixed shapes, but here we scan/reduce?
             # Better: use jax.ops.segment_sum logic or .at[].add with broadcasting
             
             # We can treat (R, B, H) as a large batch dim N
             flat_input = weighted_input.reshape(-1, P*M)
             # Initialize output: [N, D_sketch]
             output_flat = jnp.zeros((flat_input.shape[0], self.sketch_dim), dtype=flat_input.dtype)
             
             # To vectorize efficiently:
             # This is essentially a sparse matrix multiply or scatter add.
             # output_flat.at[:, hash_idxs].add(flat_input)
             # JAX's .at[].add handles duplicate indices correctly by summing.
             output_flat = output_flat.at[:, hash_idxs].add(flat_input)
             
             return output_flat.reshape(R, B, H, self.sketch_dim)
        else:
             # Flatten P, M -> [R, B, H, P*M]
             return outer.reshape(R, B, H, P * M)

    def __call__(self, x):
        # x: [Batch, Dim]
        B = x.shape[0]
        x_reshaped = x.reshape(B, self.num_heads, self.head_dim)
        
        # Normalize
        x_norm = safe_normalize(x_reshaped, axis=-1)
        
        # 1. Polynomial
        poly_feat = self._poly_features(x_norm)
        
        # 2. PRF
        prf_feat = self._prf_features(x_norm)
        prf_feat = self._apply_quadrature_weights(prf_feat)
        
        # 3. Fuse
        # Output: [R, B, H, Feature_Flat]
        fused = self._fuse_features(poly_feat, prf_feat)
        
        # Flatten R and H into feature dimension or similar?
        # Original code flattened [B, R, H, P, M] -> [B, -1].
        # Here we return [B, -1].
        # fused: [R, B, H, D_feat]
        # We need to preserve B. Flatten everything else?
        # Typically Kernel approximation is phi(x)^T phi(y).
        # We have integrated over R via sum (approximated kernel is sum_r w_r K_r).
        # So we can concat features over R? 
        # Yes, if K = sum phi_r(x) phi_r(y), then phi(x) = [phi_1(x), ..., phi_R(x)]
        
        # Transpose to [B, R, H, D_feat]
        fused = fused.transpose(1, 0, 2, 3) 
        # Flatten [B, (R * H * D_feat)]
        output = fused.reshape(B, -1)
        return output

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

class BackgroundGenerator(threading.Thread):
    """
    Prefetches data from a generator in a background thread.
    Useful for hiding data loading latency (CPU) while GPU is working.
    """
    def __init__(self, generator, max_prefetch=2):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        try:
            for item in self.generator:
                self.queue.put(item)
            self.queue.put(None)
        except Exception as e:
            # Propagate exception? For now just stop.
            print(f"BackgroundGenerator Error: {e}")
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

class XMLDataset:
    def __init__(self, X: csr_matrix, Y: List[np.ndarray], num_labels: int, batch_size: int, shuffle: bool = True):
        self.X = X
        self.Y = Y
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(X.shape[0])
        
        # Calculate fixed max lengths for TPU stability (avoid dynamic shapes causing recompilation)
        if X.shape[0] > 0:
            self.max_feat_len = int(X.getnnz(axis=1).max())
            # Y is a list of arrays
            self.max_label_len = max((len(y) for y in Y), default=0)
        else:
            self.max_feat_len = 0
            self.max_label_len = 0
            
        self.max_feat_len = max(self.max_feat_len, 1) # Ensure at least 1
        self.max_label_len = max(self.max_label_len, 1)

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
        # Retrieve raw samples
        batch_X = self.X[batch_indices]
        
        # Determine actual valid count
        actual_batch_size = len(batch_indices)
        
        # Pad features
        # Use fixed max length from __init__ to ensure stable shapes
        padded_feats = np.zeros((self.batch_size, self.max_feat_len), dtype=np.int32)
        masks = np.zeros((self.batch_size, self.max_feat_len), dtype=np.float32)
        
        for i, row_idx in enumerate(range(actual_batch_size)):
             row = batch_X[i] 
             indices = row.indices
             length = len(indices)
             # Safe clip just in case
             length = min(length, self.max_feat_len)
             padded_feats[i, :length] = indices[:length]
             masks[i, :length] = 1.0
             
        # Y labels: Pad to fixed max length
        padded_labels = np.full((self.batch_size, self.max_label_len), -1, dtype=np.int32)
        label_masks = np.zeros((self.batch_size, self.max_label_len), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            lbls = self.Y[idx]
            if len(lbls) > 0:
                length = len(lbls)
                length = min(length, self.max_label_len)
                padded_labels[i, :length] = lbls[:length]
                label_masks[i, :length] = 1.0
        
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

class BaseXML(nnx.Module):
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.num_labels = num_labels

    def get_scores(self, indices, mask):
        """Returns raw scores/logits/probabilities-ish [B, L]"""
        raise NotImplementedError

    def get_logits(self, indices, mask):
        """Returns logits for Softmax [B, L]"""
        # Default: Assume scores ARE logits (for Linear/FullSoftmax)
        return self.get_scores(indices, mask)

    def loss(self, indices, mask, labels, label_mask):
        logits = self.get_logits(indices, mask)
        B = logits.shape[0]
        
        # Create dense targets [B, L]
        targets = jnp.zeros((B, self.num_labels))
        batch_inds = jnp.arange(B)[:, None]
        safe_lbls = jnp.maximum(labels, 0)
        
        # Scatter 1s where labels exist based on label_mask
        targets = targets.at[batch_inds, safe_lbls].max(label_mask)
        
        # Normalize to distribution (Sum to 1)
        target_sum = jnp.sum(targets, axis=-1, keepdims=True)
        targets = targets / jnp.clip(target_sum, a_min=1e-9)
        
        # Optax Loss (Stable Softmax Cross Entropy)
        loss_val = optax.softmax_cross_entropy(logits=logits, labels=targets)
        
        return jnp.mean(loss_val)
        
    def predict(self, indices, mask, k=5):
        scores = self.get_scores(indices, mask)
        return jax.lax.top_k(scores, k)

class FullSoftmaxXML(BaseXML):
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, *, rngs: nnx.Rngs):
        super().__init__(num_features, num_labels, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        
    def get_scores(self, indices, mask):
        embeds = self.encoder(indices, mask)
        # Normalize embeddings to match others
        embeds = safe_normalize(embeds, axis=-1)
        return self.classifier(embeds)

class KernelXML(BaseXML):
    """
    Approximation-based XML using Attention Feature Maps.
    """
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, attention_type: str, attention_kwargs: dict = {}, *, rngs: nnx.Rngs):
        super().__init__(num_features, num_labels, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.attention_type = attention_type
        
        num_heads = attention_kwargs.get('num_heads', 4)
        if embed_dim % num_heads != 0: num_heads = 1
        
        if attention_type in ['slay', 'yat', 'yat-spherical']:
            self.feature_map = SLAYFeatures(embed_dim, num_heads, **attention_kwargs, rngs=rngs)
        elif attention_type == 'performer':
            self.feature_map = PerformerFeatures(embed_dim, num_heads, **attention_kwargs, rngs=rngs)
        else:
             raise ValueError(f"Unsupported attention type: {attention_type}")

    def get_features(self, x):
        return self.feature_map(x)
        
    def get_scores(self, indices, mask):
        query = self.encoder(indices, mask)
        query = safe_normalize(query, axis=-1)
        
        phi_query = self.get_features(query) # [B, M]
        
        W_vecs = self.classifier.kernel[...]
        W_vecs = W_vecs.T # [L, D]
        phi_W = self.get_features(W_vecs) # [L, M]
        
        scores = phi_query @ phi_W.T # [B, L]
        return scores

    def get_logits(self, indices, mask):
        scores = self.get_scores(indices, mask)
        # Handle small negative values from approx? Performer/SLAY usually positive.
        return jnp.log(jnp.clip(scores, a_min=1e-9))

class ExactSphericalYatXML(BaseXML):
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, epsilon: float = 1e-4, *, rngs: nnx.Rngs):
        super().__init__(num_features, num_labels, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.C = 2.0 + epsilon

    def kernel_fn(self, q, k_vecs):
        dot = jnp.dot(q, k_vecs.T)
        raw_kernel = (dot ** 2) / (self.C - 2 * dot + 1e-6)
        return raw_kernel

    def get_scores(self, indices, mask):
        query = self.encoder(indices, mask) 
        query = safe_normalize(query, axis=-1)
        
        W_vecs = self.classifier.kernel[...] 
        W_vecs = safe_normalize(W_vecs.T, axis=-1)
        
        return self.kernel_fn(query, W_vecs)

    def get_logits(self, indices, mask):
        scores = self.get_scores(indices, mask)
        return jnp.log(jnp.clip(scores, a_min=1e-9))

class ExactYatXML(BaseXML):
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, epsilon: float = 1e-4, *, rngs: nnx.Rngs):
        super().__init__(num_features, num_labels, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.epsilon = epsilon

    def kernel_fn(self, q, k_vecs):
        dot = jnp.dot(q, k_vecs.T) # [B, L]
        q_norm2 = jnp.sum(q**2, axis=-1, keepdims=True) # [B, 1]
        k_norm2 = jnp.sum(k_vecs**2, axis=-1) # [L]
        
        dist2 = q_norm2 + k_norm2[None, :] - 2 * dot + self.epsilon
        
        raw_kernel = (dot ** 2) / dist2
        return raw_kernel
        
    def get_scores(self, indices, mask):
        query = self.encoder(indices, mask)
        W_vecs = self.classifier.kernel[...]
        W_vecs = W_vecs.T
        return self.kernel_fn(query, W_vecs)

    def get_logits(self, indices, mask):
        scores = self.get_scores(indices, mask)
        return jnp.log(jnp.clip(scores, a_min=1e-9))

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
def run_benchmark(args):
    dataset_name = args.dataset
    method_filter = args.method
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
        # Config from args
        BATCH_SIZE = args.batch_size
        
        train_ds = XMLDataset(X_train, Y_train, n_lab, BATCH_SIZE, shuffle=True)
        test_ds = XMLDataset(X_test, Y_test, n_lab, BATCH_SIZE, shuffle=False)
        
        ds_results = {}
        
        # Models to test
        all_configs = [
            ('Yat (Exact)', {'exact': True, 'spherical': False}),
            ('Yat (Spherical)', {'exact': True, 'spherical': True}),
            ('FullSoftmax', {}),
            #('Performer', {'attention_type': 'performer', 'attention_kwargs': {'kernel_size': 64}}),
            ('SLAY (Approx)', {'attention_type': 'slay', 'attention_kwargs': {'num_features': args.num_features, 'num_quadrature_nodes': args.num_quadrature_nodes}}),
        ]
        
        if method_filter == "all":
            configs = all_configs
        else:
            configs = [c for c in all_configs if method_filter.lower() in c[0].lower()]
            if not configs:
                print(f"No method matched filter '{method_filter}'. Available: {[c[0] for c in all_configs]}")
                return {}

    # Mesh for Data Parallelism
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))
    print(f"Mesh: {mesh}")

    for name, args_model in configs:
        print(f"\nTraining {name} on {ds_name}...")
        
        # Replicated State Sharding
        replicated_sharding = NamedSharding(mesh, P())
        
        with mesh:
             # Initialize model (replicated)
             rngs = nnx.Rngs(args.seed)
             if name == 'FullSoftmax':
                 model = FullSoftmaxXML(n_feat, n_lab, args.embed_dim, rngs=rngs)
             elif args_model.get('exact'):
                  if args_model.get('spherical'):
                      model = ExactSphericalYatXML(n_feat, n_lab, args.embed_dim, rngs=rngs)
                  else:
                      model = ExactYatXML(n_feat, n_lab, args.embed_dim, rngs=rngs)
             else:
                 try:
                     model = KernelXML(n_feat, n_lab, args.embed_dim, **args_model, rngs=rngs)
                 except ValueError as e:
                     print(f"Skipping {name}: {e}")
                     continue
            
             # Create Optimizer with Gradient Clipping
             optimizer = nnx.Optimizer(model, optax.chain(
                 optax.clip_by_global_norm(1.0),
                 optax.adam(args.lr)
             ), wrt=nnx.Param)

             @nnx.jit
             def train_step(model, optimizer, indices, mask, labels, label_mask):
                    def loss_fn(m):
                        return m.loss(indices, mask, labels, label_mask)
                    
                    loss, grads = nnx.value_and_grad(loss_fn)(model)
                    optimizer.update(grads)
                    return loss
    
             @nnx.jit
             def predict_step(model, indices, mask):
                 return model.predict(indices, mask, k=5)

             def train_epoch(model, optimizer):
                total_loss = 0
                count = 0
                
                # Data Sharding Spec
                data_sharding = NamedSharding(mesh, P('data', None))
                
                # Use BackgroundGenerator for prefetching
                pbar = BackgroundGenerator(train_ds, max_prefetch=3)
                
                for batch in pbar:
                    # Shard batch explicitly
                    indices = jax.device_put(batch['features'], data_sharding)
                    mask = jax.device_put(batch['masks'], data_sharding)
                    labels = jax.device_put(batch['labels'], data_sharding)
                    label_mask = jax.device_put(batch['label_masks'], data_sharding)
                    
                    loss = train_step(model, optimizer, indices, mask, labels, label_mask)
                    
                    total_loss += loss
                    count += 1
                    
                return total_loss / count
    
             for ep in range(args.epochs):
                t0 = time.time()
                loss = train_epoch(model, optimizer)
                print(f"Ep {ep+1} | Loss: {loss:.4f} | Time: {time.time()-t0:.2f}s")
                    
                # Eval
                all_preds = []
                all_targets = []
                
                # Prefetch test data too
                test_pbar = BackgroundGenerator(test_ds, max_prefetch=3)
                
                for batch in test_pbar:
                    indices = batch['features']
                    mask = batch['masks']
                    # Predict top 5 (JIT Compiled)
                    _, top_k = predict_step(model, indices, mask)
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
    
    # Approx Params
    parser.add_argument("--num_features", type=int, default=32, help="Features for SLAY approx")
    parser.add_argument("--num_quadrature_nodes", type=int, default=2, help="Quad nodes for SLAY")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    results = run_benchmark(args)
    generate_latex(results)
