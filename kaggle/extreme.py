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
        
        # 3. Tensor Product Fusion
        # poly_feat: [B, H, P]
        # prf_feat:  [R, B, H, M]
        # Output:    [B, H, R, P, M] -> Flatten
        
        # Move R to end for easier fusion logic or just broadcast
        # poly: b h p
        # prf:  r b h m
        fused = jnp.einsum('bhp,rbhm->brhpm', poly_feat, prf_feat)
        
        # Flatten
        output = fused.reshape(B, -1)
        return output

# --- CONFIGURATION ---
GDRIVE_ID = "0B3lPMIHmG6vGU0VTR1pCejFpWjg"
RESOURCE_KEY = "0-SurjZ4z_5Tr38jENzf2Iwg"
DATASET_NAME = "Eurlex"
TRAIN_FILE = "Eurlex/eurlex_train.txt"
TEST_FILE = "Eurlex/eurlex_test.txt"

# --- DATA DOWNLOADER ---
def download_data():
    if os.path.exists(TRAIN_FILE):
        print(f"Dataset already exists at {TRAIN_FILE}")
        return

    print(f"Downloading {DATASET_NAME} from Google Drive...")
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    output = f"{DATASET_NAME}.zip"
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={GDRIVE_ID}'
        if RESOURCE_KEY:
            url += f'&resourcekey={RESOURCE_KEY}'
        gdown.download(url, output, quiet=False)

    print("Extracting...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Done.")

# --- DATA LOADER ---
def load_xml_data(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').strip().split()
        num_samples, num_features, num_labels = map(int, header)

    data = load_svmlight_file(file_path, multilabel=True, n_features=num_features, offset=1)
    # Convert to list of lists for labels
    labels = [np.array(l, dtype=np.int32) for l in data[1]]
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
        self.classifier = nnx.Linear(embed_dim, num_labels, rngs=rngs)
        
    def __call__(self, indices, mask):
        embeds = self.encoder(indices, mask)
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
        targets = targets.at[batch_indices, safe_labels].set(1.0)
        
        # Zero out the ones that were set from padding (index 0) if mask was 0
        # Wait, if label was 0 and valid, we set index 0 to 1. 
        # If label was -1 (invalid) and we mapped to 0, we set index 0 to 1.
        # We need to NOT set index 0 if it came from padding.
        # Alternative: use .add with mask value? targets.at[...].add(label_mask)
        # If duplicated labels, add accumulates? set is usually fine for binary.
        
        # Proper way: use jnp.where in index? no.
        # Use update with mask.
        targets = targets.at[batch_indices, safe_labels].max(label_mask) # max ensures 1.0 if valid
                
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

    def loss(self, indices, mask, labels_list):
        B = indices.shape[0]
        # 1. Encode Query
        query = self.encoder(indices, mask) # [B, D]
        phi_query = self.get_features(query) # [B, M]
        
        # 2. Compute Global Denominator Sum(phi(W))
        W_vecs = self.classifier.kernel[...] # Access value for JAX
        W_vecs = W_vecs.T # [L, D]
        
        phi_W = self.get_features(W_vecs) # [L, M]
        
        Z_approx_vec = jnp.sum(phi_W, axis=0) # [M]
        
        # Denominator: phi(q) . Z_vec
        denom = jnp.dot(phi_query, Z_approx_vec) + 1e-6
        log_Z = jnp.log(denom) # [B]
        
    def loss(self, indices, mask, labels, label_mask):
        B = indices.shape[0]
        # 1. Encode Query
        query = self.encoder(indices, mask) # [B, D]
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
        
        # Handle padding in labels (-1) by clamping to 0 (will face mask later)
        safe_labels = jnp.maximum(labels, 0)
        
        W_pos = W_vecs[safe_labels] # [B, K, D]
        
        # Flatten to apply feature map: [B*K, D]
        W_pos_flat = W_pos.reshape(-1, W_pos.shape[-1])
        phi_w_pos_flat = self.get_features(W_pos_flat) # [B*K, M]
        
        # Reshape back: [B, K, M]
        phi_w_pos = phi_w_pos_flat.reshape(B, labels.shape[1], -1)
        
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
        
        phi_query = self.get_features(query) # [B, M]
        W_vecs = self.classifier.kernel[...]
        W_vecs = W_vecs.T
        phi_W = self.get_features(W_vecs) # [L, M]
        
        scores = phi_query @ phi_W.T
        return jax.lax.top_k(scores, k) # values, indices


# --- METRICS ---
def precision_at_k(targets_list, pred_indices, k=5):
    p_k_sum = 0
    n_samples = len(targets_list)
    pred_indices = np.array(pred_indices)

    for i in range(n_samples):
        # targets_list[i] is likely a JAX array or numpy array with padding -1
        t_row = np.array(targets_list[i]) 
        # Filter out padding (-1)
        true_labels = set(t_row[t_row != -1])
        
        if len(true_labels) == 0: continue
        preds = pred_indices[i][:k]
        hits = len(true_labels.intersection(set(preds)))
        p_k_sum += hits / k
    return p_k_sum / n_samples

# --- RUNNER ---
def run_benchmark(dataset_path: str = "."):
    download_data()
    X_train, Y_train, n_feat, n_lab = load_xml_data(TRAIN_FILE)
    X_test, Y_test, _, _ = load_xml_data(TEST_FILE)
    n_feat = max(n_feat, X_test.shape[1])
    
    # Config
    BATCH_SIZE = 128
    EMBED_DIM = 256
    LR = 1e-3
    EPOCHS = 10
    
    train_ds = XMLDataset(X_train, Y_train, n_lab, BATCH_SIZE, shuffle=True)
    test_ds = XMLDataset(X_test, Y_test, n_lab, BATCH_SIZE, shuffle=False)
    
    results = {}
    
    # Models to test
    configs = [
        ('FullSoftmax', {}),
        ('Performer', {'attention_type': 'performer', 'attention_kwargs': {'kernel_size': 64}}),
        ('Yat', {'attention_type': 'yat', 'attention_kwargs': {'num_features': 32, 'num_quadrature_nodes': 2}}),
        ('YatSpherical', {'attention_type': 'yat-spherical', 'attention_kwargs': {'num_features': 32, 'num_quadrature_nodes': 2, 'epsilon': 1e-2}}),
        ('SLAY', {'attention_type': 'slay', 'attention_kwargs': {'num_features': 32, 'num_quadrature_nodes': 2}}),
    ]
    
    for name, args in configs:
        print(f"\nTraining {name}...")
        rngs = nnx.Rngs(0)
        
        if name == 'FullSoftmax':
            model = FullSoftmaxXML(n_feat, n_lab, EMBED_DIM, rngs=rngs)
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
            _, top_k = model.predict(indices, mask, k=5)
            all_preds.extend(top_k)
            all_targets.extend(batch['labels'])
            
        score = precision_at_k(all_targets, all_preds, k=5)
        print(f"Final P@5: {score:.4f}")
        results[name] = score
        
    return results

def generate_latex(results):
    print("\nGenerating LaTeX Table...")
    path = "extreme_results.tex"
    with open(path, "w") as f:
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tabular}{lc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & P@5 \\" + "\n")
        f.write(r"\midrule" + "\n")
        for method, score in results.items():
            f.write(f"{method} & {score:.4f} \\\\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Extreme Classification Results on Eurlex}" + "\n")
        f.write(r"\label{tab:extreme_results}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Table saved to {path}")

if __name__ == "__main__":
    results = run_benchmark()
    generate_latex(results)
