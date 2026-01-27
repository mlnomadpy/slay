#!/usr/bin/env python3
import os
import sys
import argparse
import time
import zipfile
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from typing import List
from functools import partial

jax.config.update("jax_enable_x64", False)

# --- Helpers ---

def safe_normalize(x, axis=-1, eps=1e-4):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.maximum(norm, eps)

def safe_log(x, eps=1e-8):
    return jnp.log(jnp.maximum(x, eps))

# --- Feature Maps ---

class PerformerFeatures(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 64, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        proj = jax.random.normal(rngs.params(), (self.num_heads, self.head_dim, kernel_size))
        proj = proj / jnp.linalg.norm(proj, axis=1, keepdims=True)
        self.proj_matrix = nnx.Cache(proj)

    def __call__(self, x):
        B = x.shape[0]
        x_reshaped = x.reshape(B, self.num_heads, self.head_dim)
        proj = jnp.einsum('bhd,hdm->bhm', x_reshaped, self.proj_matrix[...])
        features = jax.nn.softplus(proj)
        return features.reshape(B, -1)


class SLAYFeatures(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_features: int = 32, 
                 poly_dim: int = 16, num_quadrature_nodes: int = 2, epsilon: float = 1e-6, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        self.poly_dim = poly_dim
        self.C = 2.0 + epsilon
        
        nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        nodes = np.maximum(nodes, 0.01)
        weights = np.maximum(weights, 0.01)
        self.quad_nodes = nnx.Cache(jnp.array(nodes, dtype=jnp.float32) / self.C)
        self.quad_weights = nnx.Cache(jnp.array(weights, dtype=jnp.float32) / self.C)
        
        k1, k2 = jax.random.split(rngs.params())
        omega = jax.random.normal(k1, (num_quadrature_nodes, self.num_heads, self.head_dim, num_features))
        self.omega = nnx.Cache(omega / jnp.sqrt(self.head_dim))
        
        anchors = jax.random.normal(k2, (poly_dim, self.head_dim))
        self.anchor_vectors = nnx.Cache(anchors / jnp.linalg.norm(anchors, axis=-1, keepdims=True))

    def __call__(self, x):
        B = x.shape[0]
        x_reshaped = x.reshape(B, self.num_heads, self.head_dim)
        x_norm = safe_normalize(x_reshaped, axis=-1)
        
        # Polynomial features
        poly_proj = jnp.clip(jnp.einsum('bhd,pd->bhp', x_norm, self.anchor_vectors[...]), -1.0, 1.0)
        poly_feat = (poly_proj ** 2) / jnp.sqrt(self.poly_dim)
        
        # PRF features
        prf_proj = jnp.einsum('bhd,rhdm->rbhm', x_norm, self.omega[...])
        s_vals = jnp.maximum(self.quad_nodes[...].reshape(-1, 1, 1, 1), 1e-6)
        exp_arg = jnp.clip(prf_proj * jnp.sqrt(2.0 * s_vals) - s_vals, -20.0, 20.0)
        prf_feat = jnp.exp(exp_arg) / jnp.sqrt(self.num_features + 1e-6)
        prf_feat = prf_feat * jnp.sqrt(jnp.maximum(self.quad_weights[...].reshape(-1, 1, 1, 1), 1e-6))
        
        # Tensor product fusion
        out_chunks = []
        for r in range(prf_feat.shape[0]):
            fused_r = poly_feat[:, :, :, None] * prf_feat[r][:, :, None, :]
            out_chunks.append(fused_r.reshape(B, -1))
        return jnp.concatenate(out_chunks, axis=-1)


# --- Configuration ---
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


def download_data(dataset_name):
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return

    info = DATASETS[dataset_name]
    dataset_dir = os.path.dirname(info['train'])
    if os.path.exists(dataset_dir):
        print(f"Dataset {dataset_name} already exists")
        return

    print(f"Downloading {dataset_name}...")
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    output = f"{dataset_name}.zip"
    if not os.path.exists(output):
        gdown.download(f'https://drive.google.com/uc?id={info["id"]}', output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Done.")


def load_xml_data(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        header_line = f.readline()
        header = header_line.decode('utf-8').strip().split()
        num_samples, num_features, num_labels = map(int, header)
        offset = len(header_line)
        
    data = load_svmlight_file(file_path, multilabel=True, n_features=num_features, offset=offset)
    labels = [np.array(l, dtype=np.int32) for l in data[1]]
    print(f"  Loaded {num_samples} samples, {num_features} features, {num_labels} labels")
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
            
        batch_indices = self.indices[self.n:self.n + self.batch_size]
        self.n += self.batch_size
        
        batch_X = self.X[batch_indices]
        max_feats = max(max((len(batch_X[i].indices) for i in range(len(batch_indices))), default=0), 1)
        
        padded_feats = np.zeros((len(batch_indices), max_feats), dtype=np.int32)
        masks = np.zeros((len(batch_indices), max_feats), dtype=np.float32)
        
        for i in range(len(batch_indices)):
            row = batch_X[i]
            length = len(row.indices)
            if length > 0:
                padded_feats[i, :length] = row.indices
                masks[i, :length] = 1.0
             
        MAX_LABELS = 16
        max_labels = max(min(MAX_LABELS, max((len(self.Y[i]) for i in batch_indices), default=1)), 1)
        
        padded_labels = np.full((len(batch_indices), max_labels), -1, dtype=np.int32)
        label_masks = np.zeros((len(batch_indices), max_labels), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            lbls = self.Y[idx]
            if len(lbls) > 0:
                n = min(len(lbls), max_labels)
                padded_labels[i, :n] = lbls[:n]
                label_masks[i, :n] = 1.0
        
        return {
            'features': jnp.array(padded_feats),
            'masks': jnp.array(masks),
            'labels': jnp.array(padded_labels),
            'label_masks': jnp.array(label_masks)
        }


# --- Models with Negative Sampling ---

class MeanEmbedding(nnx.Module):
    def __init__(self, num_features: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(num_features, embed_dim, rngs=rngs)
    
    def __call__(self, indices, mask):
        embeds = self.embedding(indices)
        sum_embeds = jnp.sum(embeds * mask[:, :, None], axis=1)
        return sum_embeds / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)


class SampledSoftmaxXML(nnx.Module):
    """Full softmax with sampled negatives for training efficiency."""
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, 
                 num_negatives: int = 2048, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.num_negatives = num_negatives
        self.num_labels = num_labels
        
    def __call__(self, indices, mask):
        embeds = self.encoder(indices, mask)
        return self.classifier(safe_normalize(embeds, axis=-1))
        
    def loss(self, indices, mask, labels, label_mask, rng_key):
        B = indices.shape[0]
        K = labels.shape[1]
        
        embeds = safe_normalize(self.encoder(indices, mask), axis=-1)  # [B, D]
        W = self.classifier.kernel[...]  # [D, L]
        
        # Sample negatives
        neg_indices = jax.random.randint(rng_key, (self.num_negatives,), 0, self.num_labels)
        
        # Gather weights for positives and negatives
        safe_labels = jnp.maximum(labels, 0)  # [B, K]
        
        # Positive logits: need W[:, labels[b, k]] for each b, k
        W_pos = W[:, safe_labels.reshape(-1)].T.reshape(B, K, -1)  # [B, K, D]
        pos_logits = jnp.sum(embeds[:, None, :] * W_pos, axis=-1)  # [B, K]
        
        # Negative logits
        W_neg = W[:, neg_indices].T  # [N, D]
        neg_logits = embeds @ W_neg.T  # [B, N]
        
        # Combine and compute softmax loss
        all_logits = jnp.concatenate([pos_logits, neg_logits], axis=1)  # [B, K+N]
        log_probs = jax.nn.log_softmax(all_logits, axis=-1)
        
        # Loss is negative log prob of positives
        pos_log_probs = log_probs[:, :K] * label_mask
        return -jnp.sum(pos_log_probs) / (jnp.sum(label_mask) + 1e-6)

    def predict(self, indices, mask, k=5):
        logits = self(indices, mask)
        return jax.lax.top_k(logits, k)


class SampledKernelXML(nnx.Module):
    """Kernel-based XML with negative sampling."""
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, 
                 attention_type: str, num_negatives: int = 2048,
                 attention_kwargs: dict = {}, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.num_negatives = num_negatives
        self.num_labels = num_labels
        
        num_heads = attention_kwargs.get('num_heads', 4)
        if embed_dim % num_heads != 0:
            num_heads = 1
        
        if attention_type in ['slay', 'yat', 'yat-spherical']:
            self.feature_map = SLAYFeatures(embed_dim, num_heads, **attention_kwargs, rngs=rngs)
        elif attention_type == 'performer':
            self.feature_map = PerformerFeatures(embed_dim, num_heads, **attention_kwargs, rngs=rngs)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

    def get_features(self, x):
        return self.feature_map(x)
        
    def loss(self, indices, mask, labels, label_mask, rng_key):
        B = indices.shape[0]
        K = labels.shape[1]
        
        query = safe_normalize(self.encoder(indices, mask), axis=-1)
        phi_query = self.get_features(query)  # [B, M]
        
        W = self.classifier.kernel[...].T  # [L, D]
        W = safe_normalize(W, axis=-1)
        
        # Sample negatives
        neg_indices = jax.random.randint(rng_key, (self.num_negatives,), 0, self.num_labels)
        safe_labels = jnp.maximum(labels, 0)
        
        # Combine all indices we need
        all_needed = jnp.concatenate([safe_labels.reshape(-1), neg_indices])  # [B*K + N]
        W_subset = W[all_needed]  # [B*K + N, D]
        phi_W_subset = self.get_features(W_subset)  # [B*K + N, M]
        
        # Scores
        scores = phi_query @ phi_W_subset.T  # [B, B*K + N]
        
        # Extract positive scores for each sample
        # pos_scores[b] = scores[b, b*K:(b+1)*K]
        idx_offsets = jnp.arange(B)[:, None] * K + jnp.arange(K)[None, :]  # [B, K]
        pos_scores = jnp.take_along_axis(scores, idx_offsets, axis=1)  # [B, K]
        
        # Negative scores (shared)
        neg_scores = scores[:, B*K:]  # [B, N]
        
        all_scores = jnp.concatenate([pos_scores, neg_scores], axis=1) + 1e-8
        log_Z = jax.scipy.special.logsumexp(safe_log(all_scores), axis=1)
        
        log_probs = safe_log(pos_scores + 1e-8) - log_Z[:, None]
        return -jnp.sum(log_probs * label_mask) / (jnp.sum(label_mask) + 1e-6)
        
    def predict(self, indices, mask, k=5):
        query = safe_normalize(self.encoder(indices, mask), axis=-1)
        phi_query = self.get_features(query)
        W = safe_normalize(self.classifier.kernel[...].T, axis=-1)
        phi_W = self.get_features(W)
        scores = phi_query @ phi_W.T
        return jax.lax.top_k(scores, k)


class SampledYatXML(nnx.Module):
    """Exact Yat kernel with negative sampling."""
    def __init__(self, num_features: int, num_labels: int, embed_dim: int,
                 spherical: bool = True, epsilon: float = 0.1, 
                 num_negatives: int = 2048, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs)
        self.epsilon = epsilon
        self.spherical = spherical
        self.C = 2.0 + epsilon
        self.num_negatives = num_negatives
        self.num_labels = num_labels

    def kernel_fn(self, q, k_vecs):
        """Compute kernel between queries and key vectors."""
        dot = jnp.dot(q, k_vecs.T)
        
        if self.spherical:
            dot = jnp.clip(dot, -0.999, 0.999)
            denom = jnp.maximum(self.C - 2.0 * dot, self.epsilon)
        else:
            q_norm2 = jnp.sum(q**2, axis=-1, keepdims=True)
            k_norm2 = jnp.sum(k_vecs**2, axis=-1)
            dist2 = jnp.maximum(q_norm2 + k_norm2[None, :] - 2 * dot, 0.0)
            denom = dist2 + self.epsilon
            
        return (dot ** 2) / denom

    def loss(self, indices, mask, labels, label_mask, rng_key):
        B = indices.shape[0]
        K = labels.shape[1]
        
        query = safe_normalize(self.encoder(indices, mask), axis=-1)
        W = safe_normalize(self.classifier.kernel[...].T, axis=-1)
        
        # Sample negatives
        neg_indices = jax.random.randint(rng_key, (self.num_negatives,), 0, self.num_labels)
        safe_labels = jnp.maximum(labels, 0)
        
        # Gather subset of W
        all_indices = jnp.concatenate([safe_labels.reshape(-1), neg_indices])
        W_subset = W[all_indices]  # [B*K + N, D]
        
        # Compute kernel scores
        scores = self.kernel_fn(query, W_subset)  # [B, B*K + N]
        
        # Extract positives
        idx_offsets = jnp.arange(B)[:, None] * K + jnp.arange(K)[None, :]
        pos_scores = jnp.take_along_axis(scores, idx_offsets, axis=1)
        neg_scores = scores[:, B*K:]
        
        all_scores = jnp.concatenate([pos_scores, neg_scores], axis=1) + 1e-8
        log_Z = jax.scipy.special.logsumexp(safe_log(all_scores), axis=1)
        
        log_probs = safe_log(pos_scores + 1e-8) - log_Z[:, None]
        return -jnp.sum(log_probs * label_mask) / (jnp.sum(label_mask) + 1e-6)
        
    def predict(self, indices, mask, k=5):
        query = safe_normalize(self.encoder(indices, mask), axis=-1)
        W = safe_normalize(self.classifier.kernel[...].T, axis=-1)
        scores = self.kernel_fn(query, W)
        return jax.lax.top_k(scores, k)


# --- Metrics ---

def get_propensity_scores(labels, num_labels, A=0.55, B=1.5):
    N = len(labels)
    freqs = np.zeros(num_labels)
    for i in range(N):
        freqs[labels[i]] += 1
    C = (np.log(N) - 1) * ((B + 1) ** A)
    p = 1.0 / (1.0 + C * ((freqs + B) ** (-A)))
    p[freqs == 0] = 1e-6
    return p

def precision_at_k(targets_list, pred_indices, k=5):
    p_k_sum = np.zeros(k)
    n_samples = len(targets_list)
    pred_indices = np.array(pred_indices)

    for i in range(n_samples):
        t_row = np.array(targets_list[i])
        true_labels = set(t_row[t_row != -1])
        if len(true_labels) == 0:
            continue
        preds = pred_indices[i]
        hits = 0
        for j in range(k):
            if j < len(preds) and preds[j] in true_labels:
                hits += 1
            p_k_sum[j] += hits / (j + 1)
    return p_k_sum / n_samples

def psp_at_k(targets_list, pred_indices, propensity_scores, k=5):
    psp_sum = np.zeros(k)
    n_samples = len(targets_list)
    pred_indices = np.array(pred_indices)
    inv_p = np.zeros_like(propensity_scores)
    inv_p[propensity_scores > 0] = 1.0 / propensity_scores[propensity_scores > 0]

    for i in range(n_samples):
        t_row = np.array(targets_list[i])
        true_labels = set(t_row[t_row != -1])
        if len(true_labels) == 0:
            continue
        preds = pred_indices[i]
        score = 0.0
        for j in range(k):
            if j < len(preds) and preds[j] in true_labels:
                score += inv_p[preds[j]]
            psp_sum[j] += score / (j + 1)
    return psp_sum / n_samples


# --- Runner ---

def run_benchmark(dataset_name: str = "all", method_filter: str = "all"):
    print(f"JAX Devices: {jax.devices()}")

    if dataset_name == "all":
        datasets_to_run = list(DATASETS.keys())
    elif dataset_name in DATASETS:
        datasets_to_run = [dataset_name]
    else:
        print(f"Dataset {dataset_name} not found.")
        return {}

    final_results = {}

    for ds_name in datasets_to_run:
        print(f"\n{'='*50}")
        print(f"Benchmarking on {ds_name}")
        print('='*50)
        download_data(ds_name)
        
        info = DATASETS[ds_name]
        try:
            X_train, Y_train, n_feat, n_lab = load_xml_data(info['train'])
            X_test, Y_test, _, _ = load_xml_data(info['test'])
        except Exception as e:
            print(f"Failed to load data for {ds_name}: {e}")
            continue
            
        n_feat = max(n_feat, X_test.shape[1])
        propensity = get_propensity_scores(Y_train, n_lab)
        
        # Optimized settings for Mac
        BATCH_SIZE = 128 if jax.devices()[0].platform == "metal" else 512
        EMBED_DIM = 256
        LR = 5e-4
        EPOCHS = 10  # Fewer epochs, faster iteration
        NUM_NEGATIVES = min(2048, n_lab // 4)  # Scale negatives with label space
        
        train_ds = XMLDataset(X_train, Y_train, n_lab, BATCH_SIZE, shuffle=True)
        test_ds = XMLDataset(X_test, Y_test, n_lab, BATCH_SIZE, shuffle=False)
        
        ds_results = {}
        
        all_configs = [
            ('Yat (Exact)', {'spherical': False}),
            ('Yat (Spherical)', {'spherical': True}),
            ('Softmax', {}),
            ('Performer', {'attention_type': 'performer', 'attention_kwargs': {'kernel_size': 64}}),
            ('SLAY', {'attention_type': 'slay', 'attention_kwargs': {'num_features': 16, 'num_quadrature_nodes': 2}}),
        ]
        
        if method_filter != "all":
            configs = [c for c in all_configs if method_filter.lower() in c[0].lower()]
            if not configs:
                print(f"No method matched '{method_filter}'")
                continue
        else:
            configs = all_configs

        for name, args in configs:
            print(f"\n--- Training {name} ---")
            rngs = nnx.Rngs(42)
            rng_key = jax.random.PRNGKey(42)
            
            if name == 'Softmax':
                model = SampledSoftmaxXML(n_feat, n_lab, EMBED_DIM, NUM_NEGATIVES, rngs=rngs)
            elif 'Yat' in name:
                model = SampledYatXML(n_feat, n_lab, EMBED_DIM, 
                                      spherical=args.get('spherical', True),
                                      num_negatives=NUM_NEGATIVES, rngs=rngs)
            else:
                try:
                    model = SampledKernelXML(n_feat, n_lab, EMBED_DIM,
                                             num_negatives=NUM_NEGATIVES, **args, rngs=rngs)
                except ValueError as e:
                    print(f"Skipping {name}: {e}")
                    continue
            
            optimizer = nnx.Optimizer(
                model, 
                optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(LR, weight_decay=1e-5)),
                wrt=nnx.Param
            )

            @nnx.jit
            def train_step(model, optimizer, indices, mask, labels, label_mask, rng_key):
                def loss_fn(m):
                    return m.loss(indices, mask, labels, label_mask, rng_key)
                loss, grads = nnx.value_and_grad(loss_fn)(model)
                optimizer.update(model, grads)
                return loss
    
            for ep in range(EPOCHS):
                t0 = time.time()
                total_loss, count = 0.0, 0
                for batch in train_ds:
                    rng_key, subkey = jax.random.split(rng_key)
                    loss = train_step(model, optimizer, 
                                      batch['features'], batch['masks'],
                                      batch['labels'], batch['label_masks'], subkey)
                    if not jnp.isnan(loss):
                        total_loss += float(loss)
                        count += 1
                avg_loss = total_loss / max(count, 1)
                print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")
                
            # Evaluation
            print("  Evaluating...")
            all_preds, all_targets = [], []
            for batch in test_ds:
                _, top_k = model.predict(batch['features'], batch['masks'], k=5)
                all_preds.extend(top_k)
                all_targets.extend(batch['labels'])
                
            pk = precision_at_k(all_targets, all_preds, k=5)
            pspk = psp_at_k(all_targets, all_preds, propensity, k=5)
            
            print(f"  P@1: {pk[0]:.4f}, P@3: {pk[2]:.4f}, P@5: {pk[4]:.4f}")
            print(f"  PSP@1: {pspk[0]:.4f}, PSP@3: {pspk[2]:.4f}, PSP@5: {pspk[4]:.4f}")
            
            ds_results[name] = {
                'P@1': pk[0], 'P@3': pk[2], 'P@5': pk[4],
                'PSP@1': pspk[0], 'PSP@3': pspk[2], 'PSP@5': pspk[4]
            }
            
        final_results[ds_name] = ds_results
        
    return final_results


def generate_latex(results):
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
    print(f"\nLaTeX table saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Eurlex-4K", 
                        help="Dataset name or 'all'")
    parser.add_argument("--method", type=str, default="all", 
                        help="Method filter (e.g., 'yat', 'performer', or 'all')")
    args = parser.parse_args()
    
    results = run_benchmark(args.dataset, args.method)
    if results:
        generate_latex(results)
