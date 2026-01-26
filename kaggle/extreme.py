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
import flax.nnx as nnx
import optax
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Try to import attention classes from main.py
try:
    from main import (
        StandardAttention,
        YatAttention,
        YatSphericalAttention,
        SLAYAttention,
        SLAYRMAttention,
        SLAYNystromAttention,
        SLAYAnchorAttention,
        safe_normalize,
    )
    # Check for FastAttention / Performer
    try:
        from main import FastAttention
    except ImportError:
        FastAttention = None
        
except ImportError:
    # Fallback for script execution or interactive mode
    try:
        # If __file__ is defined (script mode)
        sys.path.append(os.path.dirname(__file__))
    except NameError:
        # Interactive mode / Jupyter: assume we are in the directory or need to add 'kaggle'
        # Try adding current directory and 'kaggle' subdirectory
        sys.path.append(os.getcwd())
        sys.path.append(os.path.join(os.getcwd(), 'kaggle'))
    
    try:
        from main import (
            StandardAttention,
            YatAttention,
            YatSphericalAttention,
            SLAYAttention,
            SLAYRMAttention,
            SLAYNystromAttention,
            SLAYAnchorAttention,
            safe_normalize,
        )
        try:
             from main import FastAttention
        except ImportError:
            FastAttention = None
    except ImportError:
        # Last ditch: check if we can find main.py in common locations
        print("Warning: standard import failed. Checking relative paths...")
        if os.path.exists("kaggle/main.py"):
             sys.path.append("kaggle")
        elif os.path.exists("../kaggle/main.py"):
             sys.path.append("../kaggle")
             
        try:
            from main import (
                StandardAttention,
                YatAttention,
                YatSphericalAttention,
                SLAYAttention,
                SLAYRMAttention,
                SLAYNystromAttention,
                SLAYAnchorAttention,
                safe_normalize,
            )
            try:
                 from main import FastAttention
            except ImportError:
                 FastAttention = None
        except ImportError as e:
            print(f"Error: Could not import logic from main.py. {e}")
            print(f"Current Path: {sys.path}")
            print(f"CWD: {os.getcwd()}")
            # Define dummies if needed or exit
            sys.exit(1)


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
        # 1. Feature Bag: indices and offsets
        batch_X = self.X[batch_indices]
        
        # JAX requires fixed shapes or padding. For simplicity here with EmbeddingBag logic,
        # we can flatten features and provide offsets.
        # However, nnx/JAX usually prefers padded arrays.
        # Let's use a padded strategy: Max features per doc in batch? Or simplified mean embedding logic.
        # For simplicity and JAX friendliness: Pre-compute mean embedding on CPU? 
        # No, let's just pad indices to max_len in this batch.
        
        max_feats = max((len(batch_X[i].indices) for i in range(len(batch_indices))), default=0)
        padded_feats = np.zeros((len(batch_indices), max_feats), dtype=np.int32)
        masks = np.zeros((len(batch_indices), max_feats), dtype=np.float32)
        
        for i, row_idx in enumerate(range(len(batch_indices))): # iterate within batch
             # manual sparse row access
             row = batch_X[i] # batch_X is sliced CSR
             indices = row.indices
             length = len(indices)
             padded_feats[i, :length] = indices
             masks[i, :length] = 1.0
             
        # Y labels: List of arrays
        batch_Y = [self.Y[i] for i in batch_indices]
        
        return {
            'features': jnp.array(padded_feats),
            'masks': jnp.array(masks),
            'labels': batch_Y # Keep as list for now, easy to handle in loop or specialized collation
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
        
    def loss(self, indices, mask, labels_list):
        logits = self(indices, mask) # [B, NumLabels]
        log_probs = nnx.log_softmax(logits, axis=-1)
        
        loss_total = 0.0
        # Iterate over batch (slow in JAX if not careful, but for benchmarking we can loop or use vmap?)
        # Since labels_list is variable length, standard vectorization is hard.
        # We'll use a mask approach or simple loop for now (XML usually has few labels per doc).
        # Better: Create a sparse target matrix or padded target matrix.
        
        # Construct multi-hot targets
        B = logits.shape[0]
        L = logits.shape[1]
        targets = jnp.zeros((B, L))
        for i, lbls in enumerate(labels_list):
            if len(lbls) > 0:
                targets = targets.at[i, lbls].set(1.0)
                
        # For multi-label, separate BCE is standard, or standard Softmax if we treat as probability distribution?
        # The PyTorch baseline used: -sum(log_probs[pos])
        # This implies we want to maximize prob of ANY positive label? Or all?
        # Usually for XML: One-vs-All (BCE) or PLT.
        # The user's PyTorch code used: log_probs[pos]. So it's "categorical cross entropy" style but summing over multiple positives.
        
        multilabel_loss = -jnp.sum(targets * log_probs) / B
        return multilabel_loss


class KernelXML(nnx.Module):
    """
    Approximation-based XML using Attention Kernels.
    Z ~ Phi(q) . Sum(Phi(W))
    P(y|q) ~ Phi(q) . Phi(w_y) / Z
    """
    def __init__(self, num_features: int, num_labels: int, embed_dim: int, attention_type: str, attention_kwargs: dict = {}, *, rngs: nnx.Rngs):
        self.encoder = MeanEmbedding(num_features, embed_dim, rngs=rngs)
        self.classifier = nnx.Linear(embed_dim, num_labels, use_bias=False, rngs=rngs) # W matrix
        
        # Instantiate the Attention class to get the feature map logic
        # We pass dummy args where needed used during init
        dummy_rngs = nnx.Rngs(0) 
        
        if attention_type == 'yat':
            self.kernel_mod = YatAttention(embed_dim, 1, **attention_kwargs, rngs=rngs)
            self.feature_fn = self.kernel_mod._prf_features # Yat uses prf features
        elif attention_type == 'yat-spherical':
            self.kernel_mod = YatSphericalAttention(embed_dim, 1, **attention_kwargs, rngs=rngs)
            self.feature_fn = self.kernel_mod._prf_features
        elif attention_type == 'slay':
             self.kernel_mod = SLAYAttention(embed_dim, 1, **attention_kwargs, rngs=rngs)
             # SLAY has split features (poly and prf), this is complex for single vector mapping.
             # SLAY uses Hadamard product of two feature maps.
             # phi(x) = poly(x) (*) prf(x) (flattened)
             # We need to define a wrapper to compute this.
             self.feature_fn = self._slay_feature_map
        elif attention_type == 'performer':
             if FastAttention is None: raise ValueError("Performer not available")
             self.kernel_mod = FastAttention(embed_dim, 1, **attention_kwargs, rngs=rngs)
             self.feature_fn = self.kernel_mod._compute_features
             
        else:
             # Fallback or generic
             raise ValueError(f"Unsupported attention type for KernelXML: {attention_type}")

    def _slay_feature_map(self, x):
         # x: [..., D]
         # SLAY expects [Batch, Head, Len, Dim]
         # We reshape to [Batch, 1, 1, D]
         orig_shape = x.shape
         x_in = x.reshape(x.shape[0], 1, 1, x.shape[-1])
         x_norm = safe_normalize(x_in, axis=-1)
         
         q_poly = self.kernel_mod._poly_features(x_norm) # [B, 1, 1, P]
         q_prf = self.kernel_mod._prf_features(x_norm)   # [R, B, 1, 1, M]
         
         # Fuse: einsum 'bhlp,rbhlm->bhlrpm'
         q_fuse = jnp.einsum('bhlp,rbhlm->bhlrpm', q_poly, q_prf)
         # Flatten last dims
         dim_flat = q_fuse.shape[-3] * q_fuse.shape[-2] * q_fuse.shape[-1]
         return q_fuse.reshape(orig_shape[0], dim_flat)

    def get_features(self, x):
        # Adapters for kernel mods that expect [B,H,L,D]
        # Our x is [Batch, Dim]
        # Some kernels like Yat use self._prf_features(x) where x is usually raw (Yat) or normalized?
        # YatAttention call: q_norm = safe_normalize(q)... q_prf = self._prf_features(q_norm)
        # So we should normalize first usually, or check what the kernel does.
        # Yat _prf_features takes x.
        
        orig_shape = x.shape
        # Fake heads/len: [B, 1, 1, D]
        x_in = x.reshape(x.shape[0], 1, 1, x.shape[-1])
        x_norm = safe_normalize(x_in, axis=-1)
        
        if hasattr(self, '_slay_feature_map') and self.feature_fn == self._slay_feature_map:
             return self._slay_feature_map(x) # implementation above handles reshapes
        
        # Generic wrapper for others (Yat, Performer)
        feats = self.feature_fn(x_norm) # [B, 1, 1, M] (or [R, ...])
        
        # Flatten
        return feats.reshape(orig_shape[0], -1)

    def loss(self, indices, mask, labels_list):
        B = indices.shape[0]
        # 1. Encode Query
        query = self.encoder(indices, mask) # [B, D]
        phi_query = self.get_features(query) # [B, M]
        
        # 2. Compute Global Denominator Sum(phi(W))
        # This is expensive to do every step if W is huge (Labels=4K for Eurlex).
        # But for Eurlex (4k) it's mostly fine on GPU.
        # W: [L, D]
        # Access kernel using index to avoid deprecated .value warnings
        W_vecs = self.classifier.kernel[...]
        W_vecs = W_vecs.T # [L, D]
        
        phi_W = self.get_features(W_vecs) # [L, M]
        
        Z_approx_vec = jnp.sum(phi_W, axis=0) # [M]
        
        # Denominator: phi(q) . Z_vec
        # Using abs/relu to ensure positivity if kernel is positive (Yat/Performer-ReLU are)
        # Yat approximation is exp(..), so positive.
        denom = jnp.dot(phi_query, Z_approx_vec) + 1e-6
        log_Z = jnp.log(denom) # [B]
        
        loss_total = 0.0
        
        # For each sample, sum log prob of positives
        for i, lbls in enumerate(labels_list):
            if len(lbls) == 0: continue
            lbls_arr = jnp.array(lbls)
            
            # phi(w_pos)
            # Gather relevant W rows
            w_pos = W_vecs[lbls_arr] # [NumPos, D]
            phi_w_pos = self.get_features(w_pos) # [NumPos, M]
            
            # Numerator: phi(q_i) . phi(w_pos)
            nums = jnp.dot(phi_w_pos, phi_query[i]) + 1e-6
            log_nums = jnp.log(nums)
            
            # Loss = -Sum(log_num - log_Z)
            loss_i = -jnp.sum(log_nums - log_Z[i])
            loss_total += loss_i
            
        return loss_total / B
        
    def predict(self, indices, mask, k=5):
        query = self.encoder(indices, mask)
        
        # Use exact dot product for prediction evaluation? 
        # Or kernel approximation? The user code uses the kernel approx for prediction in Favor.
        # "Note: If training with ReLU features, inference should essentially maximize K(x, y)."
        # Let's use kernel approx to be consistent.
        
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
        true_labels = set(targets_list[i])
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
        ('Yat', {'attention_type': 'yat'}),
        ('YatSpherical', {'attention_type': 'yat-spherical'}),
        # ('Performer', {'attention_type': 'performer'}),
        ('SLAY', {'attention_type': 'slay', 'attention_kwargs': {'num_features': 32, 'num_quadrature_nodes': 2}}),
    ]
    
    for name, args in configs:
        print(f"\nTraining {name}...")
        rngs = nnx.Rngs(0)
        
        if name == 'FullSoftmax':
            model = FullSoftmaxXML(n_feat, n_lab, EMBED_DIM, rngs=rngs)
        else:
            model = KernelXML(n_feat, n_lab, EMBED_DIM, **args, rngs=rngs)
            
        optimizer = nnx.Optimizer(model, optax.adam(LR))
        
        # JIT Step
        @nnx.jit
        def train_step(model, optimizer, indices, mask, batch_labels_tuple):
            # We can't pass list of lists to JIT easily. 
            # We must handle loss function inside python or pass a padded matrix.
            # For this benchmark, let's keep loss outside JIT or use padded matrix for labels.
            # Padded matrix strategy:
            pass # TODO
            
        # Due to variable length labels, we might use a purely python loop for loss summation
        # OR pad labels to max_labels_per_doc.
        
        def train_epoch(model, optimizer):
            total_loss = 0
            count = 0
            for batch in train_ds:
                indices = batch['features']
                mask = batch['masks']
                labels = batch['labels']
                
                # Grad of loss
                def loss_fn(m):
                    return m.loss(indices, mask, labels)
                
                grad_fn = nnx.value_and_grad(loss_fn)
                loss, grads = grad_fn(model)
                optimizer.update(model, grads) # Fix for recent flax
                
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
