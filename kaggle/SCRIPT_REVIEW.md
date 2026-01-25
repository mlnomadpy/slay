# Full Code Review: `kaggle/main.py`

## Overview

This script implements a **MiniBERT MLM (Masked Language Model) pre-training** pipeline using **JAX/Flax NNX** with support for **14 different attention mechanisms**, including the novel SLAY (Spherical Linearized Attention with Yat-Kernel) variants.

| Metric | Value |
|--------|-------|
| Total Lines | ~1,500 |
| Attention Types | 14 |
| Framework | JAX + Flax NNX + Optax |
| Task | Masked Language Modeling (BERT-style) |
| Dataset | FineWeb (HuggingFace streaming) |
| Tokenizer | RoBERTa (pretrained) |

---

## Architecture Summary

```
MiniBERT
├── TokenEmbedding (vocab → embed_dim)
├── N × ModernTransformerBlock
│   ├── RMSNorm + Attention + Dropout (Pre-Norm)
│   └── RMSNorm + FFN (GELU) + Dropout
├── RMSNorm (final)
└── Tied Output Layer (embed weights transposed)
```

---

## ✅ Strengths

### 1. **Rich Attention Mechanism Library**
The script provides a comprehensive suite of attention mechanisms:

| Category | Mechanisms |
|----------|------------|
| Standard | `rotary`, `standard` |
| Linear (O(n)) | `linear` (ELU+1), `performer` (FAVOR+), `cosformer`, `rff` |
| Yat Kernel | `yat`, `yat-spherical` |
| SLAY Variants | `slay`, `slay-tensor`, `slay-laplace`, `slay-rm`, `slay-nystrom`, `slay-anchor` |

### 2. **Modern Training Practices**
- ✅ **Warmup + Cosine Decay LR Schedule**
- ✅ **Gradient Clipping** (`max_grad_norm=1.0`)
- ✅ **Weight Decay** (`AdamW` with `weight_decay=0.01`)
- ✅ **Pre-Norm Architecture** (more stable training)
- ✅ **RMSNorm** (more efficient than LayerNorm)
- ✅ **Tied Embeddings** (input/output share weights)
- ✅ **Gradient Checkpointing** (`@nnx.remat`)

### 3. **Numerical Stability**
- ✅ `safe_normalize()` helper to prevent NaN in unit normalization
- ✅ Increased epsilon values (`1e-4` instead of `1e-6`) in attention denominators
- ✅ Clipped exponential arguments (`±10`) in SLAY PRF features
- ✅ Proper MLM loss masking (ignores `-100` labels)

### 4. **TPU/Multi-Device Support**
- ✅ Automatic mesh detection for TPU (4×2) and GPU
- ✅ `NamedSharding` for batch parallelism
- ✅ Weight partitioning via `nnx.with_partitioning`

### 5. **Comprehensive CLI**
- ✅ All hyperparameters configurable via `argparse`
- ✅ Attention-specific kwargs properly mapped
- ✅ W&B integration for experiment tracking

---

## ⚠️ Potential Issues & Recommendations

### 1. **No Padding Mask in Attention**
**Issue:** None of the attention mechanisms apply a padding mask. Padded positions (`<pad>`) attend to all positions and are attended by all positions.

**Impact:** 
- Wasted computation on pad tokens
- Potential information leakage from pads

**Recommendation:**
```python
def __call__(self, x, attention_mask=None, ...):
    # ...
    if attention_mask is not None:
        # attention_mask: [B, L] with 1=valid, 0=pad
        mask_value = jnp.finfo(scores.dtype).min
        pad_mask = (1 - attention_mask[:, None, None, :]) * mask_value
        scores = scores + pad_mask
```

### 2. **RoPE Applied to Non-Rotary Attention**
**Issue:** The `RotarySelfAttention` class applies RoPE, but the alternative attention mechanisms (`linear`, `performer`, etc.) receive `freqs_cos`/`freqs_sin` but ignore them.

**Impact:** Position information is lost for non-rotary attention variants.

**Recommendation:** Consider adding optional RoPE to all attention classes:
```python
class LinearAttention(nnx.Module):
    def __init__(self, ..., use_rope: bool = False):
        self.use_rope = use_rope
    
    def __call__(self, x, freqs_cos=None, freqs_sin=None, ...):
        # Apply RoPE to q, k before feature mapping if use_rope=True
        if self.use_rope and freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
```

### 3. **Validation Set Resampling**
**Issue:** The validation set is created with `full_dataset.take(val_set_size)` from a streaming dataset, but `process_dataset_for_mlm` applies random masking. Each evaluation uses different masks.

**Impact:** Validation loss variance is higher than necessary.

**Recommendation:** Cache the validation set with fixed masks, or use a fixed seed for validation masking.

### 4. **No Dropout in Attention**
**Issue:** Dropout is applied after attention and FFN, but not within attention (no attention dropout).

**Impact:** May lead to overfitting on attention patterns.

**Recommendation:**
```python
attn_weights = jax.nn.softmax(scores, axis=-1)
if training:
    attn_weights = nnx.Dropout(rate=0.1)(attn_weights, deterministic=False)
```

### 5. **Special Tokens in MLM Masking**
**Status:** ✅ Fixed - `create_masked_lm_predictions` now protects BOS/EOS tokens.

**However:** The function signature doesn't receive BOS/EOS token IDs from the tokenizer automatically. Verify they're passed correctly for RoBERTa (`<s>` and `</s>`).

### 6. **Memory Efficiency for Long Sequences**
**Issue:** Full attention (`O(n²)`) mechanisms (`rotary`, `standard`, `yat`, `yat-spherical`) may OOM for `maxlen=1024` with large batch sizes.

**Recommendation:** 
- Use linear attention variants for long sequences
- Add chunked attention for softmax-based methods
- Consider FlashAttention-style implementation

### 7. **Unused `decode` Parameter**
**Issue:** All attention classes accept `decode: bool = False` but never use it.

**Impact:** No issue for MLM (bidirectional), but the parameter is misleading.

**Recommendation:** Remove the parameter or implement proper KV-caching for autoregressive decoding.

---

## 🔧 Code Quality Issues

### 1. **Duplicate Code in Attention Classes**
Many attention classes share identical:
- QKV projection
- Reshape/transpose logic
- Output projection

**Recommendation:** Create a base class:
```python
class BaseAttention(nnx.Module):
    def __init__(self, embed_dim, num_heads, *, rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, rngs=rngs)
        self.out = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
    
    def _split_heads(self, x, B, L):
        return x.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x, B, L, E):
        return x.transpose(0, 2, 1, 3).reshape(B, L, E)
```

### 2. **Magic Numbers**
Several hardcoded values should be configurable or named constants:
- `eps=1e-4` (attention denominator)
- `a_min=-10.0, a_max=10.0` (exp clipping)
- `C = 2.0 + epsilon` (SLAY constant)

### 3. **Inconsistent Feature Normalization**
Some classes normalize features by `sqrt(num_features)`, others by `sqrt(poly_dim)`. This inconsistency could affect kernel approximation quality.

---

## 📊 Performance Considerations

### Memory Usage (per batch, seq_len=1024, embed_dim=768)

| Attention Type | Memory Complexity | Estimated Memory |
|----------------|-------------------|------------------|
| `standard`, `rotary`, `yat`, `yat-spherical` | O(n²) | ~4GB @ batch=16 |
| `linear`, `performer`, `cosformer`, `rff` | O(n) | ~500MB @ batch=16 |
| `slay` variants | O(n × F) where F=features | ~1GB @ batch=16 |

### Compute Complexity

| Category | Forward Complexity |
|----------|-------------------|
| Quadratic Attention | O(n² × d) |
| Linear Attention | O(n × d × F) |
| SLAY | O(n × d × R × M) |

Where:
- n = sequence length
- d = head dimension  
- F = feature dimension
- R = quadrature nodes
- M = PRF features

---

## 🧪 Testing Recommendations

### Unit Tests Needed
1. **Attention Output Shapes**: Verify all attention classes output `[B, L, E]`
2. **Gradient Flow**: Ensure no NaN/Inf gradients for edge cases
3. **Kernel Approximation**: Compare SLAY output to exact Yat-spherical for small inputs
4. **Numerical Stability**: Test with very small/large input values

### Integration Tests
1. **Forward Pass**: Full model forward with all attention types
2. **Training Step**: Single step doesn't crash
3. **Checkpoint Save/Load**: Roundtrip model state

### Suggested Test File Structure
```python
# tests/test_kaggle_main.py
def test_attention_output_shapes():
    for attn_type in ['rotary', 'standard', 'linear', ...]:
        # ...

def test_mlm_loss_masking():
    # Verify -100 labels are properly ignored

def test_numerical_stability():
    # Test with extreme values
```

---

## 🚀 Feature Suggestions

### 1. **Mixed Precision Training**
```python
# Add to config
config['use_fp16'] = True

# In train_step
if config['use_fp16']:
    loss_scale = jax.lax.dynamic_loss_scale(...)
```

### 2. **Learning Rate Finder**
Add a utility to find optimal LR:
```python
def find_lr(model, optimizer, data_iter, start_lr=1e-7, end_lr=10, num_steps=100):
    # Exponentially increase LR, record loss
    # Return LR at minimum loss gradient
```

### 3. **Gradient Accumulation**
For larger effective batch sizes:
```python
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
```

### 4. **Early Stopping**
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

### 5. **Model Parallel for Larger Models**
Current code only does data parallelism. For very large models:
```python
mesh = Mesh(devices.reshape(2, 4), ('data', 'model'))
# Shard embedding and FFN across 'model' axis
```

---

## 📁 File Organization Suggestion

Current structure is monolithic. Consider splitting:

```
kaggle/
├── main.py              # Entry point only
├── config.py            # Configuration dataclass
├── data.py              # Dataset processing
├── model/
│   ├── __init__.py
│   ├── embeddings.py    # TokenEmbedding
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── base.py      # BaseAttention
│   │   ├── standard.py  # RotarySelfAttention, StandardAttention
│   │   ├── linear.py    # LinearAttention, FastAttention, etc.
│   │   ├── yat.py       # YatAttention, YatSphericalAttention
│   │   └── slay.py      # All SLAY variants
│   ├── blocks.py        # ModernTransformerBlock
│   └── bert.py          # MiniBERT
├── training/
│   ├── __init__.py
│   ├── loss.py          # loss_fn_mlm
│   └── trainer.py       # train_step, eval_step, main_pretrain
└── utils.py             # safe_normalize, mean_pooling, etc.
```

---

## Summary

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Functionality** | ⭐⭐⭐⭐☆ | Comprehensive attention library, working MLM training |
| **Code Quality** | ⭐⭐⭐☆☆ | Some duplication, magic numbers, monolithic structure |
| **Numerical Stability** | ⭐⭐⭐⭐☆ | Good safeguards added, but no padding mask |
| **Performance** | ⭐⭐⭐⭐☆ | TPU support, gradient checkpointing, but no mixed precision |
| **Documentation** | ⭐⭐☆☆☆ | Minimal docstrings, no type hints on most functions |
| **Testability** | ⭐⭐☆☆☆ | No test file, hard to unit test due to coupling |

### Priority Fixes
1. 🔴 **Add padding mask** to all attention mechanisms
2. 🟠 **Add RoPE option** to linear attention variants
3. 🟡 **Add unit tests** for attention correctness
4. 🟢 **Refactor** into separate modules

---

## Appendix: Attention Mechanism Quick Reference

### Exact (O(n²)) Attention

| Name | Kernel | Notes |
|------|--------|-------|
| `standard` | softmax(QK^T / √d) | Vanilla transformer |
| `rotary` | softmax(QK^T / √d) with RoPE | Position via rotation |
| `yat` | softmax((q·k)² / (‖q‖² + ‖k‖² - 2q·k + ε)) | Yat kernel |
| `yat-spherical` | softmax(x² / (C - 2x)) where x = q̂·k̂ | Spherical Yat |

### Linear (O(n)) Attention

| Name | Feature Map | Notes |
|------|-------------|-------|
| `linear` | φ(x) = ELU(x) + 1 | Simple, fast |
| `performer` | φ(x) = ReLU(Ωx) | FAVOR+ style |
| `cosformer` | φ(x) = ReLU(x) × [cos(wt), sin(wt)] | Position-weighted |
| `rff` | φ(x) = √(2/m) cos(Ωx + b) | Gaussian kernel approx |

### SLAY (Yat Kernel Linearization)

| Name | Polynomial Approx | Notes |
|------|-------------------|-------|
| `slay` | Hadamard (proj²) | Fused poly + PRF |
| `slay-tensor` | TensorSketch | FFT-based, memory efficient |
| `slay-laplace` | None | Laplace factor only |
| `slay-rm` | Random Maclaurin | Unbiased estimator |
| `slay-nystrom` | Nyström | Data-dependent landmarks |
| `slay-anchor` | Anchor features | Simple squared projection |
