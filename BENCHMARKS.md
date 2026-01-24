# SLAY Benchmarking Plan

This document outlines the comprehensive benchmarking strategy for the SLAY paper (Spherical Linearized Attention with Yat-Kernel).

## Overview

### Attention Variants to Benchmark

| Category | Name | Description | Complexity |
|----------|------|-------------|------------|
| **Baselines** | `standard` | Softmax attention | O(L²) |
| | `performer` | FAVOR+ (ReLU random features) | O(L) |
| | `linear` | ELU+1 linear attention | O(L) |
| | `cosformer` | Cosformer with cos reweighting | O(L) |
| **Exact Yat** | `yat-spherical` | Exact spherical Yat (kernel-normalized) | O(L²) |
| **Linearized Yat** | `yat-performer-anchor` | Anchor polynomial features (default) | O(L) |
| | `yat-performer-laplace` | Laplace-only (no poly factor) | O(L) |
| | `yat-performer` | Hadamard fusion (shared ω) | O(L) |
| | `yat-performer-tensor` | TensorSketch polynomial | O(L) |
| | `yat-performer-rm` | Random Maclaurin polynomial | O(L) |
| | `yat-performer-nystrom` | Nyström polynomial | O(L) |

---

## Experiment 1: Kernel Approximation Quality

**Goal**: Validate that linearized versions approximate the exact spherical Yat kernel.

**Metrics**:
- Relative L2 error: `||y_approx - y_exact|| / ||y_exact||`
- Cosine similarity: `cos(y_approx, y_exact)`
- MSE: `mean((y_approx - y_exact)²)`

**Protocol**:
1. Fix QKV and output projection weights across all variants
2. Feed identical random inputs
3. Compare outputs against exact spherical Yat attention

**Variables to sweep**:
- Feature budget (R × M × P)
- Sequence length T
- Embedding dimension D

**Command**:
```bash
python tests/ablation_poly_approx.py --sweep --device cuda
```

**Output**: `tables/poly_ablation_sweep.tex`

---

## Experiment 2: Latency Scaling

**Goal**: Demonstrate O(L) vs O(L²) complexity empirically.

**Protocol**:
1. Single attention layer benchmark (no full model overhead)
2. Forward-only and forward+backward passes
3. Multiple iterations with warmup

**Sequence lengths**: 256, 512, 1024, 2048, 4096, 8192, 16384, 32768

**Command**:
```bash
python tests/benchmark_scaling.py --device cuda --mode latency
```

**Output**: `tables/latency_scaling.tex`, `figures/latency_scaling.pdf`

---

## Experiment 3: Memory Scaling

**Goal**: Show memory advantage of linearized attention.

**Protocol**:
1. Measure peak GPU memory allocation
2. Forward-only and forward+backward passes
3. Find maximum sequence length before OOM

**Command**:
```bash
python tests/benchmark_scaling.py --device cuda --mode memory
```

**Output**: `tables/memory_scaling.tex`, `figures/memory_scaling.pdf`

---

## Experiment 4: Feature Budget Trade-off

**Goal**: Understand accuracy vs. speed/memory trade-off.

**Variables**:
- R (quadrature nodes): 1, 2, 4, 8
- M (PRF features): 8, 16, 32, 64
- P (polynomial dim): 8, 16, 32, 64

**Metrics**:
- Kernel approximation error
- Latency (ms)
- Peak memory (MB)

**Command**:
```bash
python tests/benchmark_feature_budget.py --device cuda
```

**Output**: `tables/feature_budget.tex`, `figures/feature_budget_pareto.pdf`

---

## Experiment 5: End-to-End Training

**Goal**: Verify training stability and final quality.

**Models**:
- BERT-style encoder (MLM task)
- GPT-style decoder (LM task)

**Training**:
- 1 epoch on WikiText-103 or similar
- Max sequence length: 2048
- Identical hyperparameters across variants

**Metrics**:
- Training loss curve
- Final perplexity
- Gradient norm statistics

**Command**:
```bash
python main.py --model bert --attention yat-performer-anchor --epochs 1
python main.py --model gpt --attention yat-performer-anchor --epochs 1
```

**Output**: `tables/e2e_training.tex`, `figures/training_curves.pdf`

---

## Experiment 6: Synthetic Task Performance

**Goal**: Test attention quality on controlled tasks.

**Tasks**:
1. **Copy**: Copy input sequence (tests basic information flow)
2. **Sorting**: Sort input sequence (tests global dependencies)
3. **Associative Recall**: Key-value retrieval (tests memory)
4. **Needle in Haystack**: Find specific token in long context

**Metrics**:
- Final loss
- Accuracy (where applicable)
- Training time

**Command**:
```bash
python tests/benchmark_tasks.py --device cuda
```

**Output**: `tables/task_performance.tex`

---

## Experiment 7: Maximum Context Length

**Goal**: Demonstrate long-context capability under fixed memory.

**Protocol**:
1. Fix GPU memory budget (e.g., 80 GB)
2. Binary search for maximum sequence length
3. Inference mode (no gradients)

**Command**:
```bash
python tests/benchmark_max_context.py --device cuda --memory-budget 80
```

**Output**: `tables/max_context.tex`, `figures/memory_vs_seqlen.pdf`

---

## Quick Start

### Run all benchmarks (full):
```bash
python tests/run_all_benchmarks.py --device cuda --full
```

### Run quick validation:
```bash
python tests/run_all_benchmarks.py --device cuda --quick
```

### Generate all paper tables:
```bash
python tests/generate_paper_tables.py
```

---

## Expected Results

Based on the methodology:

| Experiment | Expected Finding |
|------------|------------------|
| Kernel Approx | Anchor ≈ Laplace-only < Hadamard << Nyström/TensorSketch/RM |
| Latency | Linearized ~10-40x faster at L=32768 |
| Memory | Linearized ~2x less memory at L=32768 |
| Training | Comparable perplexity to standard attention |
| Max Context | ~65K tokens (vs ~16K for standard on 80GB) |

---

## Directory Structure

```
slay/
├── tests/
│   ├── ablation_poly_approx.py    # Polynomial approximation ablation
│   ├── benchmark_attention.py     # Speed/memory/task benchmarks
│   ├── benchmark_scaling.py       # NEW: Latency/memory scaling
│   ├── benchmark_tasks.py         # NEW: Synthetic task benchmarks
│   ├── benchmark_max_context.py   # NEW: Maximum context length
│   ├── run_all_benchmarks.py      # NEW: Master benchmark runner
│   └── generate_paper_tables.py   # NEW: LaTeX table generator
├── tables/
│   ├── poly_ablation_sweep.tex
│   ├── latency_scaling.tex
│   ├── memory_scaling.tex
│   └── ...
├── figures/
│   ├── latency_scaling.pdf
│   ├── memory_scaling.pdf
│   └── ...
└── artifacts/
    └── benchmark_results.json     # Raw results for reproducibility
```

---

## Recommended Workflow

1. **Quick validation** (5-10 min):
   ```bash
   python tests/run_all_benchmarks.py --quick
   ```

2. **Full benchmark suite** (2-4 hours):
   ```bash
   python tests/run_all_benchmarks.py --full
   ```

3. **Generate paper artifacts**:
   ```bash
   python tests/generate_paper_tables.py
   ```

4. **Review results**:
   - Check `tables/` for LaTeX tables
   - Check `figures/` for plots
   - Check `artifacts/benchmark_results.json` for raw data
