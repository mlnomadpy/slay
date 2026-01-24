# Synthetic Task Benchmark Suite

## Overview

We evaluate attention mechanisms on a comprehensive suite of **22 synthetic tasks** designed to isolate and measure attention-specific capabilities. Unlike standard language modeling benchmarks that conflate attention quality with other model components, our tasks use **attention-only architectures** (no FFN, no LayerNorm) to directly measure how well each attention variant performs core attention operations.

## Methodology

### Model Architecture

```
Input → Embedding + Positional Embedding → [Attention + Residual] × L → Linear Head → Output
```

- **No Feed-Forward Networks**: Eliminates FFN's ability to compensate for weak attention
- **No Layer Normalization**: Removes normalization artifacts that could mask attention differences
- **Pure Residual Connections**: Only `h = h + Attention(h)`
- **Multiple Layers**: Default L=2 to test attention composition

### Evaluation Protocol

- **Multiple Seeds**: 3 random seeds per (task, attention) pair for statistical significance
- **Fresh Data Each Epoch**: Data generated on-the-fly to prevent memorization
- **Held-out Evaluation**: Separate evaluation batches (2× training batch size)
- **Metrics**: Final accuracy (mean ± std across seeds)

### Hyperparameters

| Parameter | Default | Quick Mode |
|-----------|---------|------------|
| Embedding dim | 64 | 64 |
| Attention heads | 4 | 4 |
| Layers | 2 | 2 |
| Batch size | 32 | 16 |
| Epochs | 200 | 50 |
| Seeds | 3 | 1 |
| Learning rate | 1e-3 | 1e-3 |

---

## Task Categories

### 1. Basic Sequence Operations (3 tasks)

Tasks testing fundamental sequence manipulation capabilities.

#### Copy
- **Description**: Reproduce the input sequence exactly
- **Input**: Random tokens `[a, b, c, d, ...]`
- **Target**: Same sequence `[a, b, c, d, ...]`
- **Sequence Length**: 32
- **Vocabulary**: 32 tokens
- **Tests**: Basic information routing through attention

#### Sort
- **Description**: Output the input sequence in sorted order
- **Input**: Random tokens `[5, 2, 8, 1, ...]`
- **Target**: Sorted sequence `[1, 2, 5, 8, ...]`
- **Sequence Length**: 16
- **Vocabulary**: 32 tokens
- **Tests**: Global comparison and reordering via attention

#### Reverse
- **Description**: Output the input sequence in reverse order
- **Input**: `[a, b, c, d]`
- **Target**: `[d, c, b, a]`
- **Sequence Length**: 32
- **Vocabulary**: 32 tokens
- **Tests**: Position-aware information routing

---

### 2. Memory & Retrieval (4 tasks)

Tasks requiring selective memory access and retrieval.

#### Retrieval (Needle in Haystack)
- **Description**: Locate a special "needle" token among distractors
- **Input**: `[noise..., NEEDLE, noise..., QUERY]`
- **Target**: Predict NEEDLE at query position
- **Sequence Length**: 64
- **Vocabulary**: 64 tokens
- **Tests**: Sparse attention to single relevant position

#### Key-Value Recall
- **Description**: Store key-value pairs, then retrieve value for queried key
- **Input**: `[k1, v1, k2, v2, ..., SEP, k_query]`
- **Target**: Corresponding value `v_query`
- **Sequence Length**: 22
- **Vocabulary**: 64 tokens
- **Tests**: Associative memory via attention

#### First Token Recall
- **Description**: At the last position, predict what the first token was
- **Input**: `[a, x, y, z, ...]`
- **Target**: `a` at last position
- **Sequence Length**: 64
- **Vocabulary**: 32 tokens
- **Tests**: Long-range retrieval to sequence start

#### Selective Copy
- **Description**: Copy only tokens that follow a MARKER token
- **Input**: `[a, MARKER, b, c, MARKER, d, ...]`
- **Target**: `[b, d, ...]` in second half
- **Sequence Length**: 64
- **Vocabulary**: 32 tokens
- **Tests**: Conditional attention based on markers

---

### 3. Long-Range Dependencies (3 tasks)

Tasks requiring attention across long distances.

#### Long Copy
- **Description**: Copy task with extended sequence length
- **Input**: 256-token random sequence
- **Target**: Same sequence
- **Sequence Length**: 256
- **Vocabulary**: 32 tokens
- **Tests**: Attention quality at longer ranges

#### Distant Match
- **Description**: Token at position L/2 must match token at position 0
- **Input**: `[a, x, y, z, ...]`
- **Target**: `a` at position L/2
- **Sequence Length**: 128
- **Vocabulary**: 32 tokens
- **Tests**: Single long-range dependency

#### Multi-Hop Reasoning
- **Description**: Follow a chain of pointers: pos0 → pos1 → pos2
- **Input**: Sequence with pointer tokens at positions 0, L/3, 2L/3
- **Target**: Value after the second pointer
- **Sequence Length**: 64
- **Vocabulary**: 32 tokens
- **Tests**: Composing multiple attention hops

---

### 4. Reasoning & State Tracking (3 tasks)

Tasks requiring stateful computation through attention.

#### Stack Operations
- **Description**: Simulate PUSH/POP on a stack
- **Input**: `[PUSH, val1, PUSH, val2, POP, ...]`
- **Target**: Popped values in correct order
- **Sequence Length**: 32
- **Vocabulary**: 32 tokens
- **Tests**: LIFO state tracking via attention

#### Induction Heads
- **Description**: Pattern completion: `[A, B, ..., A]` → predict `B`
- **Input**: Sequence with repeated (trigger, following) pair
- **Target**: Following token after trigger reappears
- **Sequence Length**: 64
- **Vocabulary**: 64 tokens
- **Tests**: In-context pattern matching (key transformer capability)

#### Pattern Completion
- **Description**: Predict next token in repeating pattern
- **Input**: `[a, b, c, d, a, b, c, d, a, b, ...]`
- **Target**: Next token in cycle
- **Sequence Length**: 32
- **Vocabulary**: 16 tokens
- **Tests**: Periodic pattern recognition

---

### 5. Arithmetic & Counting (4 tasks)

Tasks requiring numerical computation through attention.

#### Counting
- **Description**: Count occurrences of token 0 in sequence
- **Input**: `[0, 3, 0, 5, 0, 2, ...]`
- **Target**: Count (e.g., 3) at last position
- **Sequence Length**: 32
- **Vocabulary**: 32 tokens (output clamped)
- **Tests**: Aggregation via attention

#### Parity
- **Description**: Is the count of 1s even (0) or odd (1)?
- **Input**: Binary sequence `[0, 1, 1, 0, 1, ...]`
- **Target**: Parity bit at last position
- **Sequence Length**: 32
- **Vocabulary**: 2 tokens
- **Tests**: Global binary aggregation

#### Addition
- **Description**: Sum of binary digits (mod vocabulary size)
- **Input**: Binary sequence
- **Target**: Sum mod V
- **Sequence Length**: 16
- **Vocabulary**: 32 tokens
- **Tests**: Arithmetic aggregation

#### Modular Arithmetic
- **Description**: Sum of tokens modulo small prime (7)
- **Input**: Tokens in [0, 6]
- **Target**: Sum mod 7
- **Sequence Length**: 16
- **Vocabulary**: 8 tokens
- **Tests**: Modular arithmetic via attention

---

### 6. Pattern Learning (2 tasks)

Tasks requiring learning statistical patterns.

#### Bigram Prediction
- **Description**: Predict next token based on fixed bigram table
- **Input**: Random sequence
- **Target**: `bigram_table[input]` applied elementwise
- **Sequence Length**: 32
- **Vocabulary**: 32 tokens
- **Tests**: Learning position-local patterns

#### Majority Vote
- **Description**: Predict the most frequent token in sequence
- **Input**: Tokens from small vocabulary with one majority
- **Target**: Most frequent token
- **Sequence Length**: 32
- **Vocabulary**: 8 tokens
- **Tests**: Global frequency estimation

---

### 7. Robustness (2 tasks)

Tasks testing robustness to noise and redundancy.

#### Noisy Copy
- **Description**: Copy first half while ignoring noise tokens in second half
- **Input**: `[a, b, c, ..., NOISE, x, NOISE, ...]`
- **Target**: `[a, b, c, ...]`
- **Sequence Length**: 64
- **Vocabulary**: 32 tokens
- **Tests**: Selective attention ignoring noise

#### Compression
- **Description**: Remove consecutive duplicates
- **Input**: `[a, a, a, b, b, c, c, c, c, ...]`
- **Target**: `[a, b, c, ...]`
- **Sequence Length**: 32
- **Vocabulary**: 16 tokens
- **Tests**: Detecting and collapsing runs

---

### 8. Aggregation (1 task)

#### Histogram
- **Description**: Output count of each token type
- **Input**: Sequence with tokens from small vocabulary
- **Target**: Histogram counts in first V positions
- **Sequence Length**: 32
- **Vocabulary**: 16 tokens
- **Tests**: Multi-class counting via attention

---

## Task Summary Table

| Task | Category | Seq Len | Vocab | Primary Test |
|------|----------|---------|-------|--------------|
| copy | Basic | 32 | 32 | Information routing |
| sort | Basic | 16 | 32 | Global reordering |
| reverse | Basic | 32 | 32 | Position-aware routing |
| retrieval | Memory | 64 | 64 | Sparse retrieval |
| kv_recall | Memory | 22 | 64 | Associative memory |
| first_token | Memory | 64 | 32 | Long-range retrieval |
| selective_copy | Memory | 64 | 32 | Conditional attention |
| long_copy | Long-Range | 256 | 32 | Extended range |
| distant_match | Long-Range | 128 | 32 | Single long dependency |
| multihop | Long-Range | 64 | 32 | Composed attention |
| stack | Reasoning | 32 | 32 | State tracking |
| induction | Reasoning | 64 | 64 | Pattern matching |
| pattern | Reasoning | 32 | 16 | Periodic patterns |
| counting | Arithmetic | 32 | 32 | Aggregation |
| parity | Arithmetic | 32 | 2 | Binary aggregation |
| addition | Arithmetic | 16 | 32 | Arithmetic sum |
| modular | Arithmetic | 16 | 8 | Modular arithmetic |
| bigram | Pattern | 32 | 32 | Local patterns |
| majority | Pattern | 32 | 8 | Frequency estimation |
| noisy_copy | Robustness | 64 | 32 | Noise filtering |
| compression | Robustness | 32 | 16 | Run detection |
| histogram | Aggregation | 32 | 16 | Multi-class counting |

---

## Attention-Critical Tasks

The following subset of tasks are specifically designed to **require attention** and cannot be solved by position-only or FFN-based shortcuts:

- **copy** - Requires routing information from input to output positions
- **retrieval** - Requires attending to single relevant position
- **kv_recall** - Requires key-value matching via attention
- **first_token** - Requires long-range attention to position 0
- **selective_copy** - Requires conditional attention to marked positions
- **long_copy** - Requires attention across 256 positions
- **distant_match** - Requires single long-range attention
- **multihop** - Requires composing multiple attention operations
- **induction** - Requires in-context pattern matching (classic attention test)

Use `--attention-only` flag to run only these tasks:
```bash
python tests/benchmark_tasks.py --device cuda --attention-only
```

---

## Expected Results

A well-functioning attention mechanism should achieve:

| Task Type | Expected Accuracy |
|-----------|-------------------|
| Copy-like tasks | >95% |
| Retrieval tasks | >90% |
| Induction | >85% |
| Arithmetic | >70% |
| Complex reasoning | >60% |

Poor approximations will show:
- Degraded performance on long-range tasks (long_copy, distant_match)
- Failure on retrieval tasks (low sparse attention quality)
- Poor induction (inability to form proper attention patterns)

---

## Usage

```bash
# Full suite (22 tasks, 3 seeds, ~2-3 hours on GPU)
python tests/benchmark_tasks.py --device cuda

# Quick validation (~15 min)
python tests/benchmark_tasks.py --device cuda --quick

# Single category
python tests/benchmark_tasks.py --device cuda --category memory

# Specific tasks
python tests/benchmark_tasks.py --device cuda --tasks copy retrieval induction

# Attention-critical tasks only
python tests/benchmark_tasks.py --device cuda --attention-only

# Custom attention variants
python tests/benchmark_tasks.py --device cuda \
    --attentions standard yat-performer-anchor yat-performer-laplace
```

## Output Files

- `tables/task_*.tex` - LaTeX tables per category
- `tables/task_performance.tex` - Combined LaTeX table
- `artifacts/task_results.json` - Full results with config
- `artifacts/task_performance.md` - Markdown summary

---

## Citation

If using this benchmark suite, please cite:

```bibtex
@article{slay2024,
  title={Spherical Linearized Attention with Yat-Kernel},
  author={...},
  year={2024}
}
```
