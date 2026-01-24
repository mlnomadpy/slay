# Co-author review notes (main.tex)

Date: 2026-01-23

## Executive summary
The paper has a compelling idea: enforce a spherical constraint so the YAT/E-product becomes an isotropic kernel in the angular variable, then linearize via a Laplace/Bernstein mixture and approximate with strictly positive random features.

The core narrative is solid, but the current draft has (i) a few *correctness/consistency* issues (most importantly the stated complexity vs the stated tensor-product feature construction), (ii) missing experimental specification (dataset, training details, baselines), and (iii) some template/compilation risks (fontspec/XeLaTeX in ICML style).

Below are the actionable improvements, grouped by priority.

---

## Strengths
- Clear motivation: quadratic attention bottleneck + desire for geometry-aware alternatives.
- Spherical reduction is elegant: kernel depends only on $x=\hat q^\top \hat k\in[-1,1]$.
- Integral representation is clean and standard (Bernstein/Laplace for $1/y$).
- Positivity story is appealing for stability (FAVOR$^+$ angle).
- Theoretical appendix includes key closure facts (mixture/product PD).

---

## Major issues to fix (high priority)

### 1) Complexity claim does not match the proposed feature map
In the method, the per-node feature map is defined as a **tensor (Kronecker) product**
\[
\Psi_r(u)=\sqrt{w_r}\,(\phi_{\text{poly}}(u)\otimes \phi_{\text{PRF}}(u;s_r)).
\]
If $\phi_{\text{poly}}\in\mathbb{R}^{D_p}$ and $\phi_{\text{PRF}}\in\mathbb{R}^{D_r}$, then $\Psi_r\in\mathbb{R}^{D_pD_r}$, and concatenating over $R$ nodes gives total feature dimension
\[
 m = R\,D_pD_r.
\]
But the Complexity section claims time/space $O(L\,R\,(D_p + D_r))$.

This is a **material inconsistency**. Either:
- (A) the main method must *avoid explicit tensor expansion* (needs a concrete computational trick, e.g., structured/sketched tensor product / TensorSketch-style compression for the tensor RKHS), **or**
- (B) the complexity statement must be updated to $O(L\,R\,D_pD_r)$ (and then you need to justify that this is still feasible), **or**
- (C) the main method should switch to a different fusion (Hadamard) and clearly state it targets a different kernel (currently relegated to an appendix baseline).

Recommendation: if you want to keep the “positivity + correct target kernel” story, adopt (A): introduce a *sketched tensor product* feature map so the effective dimension scales like $O(D_p + D_r)$ or $O(\tilde D)$ rather than $D_pD_r$, and spell out the shapes and computation.

### 2) Section numbering / missing sections
The main paper jumps from “4. Theoretical Guarantees” directly to “7. Experiments”. This reads like missing content (likely Sections 5–6).

Recommendation: add the missing sections or renumber to be consistent.

### 3) Experimental details are currently insufficient to be credible/reproducible
Key missing items:
- Dataset(s): what corpus for BERT/GPT? tokenization? number of tokens? preprocessing?
- Training setup: optimizer, LR schedule, batch size, warmup, weight decay, dropout, grad clip, mixed precision, etc.
- Architecture details: number of layers/heads, d_model, d_ff, activation (you mention activation-free motivation but these are Transformer baselines).
- Random feature details: how are $\omega_i$ sampled (Gaussian vs orthogonal)? fixed vs resampled? per-layer/per-head sharing?
- Quadrature: how are Gauss–Laguerre nodes/weights computed? what values of $R$ used?
- Baselines: you compare mostly to “standard attention”. For a linear attention paper, you need Performer / FAVOR+, CosFormer, Linear Transformer, etc.
- Benchmark methodology: framework (PyTorch version), compilation flags, kernel implementation, sequence length scaling, causal prefix-sum implementation specifics.

Recommendation: add a dedicated “Experimental Setup” subsection (in the main paper, not only appendix) that pins down all of the above.


### 5) Claims that need tightening / alignment with what is actually proven
- “Strictly positive random-feature approximation” is true for the exponential PRF features; but the *overall* estimator can become negative if the polynomial feature map is approximate and not positivity-preserving.
- The “denominator positivity” argument relies on non-negativity of the approximate scores. That is guaranteed for the *exact* polynomial kernel value $(q^\top k)^2$ but not automatically for every approximation.

Recommendation: explicitly separate:
1) positivity of the *true kernel*,
2) positivity of the *exact-feature inner product*,
3) conditions under which approximations preserve non-negativity.

---

## Medium-priority improvements (clarity / rigor)

### 6) Clarify the role of normalization in a Transformer block
You normalize queries/keys to unit norm. This changes the model class and can interact with scaling, residuals, and layer norms.

Recommendation: explicitly state:
- where normalization is applied (per-head? per-token? before or after linear projections?),
- whether it is differentiable everywhere (what happens at near-zero norms?),
- whether you use an $\epsilon$ in the normalization itself.

### 7) Kernel/property statements: scope and assumptions
You correctly mention $d\ge 2$ for Schoenberg theory. Still, tighten:
- Make sure PD is stated on $\mathbb{S}^{d-1}$ with the dot-product form.
- When using the series $e^{2sx}=\sum_n (2s)^n x^n/n!$, it’s worth stating that $(q^\top k)^n$ is PD as a polynomial kernel restriction.
- If you interchange integral/sum/expectation anywhere, add a one-liner justification (dominated convergence / nonnegative integrand).

### 8) Quadrature discussion: make it concrete
Right now the Gauss–Laguerre section is conceptually fine, but the paper needs actual values used and how sensitive results are.

Recommendation:
- include a small table of $R\in\{4,8,16\}$ and show accuracy/latency trade-off,
- specify the $\epsilon$ used and how it affects the kernel sharpness.

### 9) Parameter count vs compute overhead
You report ~0.02% parameter overhead. But random features can be fixed (0 params) or learned (many params). Also runtime cost is the bigger issue.

Recommendation:
- state whether random projection matrices are fixed buffers or learned weights,
- report MACs / throughput / wall-clock per token (not only params).

---

## Minor edits / polish

### Writing/structure
- The abstract is strong but slightly dense; consider one sentence explicitly stating what changes vs prior YAT/Performer.
- Define acronyms consistently: YAT vs Yat vs \E-product; choose one naming scheme.
- Ensure “YAT” vs “Yat-kernel” capitalization is consistent.

### Algorithm
- The pseudocode is very high-level. Consider adding shapes or at least indicate feature dimension and what is cached (e.g., $\Psi(K)^\top V$).

### References
- Several bib entries look incomplete / non-standard (e.g., `nmn2026` entry missing venue/year details).
- Consider adding citations on spherical kernels / Schoenberg expansions if you want a more canonical PD argument.

### Appendix completeness
- Some appendix headers are empty placeholders (e.g., “Implementation Notes: Quadrature Scaling and Shapes” is introduced but not populated beyond later subsections).

---

## Suggested concrete additions (if we want to strengthen the paper)

1) Add a “Related Work” section:
- Performer/FAVOR+, linear transformers, cosine/orthogonal features, other geometry-aware attention.

2) Add an ablation table:
- effect of $R$, $D_r$, polynomial-feature choice (exact vs RM vs anchors), and normalization placement.

### Proposed ablation: polynomial approximations (make “we choose the best” defensible)
Goal: justify a *default* polynomial approximation choice under a clear budget/metric, and show how alternatives trade off accuracy vs stability vs compute.

Suggested design (one small table + one small figure):
- **Methods compared:** Exact $(q^\top k)^2$ (value-only), TensorSketch, Random Maclaurin, Nystr\"om, Anchor features.
- **Budgets:** sweep $D_p\in\{64,128,256,512,1024\}$ (or a smaller set), with fixed $R$ and $D_r$.
- **Kernel fidelity metric (cheap sanity check):** sample random unit vectors, estimate kernel MSE on $x\mapsto x^2$ and/or on the full mixture target $x^2 e^{2s x}$ for a representative $s$.
- **Stability metric:** fraction of negative approximated polynomial inner products (or negative full scores) on random pairs; also report minimum/quantiles of the attention denominator per batch.
- **Downstream metric:** validation loss/perplexity at fixed wall-clock or fixed FLOPs.
- **Compute metric:** attention-only latency and memory.

How to phrase the claim safely in the paper:
- “We evaluate several approximations to $(q^\top k)^2$ and use **[X] by default** because it provides the best **[metric]** under a fixed compute budget; other variants are reported in ablations.”
- If you keep the “denominator positivity” claim: “Strict non-negativity is guaranteed when the polynomial component is computed exactly (or via a positivity-preserving approximation); other approximations may violate non-negativity and are treated as baselines.”

3) Add a kernel-approximation sanity plot:
- approximate kernel vs ground truth $x\mapsto x^2/(C-2x)$ for different feature budgets.

4) Add a clear implementation paragraph:
- how to compute the (possibly sketched) tensor product efficiently; this ties directly to the complexity claim.

---

## Quick checklist of edits to make next
- [ ] Fix tensor-product vs complexity mismatch (choose A/B/C and update method + complexity + experiments accordingly).
- [ ] Fix section numbering (add missing sections or renumber).
- [ ] Add missing experimental details + strong linear baselines.
- [ ] Decide compilation target (pdfLaTeX vs XeLaTeX) and adjust fonts accordingly.
- [ ] Tighten positivity/denominator arguments to match approximation choices.
