"""
Denominator Positivity Histogram - ICML Paper Figure

Visualization showing histogram of denominator values across long sequences.
Demonstrates stability of SLAY (anchor features) vs signed polynomial baselines.

Outputs PDF with vector graphics for camera-ready quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import math
from viz_utils import log_data

# ============================================================================
# Publication Settings (ICML)
# ============================================================================
COLUMN_WIDTH = 3.25
FULL_WIDTH = 6.75

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['lines.linewidth'] = 1.5

# Colorblind-safe palette
COLORS = {
    'yat': '#EE7733',           # Orange
    'spherical_yat': '#CC3311', # Red
    'slay_anchor': '#EE3377',   # Magenta
    'tensor_sketch': '#0077BB', # Blue
    'random_maclaurin': '#009988', # Teal
    'nystrom': '#33BBEE',       # Cyan
}

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# Denominator Computation Functions
# ============================================================================
def compute_yat_denominator(q, k, epsilon=1e-2):
    """
    Compute YAT denominator: ||q - k||² + ε
    
    Returns denominators for all (i, j) pairs (causal only).
    """
    B, H, T, D = q.shape
    
    q_sq = (q ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
    k_sq = (k ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
    dots = torch.matmul(q, k.transpose(-1, -2))  # (B, H, T, T)
    
    # ||q - k||² = ||q||² + ||k||² - 2·q·k
    dist_sq = q_sq + k_sq.transpose(-1, -2) - 2 * dots
    denom = dist_sq + epsilon
    
    # Apply causal mask - only keep lower triangular
    causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    denom = denom.masked_select(causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1))
    
    return denom.flatten()


def compute_spherical_yat_denominator(q, k, epsilon=1e-2):
    """
    Compute spherical YAT denominator: C - 2x where x = q̂·k̂, C = 2 + ε
    """
    B, H, T, D = q.shape
    
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    dots = torch.matmul(q_norm, k_norm.transpose(-1, -2))  # (B, H, T, T)
    
    C = 2.0 + epsilon
    denom = C - 2 * dots
    
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    denom = denom.masked_select(causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1))
    
    return denom.flatten()


def compute_slay_anchor_denominator(q, k, num_anchors=32, num_prf=8, num_nodes=2, epsilon=1e-2):
    """
    Compute SLAY (anchor) denominator: φ(Q)(φ(K)^T 1)
    
    Using anchor features for polynomial and PRF for exponential.
    """
    B, H, T, D = q.shape
    
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    # Anchor features for polynomial
    anchors = torch.randn(num_anchors, D, device=q.device, dtype=q.dtype)
    anchors = F.normalize(anchors, p=2, dim=-1)
    
    q_poly = (torch.matmul(q_norm, anchors.T) ** 2) / math.sqrt(num_anchors)  # (B, H, T, P)
    k_poly = (torch.matmul(k_norm, anchors.T) ** 2) / math.sqrt(num_anchors)
    
    # Gauss-Laguerre quadrature
    C = 2.0 + epsilon
    nodes, weights = np.polynomial.laguerre.laggauss(num_nodes)
    s_vals = torch.tensor(nodes / C, dtype=q.dtype, device=q.device)
    w_vals = torch.tensor(weights / C, dtype=q.dtype, device=q.device)
    
    # PRF and accumulate denominator
    omega = torch.randn(num_nodes, D, num_prf, device=q.device, dtype=q.dtype)
    
    total_denom = torch.zeros(B, H, T, device=q.device, dtype=q.dtype)
    
    for r in range(num_nodes):
        s_r = s_vals[r]
        w_r = w_vals[r]
        
        sqrt_2s = torch.sqrt(2.0 * torch.clamp(s_r, min=0))
        
        q_proj = torch.matmul(q_norm, omega[r]) * sqrt_2s - s_r
        k_proj = torch.matmul(k_norm, omega[r]) * sqrt_2s - s_r
        
        q_prf = torch.exp(torch.clamp(q_proj, -20, 20)) / math.sqrt(num_prf)
        k_prf = torch.exp(torch.clamp(k_proj, -20, 20)) / math.sqrt(num_prf)
        
        # Accumulate k features for cumsum (causal)
        k_outer = torch.einsum('bhtp,bhtm->bhtpm', k_poly, k_prf)  # (B, H, T, P, M)
        k_cumsum = torch.cumsum(k_outer, dim=2)  # Causal cumsum
        
        # Dot with query features
        q_outer = torch.einsum('bhtp,bhtm->bhtpm', q_poly, q_prf)
        
        # Sum to get denominator contribution
        denom_r = torch.einsum('bhtpm,bhtpm->bht', q_outer, k_cumsum)
        total_denom += w_r * denom_r
    
    return total_denom.flatten()


def compute_tensor_sketch_denominator(q, k, sketch_dim=64, epsilon=1e-2):
    """
    Compute denominator using TensorSketch polynomial features.
    These are SIGNED and can produce negative values.
    """
    B, H, T, D = q.shape
    
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    # TensorSketch: signed random signs and hash buckets
    signs1 = torch.randint(0, 2, (D,), device=q.device, dtype=q.dtype) * 2 - 1
    signs2 = torch.randint(0, 2, (D,), device=q.device, dtype=q.dtype) * 2 - 1
    hash1 = torch.randint(0, sketch_dim, (D,), device=q.device)
    hash2 = torch.randint(0, sketch_dim, (D,), device=q.device)
    
    def tensor_sketch(x):
        # Apply hash and signs
        sketch = torch.zeros(*x.shape[:-1], sketch_dim, device=x.device, dtype=x.dtype)
        
        for i in range(D):
            b1, b2 = hash1[i], hash2[i]
            s1, s2 = signs1[i], signs2[i]
            
            # Outer product contribution (x_i * x_j) with signs
            for j in range(D):
                b = (hash1[i] + hash2[j]) % sketch_dim
                s = signs1[i] * signs2[j]
                sketch[..., b] += s * x[..., i] * x[..., j]
        
        return sketch / math.sqrt(sketch_dim)
    
    q_sketch = tensor_sketch(q_norm)
    k_sketch = tensor_sketch(k_norm)
    
    # Cumsum for causal and compute denominator
    k_cumsum = torch.cumsum(k_sketch, dim=2)
    
    # Dot product gives denominator
    denom = (q_sketch * k_cumsum).sum(dim=-1)
    
    return denom.flatten()


def compute_random_maclaurin_denominator(q, k, num_features=64, epsilon=1e-2):
    """
    Compute denominator using Random Maclaurin polynomial features.
    These are SIGNED and can produce negative values.
    """
    B, H, T, D = q.shape
    
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    # Random Maclaurin: r1, r2 ~ N(0, I/D)
    r1 = torch.randn(num_features, D, device=q.device, dtype=q.dtype) / math.sqrt(D)
    r2 = torch.randn(num_features, D, device=q.device, dtype=q.dtype) / math.sqrt(D)
    
    # φ(x) = (r1·x)(r2·x) - this can be negative!
    q_proj1 = torch.matmul(q_norm, r1.T)  # (B, H, T, P)
    q_proj2 = torch.matmul(q_norm, r2.T)
    q_rm = q_proj1 * q_proj2 / math.sqrt(num_features)
    
    k_proj1 = torch.matmul(k_norm, r1.T)
    k_proj2 = torch.matmul(k_norm, r2.T)
    k_rm = k_proj1 * k_proj2 / math.sqrt(num_features)
    
    # Cumsum for causal
    k_cumsum = torch.cumsum(k_rm, dim=2)
    
    # Dot product
    denom = (q_rm * k_cumsum).sum(dim=-1)
    
    return denom.flatten()


# ============================================================================
# Plotting
# ============================================================================
def plot_denominator_histograms(output_path='denominator_histogram.pdf'):
    """
    Main figure: Histogram comparison of denominator values.
    Shows SLAY anchor is always positive, while signed baselines can be negative.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate long sequence
    B, H, T, D = 2, 4, 2048, 64
    
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    
    print("  Computing denominators for each method...")
    
    with torch.no_grad():
        denom_yat = compute_yat_denominator(q, k).cpu().numpy()
        denom_sph = compute_spherical_yat_denominator(q, k).cpu().numpy()
        denom_slay = compute_slay_anchor_denominator(q, k).cpu().numpy()
        denom_ts = compute_tensor_sketch_denominator(q, k).cpu().numpy()
        denom_rm = compute_random_maclaurin_denominator(q, k).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.0))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    data = [
        ('(a) ⵟ (YAT)', denom_yat, COLORS['yat']),
        ('(b) ⵟ$_{sph}$ (Sph. YAT)', denom_sph, COLORS['spherical_yat']),
        ('(c) SLAY (Anchor)', denom_slay, COLORS['slay_anchor']),
        ('(d) TensorSketch', denom_ts, COLORS['tensor_sketch']),
        ('(e) Random Maclaurin', denom_rm, COLORS['random_maclaurin']),
    ]
    
    for idx, (title, values, color) in enumerate(data):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        # Compute stats
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        neg_frac = (values < 0).mean() * 100
        
        # Histogram
        bins = np.linspace(min(values.min(), -0.1), values.max(), 50)
        ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Mark zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=10)
        
        # Shade negative region
        if min_val < 0:
            ax.axvspan(min_val, 0, alpha=0.2, color='red', zorder=0)
        
        # Stats annotation
        stats_text = f'Min: {min_val:.3f}\nMean: {mean_val:.2f}\nNeg: {neg_frac:.1f}%'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=6,
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Denominator value')
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    
    # Hide last subplot
    axes[5].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'sequence_length': T,
        'batch_size': B,
        'num_heads': H,
        'embed_dim': D,
        'yat_denominator': denom_yat,
        'spherical_yat_denominator': denom_sph,
        'slay_anchor_denominator': denom_slay,
        'tensor_sketch_denominator': denom_ts,
        'random_maclaurin_denominator': denom_rm,
        'statistics': {
            'yat_min': float(denom_yat.min()), 'yat_negative_pct': float((denom_yat < 0).mean() * 100),
            'sph_min': float(denom_sph.min()), 'sph_negative_pct': float((denom_sph < 0).mean() * 100),
            'slay_min': float(denom_slay.min()), 'slay_negative_pct': float((denom_slay < 0).mean() * 100),
            'ts_min': float(denom_ts.min()), 'ts_negative_pct': float((denom_ts < 0).mean() * 100),
            'rm_min': float(denom_rm.min()), 'rm_negative_pct': float((denom_rm < 0).mean() * 100),
        }
    }, 
    description="Denominator positivity analysis across attention mechanisms",
    goal="Verify that SLAY maintains positive denominators (numerical stability) unlike signed polynomial baselines.",
    what_to_look_for="1) Check if any denominator values are negative (causes NaN/instability). "
                     "2) Compare minimum values across methods. "
                     "3) Look at the percentage of negative samples for each method.",
    expected_conclusion="SLAY (anchor) and spherical YAT have 0% negative denominators, ensuring stability. "
                       "Tensor Sketch and Random Maclaurin have significant negative fractions, causing training instability.")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_stability_summary(output_path='denominator_stability.pdf'):
    """
    Summary bar chart showing fraction of negative denominators for each method.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    methods = ['ⵟ', 'ⵟ$_{sph}$', 'SLAY\n(Anchor)', 'Tensor\nSketch', 'Random\nMaclaurin']
    colors = [COLORS['yat'], COLORS['spherical_yat'], COLORS['slay_anchor'], 
              COLORS['tensor_sketch'], COLORS['random_maclaurin']]
    
    # Test across multiple random seeds
    neg_fracs = [[] for _ in range(5)]
    
    print("  Running stability test across multiple seeds...")
    
    for seed in range(5):
        torch.manual_seed(seed)
        
        B, H, T, D = 1, 4, 1024, 64
        q = torch.randn(B, H, T, D, device=device)
        k = torch.randn(B, H, T, D, device=device)
        
        with torch.no_grad():
            denoms = [
                compute_yat_denominator(q, k),
                compute_spherical_yat_denominator(q, k),
                compute_slay_anchor_denominator(q, k),
                compute_tensor_sketch_denominator(q, k),
                compute_random_maclaurin_denominator(q, k),
            ]
        
        for i, d in enumerate(denoms):
            neg_fracs[i].append((d < 0).float().mean().item() * 100)
    
    # Compute mean and std
    means = [np.mean(nf) for nf in neg_fracs]
    stds = [np.std(nf) for nf in neg_fracs]
    
    # Plot
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=0.8, capsize=3)
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel('Negative denominators (%)')
    ax.set_title('Denominator stability comparison')
    ax.set_ylim(0, max(means) * 1.3 + 5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate exact values
    for bar, mean in zip(bars, means):
        if mean > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{mean:.1f}%', ha='center', va='bottom', fontsize=7)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    '0%', ha='center', va='bottom', fontsize=7, color='green')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'methods': methods,
        'negative_pct_means': np.array(means),
        'negative_pct_stds': np.array(stds),
        'num_seeds_tested': 5,
    }, 
    description="Stability summary: fraction of negative denominators by method",
    goal="Summarize denominator stability across multiple random seeds.",
    what_to_look_for="1) Which methods have zero negative denominators (green = stable). "
                     "2) Compare mean and std of negative percentages. "
                     "3) Note the stark contrast between SLAY/YAT and signed polynomial methods.",
    expected_conclusion="SLAY and exact YAT variants consistently have 0% negative denominators across all seeds, "
                       "while Tensor Sketch and Random Maclaurin show significant instability.")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print(" ICML Figure: Denominator Positivity Analysis")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/2] Generating denominator histogram...")
    path1 = plot_denominator_histograms('assets/denominator_histogram.pdf')
    print(f"  ✓ Saved: {path1}")
    
    print("\n[2/2] Generating stability summary...")
    path2 = plot_stability_summary('assets/denominator_stability.pdf')
    print(f"  ✓ Saved: {path2}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}  (detailed histograms)")
    print(f"  • {path2}  (stability bar chart)")
    print("=" * 60)


if __name__ == "__main__":
    main()
