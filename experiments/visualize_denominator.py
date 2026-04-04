"""
Denominator Positivity Histogram - ICML Paper Figure

Visualization showing histogram of denominator values across long sequences.
Demonstrates stability of LAY/SLAY (anchor features) vs signed polynomial baselines.

Outputs PDF with vector graphics for camera-ready quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from viz_utils import COLORS, DS, setup_icml_style, log_data

# Apply unified publication settings
setup_icml_style()

# Use design system constants
COLUMN_WIDTH = DS.COLUMN_WIDTH
FULL_WIDTH = DS.FULL_WIDTH

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
    Compute spherical YAT denominator: C - 2x where x = q̂·k̂, C = 2 + epsilon
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


def compute_lay_denominator(q, k, num_rff=64, num_anchors=32, epsilon=1e-2):
    """
    Compute LAY denominator using RFF for RBF and Anchor for polynomial.
    """
    B, H, T, D = q.shape
    
    # Anchor features for (q·k)² polynomial component
    anchors = torch.randn(num_anchors, D, device=q.device, dtype=q.dtype)
    anchors = F.normalize(anchors, p=2, dim=-1)
    
    q_poly = (torch.matmul(q, anchors.T) ** 2) / math.sqrt(num_anchors)
    k_poly = (torch.matmul(k, anchors.T) ** 2) / math.sqrt(num_anchors)
    
    # Gauss-Laguerre for Laplace transform
    num_nodes = 3
    nodes, weights = np.polynomial.laguerre.laggauss(num_nodes)
    s_vals = torch.tensor(nodes / epsilon, dtype=q.dtype, device=q.device)
    w_vals = torch.tensor(weights / epsilon, dtype=q.dtype, device=q.device)
    
    total_denom = torch.zeros(B, H, T, device=q.device, dtype=q.dtype)
    
    for r in range(num_nodes):
        s_r = s_vals[r]
        w_r = w_vals[r]
        
        omega = torch.randn(D, num_rff, device=q.device, dtype=q.dtype) * math.sqrt(2 * s_r)
        
        q_proj = torch.matmul(q, omega)
        k_proj = torch.matmul(k, omega)
        
        q_cos = torch.cos(q_proj) / math.sqrt(num_rff)
        q_sin = torch.sin(q_proj) / math.sqrt(num_rff)
        k_cos = torch.cos(k_proj) / math.sqrt(num_rff)
        k_sin = torch.sin(k_proj) / math.sqrt(num_rff)
        
        # We need outer product of Poly features and RFF features
        # Or more efficiently: (q_poly * q_rff) · sum(k_poly * k_rff)
        # q_rff is concatenation of [q_cos, q_sin]
        
        # Let's do components separately to avoid massive tensors
        for q_trig, k_trig in [(q_cos, k_cos), (q_sin, k_sin)]:
             q_combined = torch.einsum('bhtp,bhtm->bhtpm', q_poly, q_trig)
             k_combined = torch.einsum('bhtp,bhtm->bhtpm', k_poly, k_trig)
             
             k_cumsum = torch.cumsum(k_combined, dim=2)
             denom_part = torch.einsum('bhtpm,bhtpm->bht', q_combined, k_cumsum)
             total_denom += w_r * denom_part

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
    Shows LAY/SLAY anchor is always positive, while signed baselines can be negative.
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
        denom_lay = compute_lay_denominator(q, k).cpu().numpy()
        denom_ts = compute_tensor_sketch_denominator(q, k).cpu().numpy()
        denom_rm = compute_random_maclaurin_denominator(q, k).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.0))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    data = [
        ('(a) ⵟ (YAT)', denom_yat, COLORS['yat_exact']),
        ('(b) ⵟ$_{sph}$ (Sph. YAT)', denom_sph, COLORS['yat_spherical']),
        ('(c) LAY (Anchor)', denom_lay, COLORS['lay']),
        ('(d) SLAY (Anchor)', denom_slay, COLORS['slay']),
        ('(e) TensorSketch', denom_ts, COLORS['tensor_sketch']),
        ('(f) Random Maclaurin', denom_rm, COLORS['random_maclaurin']),
    ]
    
    for idx, (title, values, color) in enumerate(data):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        # Compute stats
        min_val = values.min()
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

        # --- Save individual plot ---
        safe_name = title.split(') ')[-1].replace(' ', '_').replace('$', '').replace('{', '').replace('}', '').lower()
        safe_name = safe_name.replace('ⵟ', 'yat').replace('anchor', '').replace('(', '').replace(')', '')
        safe_name = "".join([c for c in safe_name if c.isalnum() or c=='_']).strip('_')
        
        indiv_path = output_path.replace('.pdf', f'_{safe_name}.pdf')
        
        fig_indiv, ax_indiv = plt.subplots(figsize=(4, 3))
        fig_indiv.patch.set_facecolor('white')
        
        ax_indiv.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_indiv.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=10)
        if min_val < 0:
            ax_indiv.axvspan(min_val, 0, alpha=0.2, color='red', zorder=0)
            
        ax_indiv.text(0.95, 0.95, stats_text, transform=ax_indiv.transAxes, fontsize=8,
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_indiv.set_xlabel('Denominator value')
        ax_indiv.set_ylabel('Count')
        ax_indiv.set_title(title.split(') ')[-1], fontsize=10)
        ax_indiv.grid(True, alpha=0.3)
        
        fig_indiv.tight_layout()
        fig_indiv.savefig(indiv_path, format='pdf', bbox_inches='tight')
        plt.close(fig_indiv)
        print(f"  [OK] Saved individual: {indiv_path}")
    
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'yat_denom_min': float(denom_yat.min()),
        'lay_denom_min': float(denom_lay.min()),
        'slay_denom_min': float(denom_slay.min()),
        'ts_denom_min': float(denom_ts.min()),
    }, description="Denominator statistics")
    
    return output_path


def plot_stability_summary(output_path='denominator_stability.pdf'):
    """
    Summary bar chart showing fraction of negative denominators for each method.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    methods = ['ⵟ', 'ⵟ$_{sph}$', 'LAY', 'SLAY', 'Tensor\nSketch', 'Random\nMaclaurin']
    colors = [COLORS['yat_exact'], COLORS['yat_spherical'], COLORS['lay'], COLORS['slay'], 
              COLORS['tensor_sketch'], COLORS['random_maclaurin']]
    
    # Test across multiple random seeds
    neg_fracs = [[] for _ in range(6)]
    
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
                compute_lay_denominator(q, k),
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
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {'means': means}, description="Stability summary")
    
    return output_path


def main():
    print("=" * 60)
    print(" ICML Figure: Denominator Positivity Analysis")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/2] Generating denominator histogram...")
    path1 = plot_denominator_histograms('assets/denominator_histogram.pdf')
    print(f"  [OK] Saved: {path1}")
    
    print("\n[2/2] Generating stability summary...")
    path2 = plot_stability_summary('assets/denominator_stability.pdf')
    print(f"  [OK] Saved: {path2}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}")
    print(f"  • {path2}")
    print("=" * 60)


if __name__ == "__main__":
    main()
