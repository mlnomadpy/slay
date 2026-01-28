"""
Attention Entropy Visualization - ICML Paper Figure

Shows attention entropy/sparsity as a function of token distance and cosine similarity.
Demonstrates that SLAY induces structured selectivity similar to YAT/spherical YAT.

Outputs PDF with vector graphics for camera-ready quality.
"""

import torch
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
    'softmax': '#0077BB',       # Blue
    'yat': '#EE7733',           # Orange
    'spherical_yat': '#CC3311', # Red
    'slay': '#EE3377',          # Magenta
    'performer': '#009988',     # Teal
    'linear': '#33BBEE',        # Cyan
}

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# Attention Computation Functions
# ============================================================================
def compute_softmax_attention(q, k, v, scale=None):
    """Standard softmax attention with causal mask."""
    B, H, T, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float('-inf'))
    
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    
    return attn, out


def compute_yat_attention(q, k, v, epsilon=1e-2):
    """YAT attention with causal mask."""
    B, H, T, D = q.shape
    
    dots = torch.matmul(q, k.transpose(-1, -2))
    q_sq = (q ** 2).sum(dim=-1, keepdim=True)
    k_sq = (k ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)
    dist_sq = q_sq + k_sq - 2 * dots
    
    kernel = (dots ** 2) / (dist_sq + epsilon)
    
    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    kernel = kernel.masked_fill(mask, 0.0)
    
    # Normalize
    attn = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)
    out = torch.matmul(attn, v)
    
    return attn, out


def compute_spherical_yat_attention(q, k, v, epsilon=1e-2):
    """Spherical YAT attention with causal mask."""
    B, H, T, D = q.shape
    
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    dots = torch.matmul(q_norm, k_norm.transpose(-1, -2))
    
    C = 2.0 + epsilon
    kernel = (dots ** 2) / (C - 2 * dots)
    kernel = torch.clamp(kernel, min=0)
    
    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    kernel = kernel.masked_fill(mask, 0.0)
    
    # Normalize
    attn = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)
    out = torch.matmul(attn, v)
    
    return attn, out


def compute_slay_attention(q, k, v, num_anchors=32, epsilon=1e-2):
    """SLAY (anchor) attention with causal mask."""
    B, H, T, D = q.shape
    
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    
    # Anchor features
    anchors = torch.randn(num_anchors, D, device=q.device, dtype=q.dtype)
    anchors = F.normalize(anchors, p=2, dim=-1)
    
    q_feat = (torch.matmul(q_norm, anchors.T) ** 2) / math.sqrt(num_anchors)
    k_feat = (torch.matmul(k_norm, anchors.T) ** 2) / math.sqrt(num_anchors)
    
    kernel = torch.matmul(q_feat, k_feat.transpose(-1, -2))
    
    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    kernel = kernel.masked_fill(mask, 0.0)
    
    # Normalize
    attn = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)
    out = torch.matmul(attn, v)
    
    return attn, out


def compute_performer_attention(q, k, v, num_features=64):
    """Performer (FAVOR+) attention with ReLU features."""
    B, H, T, D = q.shape
    
    # Random projections
    omega = torch.randn(D, num_features, device=q.device, dtype=q.dtype) / math.sqrt(D)
    
    # ReLU random features
    q_proj = torch.matmul(q, omega)
    k_proj = torch.matmul(k, omega)
    
    q_feat = F.relu(q_proj) / math.sqrt(num_features)
    k_feat = F.relu(k_proj) / math.sqrt(num_features)
    
    # Linear attention with causal cumsum
    k_cumsum = torch.cumsum(k_feat, dim=2)
    kv_cumsum = torch.cumsum(torch.einsum('bhtm,bhtd->bhtmd', k_feat, v), dim=2)
    
    denom = (q_feat.unsqueeze(-1) * k_cumsum.unsqueeze(-1)).sum(dim=-2) + 1e-6
    out = torch.einsum('bhtm,bhtmd->bhtd', q_feat, kv_cumsum) / denom
    
    # Approximate attention matrix for visualization (not actually computed)
    kernel = torch.matmul(q_feat, k_feat.transpose(-1, -2))
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    kernel = kernel.masked_fill(mask, 0.0)
    attn = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)
    
    return attn, out


# ============================================================================
# Entropy Computation
# ============================================================================
def compute_attention_entropy(attn):
    """
    Compute normalized entropy of attention distribution.
    
    H(p) = -sum(p * log(p))
    Normalized by log(T) to get value in [0, 1]
    """
    # Add small epsilon to avoid log(0)
    attn_safe = attn + 1e-10
    
    # Compute entropy
    entropy = -torch.sum(attn_safe * torch.log(attn_safe), dim=-1)
    
    # Normalize by max possible entropy (uniform distribution)
    T = attn.shape[-1]
    max_entropy = torch.log(torch.tensor(T, dtype=attn.dtype, device=attn.device))
    
    return entropy / max_entropy


def compute_attention_sparsity(attn, threshold=0.01):
    """
    Compute sparsity as fraction of attention weights below threshold.
    """
    return (attn < threshold).float().mean(dim=-1)


# ============================================================================
# Plotting
# ============================================================================
def plot_entropy_vs_position(output_path='attention_entropy.pdf'):
    """
    Plot attention entropy as a function of query position.
    Shows how attention becomes more focused at different positions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, H, T, D = 2, 4, 256, 64
    
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    
    methods = [
        ('Softmax', compute_softmax_attention),
        ('ⵟ (YAT)', compute_yat_attention),
        ('ⵟ$_{sph}$', compute_spherical_yat_attention),
        ('SLAY', compute_slay_attention),
        ('FAVOR+', compute_performer_attention),
    ]
    
    colors_list = [COLORS['softmax'], COLORS['yat'], COLORS['spherical_yat'], 
                   COLORS['slay'], COLORS['performer']]
    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [2.0, 1.8, 1.6, 2.5, 1.4]  # Vary widths, SLAY thicker
    markers = ['', '', '', 'o', '']  # Add markers to SLAY
    
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    fig.patch.set_facecolor('white')
    
    # ---- Panel (a): Entropy vs position ----
    ax = axes[0]
    ax.set_facecolor('white')
    
    positions = np.arange(T)
    
    for i, ((name, fn), color) in enumerate(zip(methods, colors_list)):
        with torch.no_grad():
            attn, _ = fn(q.float(), k.float(), v.float())
            entropy = compute_attention_entropy(attn)
            
            # Average over batch and heads
            entropy_mean = entropy.mean(dim=(0, 1)).cpu().numpy()
        
        # Smooth with rolling average
        window = 10
        entropy_smooth = np.convolve(entropy_mean, np.ones(window)/window, mode='valid')
        
        # Use different styles for each line
        marker = markers[i]
        markevery = 25 if marker else None
        ax.plot(positions[window-1:], entropy_smooth, label=name, color=color, 
                linewidth=linewidths[i], linestyle=linestyles[i],
                marker=marker, markevery=markevery, markersize=4)
    
    ax.set_xlabel('Query position')
    ax.set_ylabel('Normalized entropy')
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=7)
    ax.set_title('(a) Attention entropy vs position')
    ax.grid(True, alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    # ---- Panel (b): Entropy distribution ----
    ax = axes[1]
    ax.set_facecolor('white')
    
    for i, ((name, fn), color) in enumerate(zip(methods, colors_list)):
        with torch.no_grad():
            attn, _ = fn(q.float(), k.float(), v.float())
            entropy = compute_attention_entropy(attn)
            
            # Flatten all entropy values
            entropy_flat = entropy[:, :, T//4:].flatten().cpu().numpy()  # Skip early positions
        
        # Use step histogram for better visibility
        ax.hist(entropy_flat, bins=30, alpha=0.3 if i < 2 else 0.7, 
                label=name, color=color, density=True, 
                histtype='step', linewidth=linewidths[i])
    
    ax.set_xlabel('Normalized entropy')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=7)
    ax.set_title('(b) Entropy distribution')
    ax.grid(True, alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'sequence_length': T, 'batch_size': B, 'num_heads': H, 'embed_dim': D,
        'positions': positions,
        'methods': [name for name, _ in methods],
    }, 
    description="Attention entropy vs position for different attention mechanisms",
    goal="Analyze how attention entropy (spread) changes as query position increases in causal attention.",
    what_to_look_for="1) Early positions have low entropy (few tokens to attend to). "
                     "2) Compare steady-state entropy across mechanisms. "
                     "3) Note if any mechanism maintains lower entropy (more focused attention).",
    expected_conclusion="YAT and spherical YAT maintain lower entropy than softmax at longer positions, "
                       "indicating more focused/selective attention. SLAY approximates this behavior.")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_entropy_vs_similarity(output_path='entropy_vs_similarity.pdf'):
    """
    Plot attention entropy vs semantic similarity of tokens.
    Shows how different attention mechanisms respond to token similarity.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sequences with varying similarity
    B, H, T, D = 4, 2, 128, 32
    
    similarity_levels = np.linspace(0.0, 0.9, 10)
    
    methods = [
        ('Softmax', compute_softmax_attention),
        ('ⵟ (YAT)', compute_yat_attention),
        ('ⵟ$_{sph}$', compute_spherical_yat_attention),
        ('SLAY', compute_slay_attention),
    ]
    
    colors_list = [COLORS['softmax'], COLORS['yat'], COLORS['spherical_yat'], COLORS['slay']]
    
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    results = {name: [] for name, _ in methods}
    
    for sim in similarity_levels:
        # Create tokens with controlled similarity
        # Base token
        base = torch.randn(B, H, 1, D, device=device)
        base = F.normalize(base, p=2, dim=-1)
        
        # Similar tokens: mix of base and noise
        noise = torch.randn(B, H, T, D, device=device)
        noise = F.normalize(noise, p=2, dim=-1)
        
        tokens = sim * base + (1 - sim) * noise
        tokens = F.normalize(tokens, p=2, dim=-1) * np.sqrt(D)
        
        q, k, v = tokens, tokens, tokens
        
        for name, fn in methods:
            with torch.no_grad():
                attn, _ = fn(q.float(), k.float(), v.float())
                entropy = compute_attention_entropy(attn)
                mean_entropy = entropy[:, :, T//2:].mean().item()
            
            results[name].append(mean_entropy)
    
    for (name, _), color in zip(methods, colors_list):
        ax.plot(similarity_levels, results[name], 'o-', label=name, 
                color=color, markersize=5, linewidth=1.5)
    
    ax.set_xlabel('Token similarity (cosine)')
    ax.set_ylabel('Mean attention entropy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='best', framealpha=0.95, fontsize=7)
    ax.set_title('Entropy vs token similarity')
    ax.grid(True, alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'similarity_levels': similarity_levels,
        'softmax_entropy': np.array(results['Softmax']),
        'yat_entropy': np.array(results['ⵟ (YAT)']),
        'spherical_yat_entropy': np.array(results['ⵟ$_{sph}$']),
        'slay_entropy': np.array(results['SLAY']),
    }, 
    description="Attention entropy vs token cosine similarity",
    goal="Examine how attention entropy responds to varying degrees of token similarity.",
    what_to_look_for="1) At low similarity (random tokens), compare baseline entropy. "
                     "2) How does entropy change as tokens become more similar? "
                     "3) Which mechanism discriminates similar tokens best?",
    expected_conclusion="When tokens are highly similar (hard to distinguish), YAT kernels maintain lower entropy "
                       "(better discrimination) than softmax. This shows YAT's geometric selectivity.")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_attention_patterns(output_path='attention_patterns.pdf'):
    """
    Visualize attention patterns for different mechanisms.
    Shows how YAT/SLAY produce more focused attention than softmax.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, H, T, D = 1, 1, 64, 32
    
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    
    methods = [
        ('Softmax', compute_softmax_attention),
        ('ⵟ (YAT)', compute_yat_attention),
        ('ⵟ$_{sph}$ (Sph. YAT)', compute_spherical_yat_attention),
        ('SLAY', compute_slay_attention),
    ]
    
    
    # Use wider figure to accommodate colorbar
    fig = plt.figure(figsize=(FULL_WIDTH + 0.5, 2.0))
    fig.patch.set_facecolor('white')
    
    # Create GridSpec: 4 columns for plots, 1 narrow for colorbar
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.3)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])  # Colorbar axis
    
    for idx, (name, fn) in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        with torch.no_grad():
            attn, _ = fn(q.float(), k.float(), v.float())
            attn_matrix = attn[0, 0].cpu().numpy()
        
        # Plot heatmap
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto', 
                       vmin=0, vmax=attn_matrix.max())
        
        ax.set_title(name, fontsize=8)
        ax.set_xlabel('Key pos', fontsize=7)
        if idx == 0:
            ax.set_ylabel('Query pos', fontsize=7)
        
        ax.set_xticks([0, T//2, T-1])
        ax.set_yticks([0, T//2, T-1])
        ax.tick_params(labelsize=6)
    
    # Add colorbar to dedicated axis
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Attention weight', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'sequence_length': T, 'embed_dim': D,
        'methods': [name for name, _ in methods],
    }, 
    description="Attention pattern heatmaps for different mechanisms",
    goal="Visualize the structure of attention matrices for different mechanisms.",
    what_to_look_for="1) Compare sparsity/concentration of attention weights. "
                     "2) Look for diagonal bias (local attention) or uniform spread. "
                     "3) Compare YAT/SLAY patterns against softmax baseline.",
    expected_conclusion="YAT and SLAY show more structured, concentrated attention patterns compared to softmax, "
                       "indicating selective attention to relevant tokens rather than uniform spreading.")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print(" ICML Figure: Attention Entropy Analysis")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/3] Generating entropy vs position plot...")
    path1 = plot_entropy_vs_position('assets/attention_entropy.pdf')
    print(f"  ✓ Saved: {path1}")
    
    print("\n[2/3] Generating entropy vs similarity plot...")
    path2 = plot_entropy_vs_similarity('assets/entropy_vs_similarity.pdf')
    print(f"  ✓ Saved: {path2}")
    
    print("\n[3/3] Generating attention pattern heatmaps...")
    path3 = plot_attention_patterns('assets/attention_patterns.pdf')
    print(f"  ✓ Saved: {path3}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}  (main figure)")
    print(f"  • {path2}  (similarity analysis)")
    print(f"  • {path3}  (attention heatmaps)")
    print("=" * 60)


if __name__ == "__main__":
    main()
