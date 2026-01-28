"""
Neuron Decision Boundary Visualization - ICML Paper Figure

Publication-quality figure showing decision boundaries of various kernel-based classifiers.
Outputs PDF with vector graphics for camera-ready quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from matplotlib.colors import ListedColormap

# ============================================================================
# Publication Settings
# ============================================================================
# ICML column width is about 3.25 inches, full width is 6.75 inches
COLUMN_WIDTH = 6.75  # Full width figure
ASPECT_RATIO = 0.5   # Height = width * aspect_ratio

# Use Type 1 fonts for PDF (required by many venues)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 0.8

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# Data Generation
# ============================================================================
def generate_data(num_neurons=5, grid_res=300, bounds=(-2.5, 2.5)):
    """Generate random neurons and a dense grid of points."""
    neurons = torch.rand(num_neurons, 2) * 4.0 - 2.0  # Uniform in [-2, 2]
    
    x = np.linspace(bounds[0], bounds[1], grid_res)
    y = np.linspace(bounds[0], bounds[1], grid_res)
    xx, yy = np.meshgrid(x, y)
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    return neurons, grid_points, xx, yy


# ============================================================================
# Kernel Functions
# ============================================================================
def linear_softmax_scores(neurons, grid):
    """Linear kernel: K(x, w) = x · w"""
    return torch.matmul(grid, neurons.T)


def linear_elu_scores(neurons, grid):
    """Linear approximation: φ(x) = ELU(x) + 1"""
    phi_grid = F.elu(grid) + 1.0
    phi_neurons = F.elu(neurons) + 1.0
    return torch.matmul(phi_grid, phi_neurons.T)


class PerformerFAVORPlus:
    """FAVOR+ from Performer: ReLU random features."""
    def __init__(self, input_dim=2, num_features=64, seed=42):
        self.num_features = num_features
        torch.manual_seed(seed)
        self.omega = torch.randn(input_dim, num_features) / math.sqrt(input_dim)
    
    def compute_features(self, x):
        proj = x @ self.omega
        return torch.relu(proj) / math.sqrt(self.num_features)


def performer_favor_scores(neurons, grid):
    """Performer FAVOR+ approximation."""
    favor = PerformerFAVORPlus(input_dim=2, num_features=128)
    phi_grid = favor.compute_features(grid)
    phi_neurons = favor.compute_features(neurons)
    return torch.matmul(phi_grid, phi_neurons.T)


def yat_exact_scores(neurons, grid, epsilon=1e-6):
    """Yat kernel: K(x, w) = (x·w)² / (||x-w||² + ε)"""
    dot = torch.matmul(grid, neurons.T)
    grid_norm_sq = (grid ** 2).sum(dim=1, keepdim=True)
    neurons_norm_sq = (neurons ** 2).sum(dim=1, keepdim=True).T
    numerator = dot ** 2
    denominator = grid_norm_sq + neurons_norm_sq - 2 * dot + epsilon
    return numerator / denominator


def spherical_yat_exact_scores(neurons, grid, epsilon=1e-2):
    """Spherical Yat: K(x̂, ŵ) = (x̂·ŵ)² / (2 - 2x̂·ŵ + ε)"""
    grid_norm = F.normalize(grid, p=2, dim=-1)
    neurons_norm = F.normalize(neurons, p=2, dim=-1)
    dot = torch.matmul(grid_norm, neurons_norm.T)
    numerator = dot ** 2
    denominator = torch.clamp(2.0 - 2.0 * dot + epsilon, min=1e-6)
    return numerator / denominator


class YatAnchorFeatures:
    """Anchor kernel features for spherical Yat approximation (Section 2.3.2)."""
    def __init__(self, input_dim=2, num_anchors=32, num_prf_features=16, 
                 num_quadrature_nodes=3, epsilon=1e-6, seed=42):
        self.num_anchors = num_anchors
        self.num_prf_features = num_prf_features
        self.num_quadrature_nodes = num_quadrature_nodes
        self.C = 2.0 + epsilon
        
        torch.manual_seed(seed)
        anchors = torch.randn(num_anchors, input_dim)
        self.anchors = F.normalize(anchors, p=2, dim=-1)
        
        nodes, weights = np.polynomial.laguerre.laggauss(num_quadrature_nodes)
        self.nodes = torch.tensor(nodes, dtype=torch.float32) / self.C
        self.weights = torch.tensor(weights, dtype=torch.float32) / self.C
        self.omega = torch.randn(num_quadrature_nodes, input_dim, num_prf_features)
    
    def _poly_features(self, x_norm):
        return (torch.matmul(x_norm, self.anchors.T) ** 2) / math.sqrt(self.num_anchors)
    
    def _prf_features(self, x_norm):
        R, M = self.num_quadrature_nodes, self.num_prf_features
        proj = torch.einsum('nd,rdm->nrm', x_norm, self.omega)
        sqrt_2s = torch.sqrt(2.0 * self.nodes.clamp(min=0)).view(1, R, 1)
        s_vals = self.nodes.view(1, R, 1)
        exp_arg = torch.clamp(proj * sqrt_2s - s_vals, min=-20.0, max=20.0)
        prf_feat = torch.exp(exp_arg) / math.sqrt(M)
        sq_weights = torch.sqrt(self.weights.clamp(min=0)).view(1, R, 1)
        return prf_feat * sq_weights
    
    def compute_features(self, x):
        x_norm = F.normalize(x, p=2, dim=-1)
        poly = self._poly_features(x_norm)
        prf = self._prf_features(x_norm)
        N, P = poly.shape
        R, M = prf.shape[1], prf.shape[2]
        poly_exp = poly.unsqueeze(2).unsqueeze(3)
        prf_exp = prf.unsqueeze(1)
        fused = poly_exp * prf_exp
        return fused.reshape(N, -1)


def spherical_yat_anchor_scores(neurons, grid):
    """Spherical Yat with Anchor kernel features."""
    anchor_feat = YatAnchorFeatures(input_dim=2, num_anchors=32, 
                                     num_prf_features=16, num_quadrature_nodes=3)
    phi_grid = anchor_feat.compute_features(grid)
    phi_neurons = anchor_feat.compute_features(neurons)
    return torch.matmul(phi_grid, phi_neurons.T)


# ============================================================================
# Publication-Quality Visualization
# ============================================================================
def create_colormap(num_classes=5):
    """High-contrast colormap suitable for colorblind readers."""
    # Tableau colorblind-safe palette
    colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Grey
    ]
    return ListedColormap(colors[:num_classes])


def plot_decision_boundaries_publication(neurons, grid_points, xx, yy, methods, output_path):
    """Create publication-quality figure for ICML."""
    
    num_methods = len(methods)
    num_neurons = len(neurons)
    cmap = create_colormap(num_neurons)
    
    # ICML: 2 rows x 3 cols, full page width
    fig_width = COLUMN_WIDTH
    fig_height = fig_width * 0.55  # Slightly taller for labels
    
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    # Panel labels for referencing in paper
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for idx, (name, score_fn) in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        # Compute scores and predictions
        with torch.no_grad():
            scores = score_fn(neurons, grid_points)
            predictions = torch.argmax(scores, dim=1).numpy().reshape(xx.shape)
        
        # Plot decision regions - softer alpha for cleaner look
        ax.contourf(xx, yy, predictions, levels=np.arange(-0.5, num_neurons + 0.5, 1),
                   cmap=cmap, alpha=0.4)
        
        # Decision boundaries - crisp black lines
        ax.contour(xx, yy, predictions, levels=np.arange(0.5, num_neurons, 1),
                  colors='black', linewidths=0.6, alpha=0.8)
        
        # Plot neurons as filled stars
        for i in range(num_neurons):
            ax.scatter(neurons[i, 0], neurons[i, 1], 
                      c=[cmap.colors[i]], s=120, marker='*', 
                      edgecolors='black', linewidths=0.8, zorder=10)
        
        # Panel label in corner
        ax.text(0.05, 0.95, panel_labels[idx], transform=ax.transAxes,
               fontsize=9, fontweight='bold', va='top', ha='left')
        
        # Subplot title (method name)
        ax.set_title(name, fontsize=8, pad=3)
        
        # Clean axis styling
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(direction='in', length=2, width=0.5)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    
    # Hide unused subplots
    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
    
    # Save as PDF (vector graphics)
    plt.savefig(output_path, format='pdf', dpi=300, 
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    return output_path


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print(" ICML Figure: Neural Decision Boundaries")
    print("=" * 60)
    
    print("\n[1/3] Generating neurons and grid...")
    neurons, grid_points, xx, yy = generate_data(num_neurons=5, grid_res=500)
    print(f"  • {len(neurons)} neurons (uniformly distributed)")
    print(f"  • {xx.size:,} grid points for smooth contours")
    
    print("\n[2/3] Kernels to visualize:")
    methods = [
        ("Linear", linear_softmax_scores),
        ("FAVOR+", performer_favor_scores),
        ("Linear (ELU+1)", linear_elu_scores),
        ("ⵟ", yat_exact_scores),
        ("ⵟ_sph", spherical_yat_exact_scores),
        ("SLAY (Anchor)", spherical_yat_anchor_scores),
    ]
    for name, _ in methods:
        print(f"  • {name}")
    
    print("\n[3/3] Generating publication PDF...")
    output_path = 'figures/decision_boundaries.pdf'
    
    # Create figures directory if needed
    import os
    os.makedirs('figures', exist_ok=True)
    
    plot_decision_boundaries_publication(neurons, grid_points, xx, yy, methods, output_path)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Format: PDF (vector graphics)")
    print(f"  Size: {COLUMN_WIDTH:.2f}\" width (ICML full-width)")
    print("=" * 60)


if __name__ == "__main__":
    main()
