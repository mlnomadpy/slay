"""
Spherical Heatmap Visualization - ICML Paper Figure

2D visualization of attention weights on the sphere S² for a fixed query.
Shows angular locality of YAT/spherical YAT vs softmax's global mass.

Outputs PDF with vector graphics for camera-ready quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from viz_utils import log_data

# ============================================================================
# Publication Settings (ICML)
# ============================================================================
FULL_WIDTH = 6.75

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

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# Kernel Functions (return attention weights over S²)
# ============================================================================
def softmax_kernel(query, keys, temperature=1.0):
    """
    Softmax attention: exp(q·k / T) / sum(exp(q·k / T))
    """
    dots = torch.matmul(keys, query)
    scores = torch.exp(dots / temperature)
    return scores / scores.sum()


def yat_kernel(query, keys, epsilon=1e-2):
    """
    YAT kernel (unnormalized):
    k(q, k) = (q·k)² / (||q-k||² + ε)
    """
    dots = torch.matmul(keys, query)
    diffs = keys - query.unsqueeze(0)
    dist_sq = (diffs ** 2).sum(dim=-1)
    scores = (dots ** 2) / (dist_sq + epsilon)
    return scores / scores.sum()


def spherical_yat_kernel(query, keys, epsilon=1e-2):
    """
    Spherical YAT kernel:
    k(x) = x² / (C - 2x) where x = q̂·k̂ and C = 2 + ε
    """
    q_norm = F.normalize(query.unsqueeze(0), p=2, dim=-1).squeeze(0)
    k_norm = F.normalize(keys, p=2, dim=-1)
    
    dots = torch.matmul(k_norm, q_norm)
    C = 2.0 + epsilon
    scores = (dots ** 2) / (C - 2 * dots)
    scores = torch.clamp(scores, min=0)  # Ensure non-negative
    return scores / scores.sum()


def slay_anchor_kernel(query, keys, num_anchors=32, num_prf=16, num_nodes=3, epsilon=1e-2):
    """
    SLAY (anchor) approximation of spherical YAT.
    Uses anchor features for polynomial and PRF for exponential.
    """
    # Normalize
    q_norm = F.normalize(query.unsqueeze(0), p=2, dim=-1).squeeze(0)
    k_norm = F.normalize(keys, p=2, dim=-1)
    
    d = query.shape[-1]
    
    # Anchor features for polynomial
    anchors = torch.randn(num_anchors, d)
    anchors = F.normalize(anchors, p=2, dim=-1)
    
    q_poly = (torch.matmul(q_norm.unsqueeze(0), anchors.T) ** 2).squeeze(0) / np.sqrt(num_anchors)
    k_poly = (torch.matmul(k_norm, anchors.T) ** 2) / np.sqrt(num_anchors)
    
    # Gauss-Laguerre quadrature
    C = 2.0 + epsilon
    nodes, weights = np.polynomial.laguerre.laggauss(num_nodes)
    nodes = torch.tensor(nodes, dtype=torch.float32) / C
    weights = torch.tensor(weights, dtype=torch.float32) / C
    
    # PRF features (simplified - using random projections)
    total_score = torch.zeros(keys.shape[0])
    
    for r in range(num_nodes):
        s_r = nodes[r].item()
        w_r = weights[r].item()
        
        omega = torch.randn(d, num_prf)
        sqrt_2s = np.sqrt(max(2.0 * s_r, 0))
        
        q_proj = torch.matmul(q_norm.unsqueeze(0), omega).squeeze(0) * sqrt_2s - s_r
        k_proj = torch.matmul(k_norm, omega) * sqrt_2s - s_r
        
        q_prf = torch.exp(torch.clamp(q_proj, -20, 20)) / np.sqrt(num_prf)
        k_prf = torch.exp(torch.clamp(k_proj, -20, 20)) / np.sqrt(num_prf)
        
        # Tensor product (simplified: element-wise for visualization)
        poly_score = torch.matmul(k_poly, q_poly)
        prf_score = torch.matmul(k_prf, q_prf)
        
        total_score += w_r * poly_score * prf_score
    
    total_score = torch.clamp(total_score, min=0)
    return total_score / total_score.sum()


# ============================================================================
# Generate Points on S²
# ============================================================================
def fibonacci_sphere(n_points=1000):
    """Generate quasi-uniformly distributed points on S² using Fibonacci spiral."""
    indices = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.stack([x, y, z], axis=-1)


def grid_sphere(n_theta=50, n_phi=100):
    """Generate grid of points on S² for heatmap."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    return x, y, z, theta_grid, phi_grid


# ============================================================================
# Plotting
# ============================================================================
def plot_spherical_heatmaps(output_path='spherical_heatmap.pdf'):
    """Create 2x2 comparison of attention patterns on S²."""
    
    fig = plt.figure(figsize=(FULL_WIDTH, 4.5))
    fig.patch.set_facecolor('white')
    
    # Fixed query at the "north pole"
    query = torch.tensor([0.0, 0.0, 1.0])
    
    # Generate sphere grid
    n_theta, n_phi = 40, 80
    x, y, z, theta_grid, phi_grid = grid_sphere(n_theta, n_phi)
    keys = torch.tensor(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1), dtype=torch.float32)
    
    kernels = [
        ('(a) Softmax', lambda q, k: softmax_kernel(q, k, temperature=0.5)),
        ('(b) ⵟ (YAT)', lambda q, k: yat_kernel(q, k)),
        ('(c) ⵟ$_{sph}$ (Sph. YAT)', lambda q, k: spherical_yat_kernel(q, k)),
        ('(d) SLAY (Anchor)', lambda q, k: slay_anchor_kernel(q, k)),
    ]
    
    for idx, (title, kernel_fn) in enumerate(kernels):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.set_facecolor('white')
        
        # Compute attention weights
        with torch.no_grad():
            weights = kernel_fn(query, keys).numpy()
        
        # Reshape to grid
        weights_grid = weights.reshape(n_phi, n_theta)
        
        # Normalize for visualization (log scale for better contrast)
        weights_grid = np.log10(weights_grid + 1e-10)
        weights_grid = (weights_grid - weights_grid.min()) / (weights_grid.max() - weights_grid.min() + 1e-10)
        
        # Plot sphere with heatmap
        surf = ax.plot_surface(x, y, z, facecolors=cm.viridis(weights_grid),
                               linewidth=0, antialiased=True, shade=False)
        
        # Mark query point
        ax.scatter([0], [0], [1], color='red', s=100, marker='*', 
                   edgecolors='white', linewidths=0.5, zorder=10)
        
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_box_aspect([1, 1, 1])
        
        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cm.viridis)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention weight (log scale)', fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Med', 'High'])
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'grid_resolution': {'n_theta': n_theta, 'n_phi': n_phi},
        'query_position': 'north pole [0, 0, 1]',
        'kernels_compared': ['Softmax', 'YAT', 'Spherical YAT', 'SLAY (Anchor)'],
    }, 
    description="3D spherical heatmap of attention weights on S^2",
    goal="Visualize how different attention mechanisms weight keys distributed on a unit sphere relative to a fixed query.",
    what_to_look_for="1) How attention weight changes with angular distance from query. "
                     "2) Compare spread/concentration of attention across mechanisms. "
                     "3) Note symmetry and smoothness of the attention distribution.",
    expected_conclusion="Spherical YAT and SLAY show more concentrated attention around the query direction, "
                       "while softmax distributes weight more uniformly. YAT kernels respect geometric structure.")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_polar_projection(output_path='spherical_polar.pdf'):
    """
    2D polar projection showing attention weights.
    Easier to read than 3D for paper.
    """
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.0), 
                              subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    
    # Fixed query at angle 0
    query = torch.tensor([1.0, 0.0, 0.0])
    
    # Generate circle of keys at different angles
    n_angles = 100
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    
    # Keys on 2D unit circle embedded in 3D
    keys = torch.tensor(np.stack([np.cos(angles), np.sin(angles), np.zeros(n_angles)], axis=-1), 
                        dtype=torch.float32)
    
    kernels = [
        ('Softmax', lambda q, k: softmax_kernel(q, k, temperature=0.5)),
        ('ⵟ', lambda q, k: yat_kernel(q, k)),
        ('ⵟ$_{sph}$', lambda q, k: spherical_yat_kernel(q, k)),
        ('SLAY', lambda q, k: slay_anchor_kernel(q, k)),
    ]
    
    colors = ['#0077BB', '#EE7733', '#CC3311', '#EE3377']
    
    for idx, (ax, (title, kernel_fn), color) in enumerate(zip(axes, kernels, colors)):
        ax.set_facecolor('white')
        
        with torch.no_grad():
            weights = kernel_fn(query, keys).numpy()
        
        # Scale for visualization (normalize max to 1)
        weights = weights / weights.max()
        
        # Plot as filled area
        ax.fill(np.append(angles, angles[0]), np.append(weights, weights[0]), 
                color=color, alpha=0.5)
        ax.plot(np.append(angles, angles[0]), np.append(weights, weights[0]), 
                color=color, linewidth=1.5)
        
        # Mark query direction
        ax.scatter([0], [1.1], color='red', s=40, marker='*', zorder=10)
        
        ax.set_title(title, fontsize=9, pad=5)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['0.5', '1.0'], fontsize=6)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'num_angles': n_angles,
        'angles_radians': angles,
        'kernels_compared': ['Softmax', 'YAT', 'Spherical YAT', 'SLAY'],
    }, 
    description="2D polar projection of attention weights",
    goal="Show attention weight as a function of angle in a clear 2D polar format.",
    what_to_look_for="1) How quickly attention drops off with angular distance. "
                     "2) Compare the 'sharpness' of each kernel's attention profile. "
                     "3) Verify SLAY approximates spherical YAT profile closely.",
    expected_conclusion="Spherical YAT produces sharper, more localized attention than softmax. "
                       "SLAY closely matches the spherical YAT profile, validating the approximation.")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print(" ICML Figure: Spherical Heatmap Visualization")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/2] Generating 3D spherical heatmaps...")
    path1 = plot_spherical_heatmaps('assets/spherical_heatmap.pdf')
    print(f"  ✓ Saved: {path1}")
    
    print("\n[2/2] Generating 2D polar projection...")
    path2 = plot_polar_projection('assets/spherical_polar.pdf')
    print(f"  ✓ Saved: {path2}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}  (3D view)")
    print(f"  • {path2}  (2D polar, recommended for paper)")
    print("=" * 60)


if __name__ == "__main__":
    main()
