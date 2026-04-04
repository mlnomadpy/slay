"""
Spherical Heatmap Visualization - ICML Paper Figure

Publication-quality figure showing attention concentration on the sphere.
Demonstrates YAT's angular locality vs Softmax's global diffusion.

Outputs PDF with vector graphics for camera-ready quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import matplotlib as mpl
from viz_utils import (
    COLORS, DS, setup_icml_style, log_data,
    DEFAULT_KERNELS, get_default_kernels, get_kernel
)

# Apply unified publication settings
setup_icml_style()

# Use design system constants
COLUMN_WIDTH = DS.COLUMN_WIDTH
FULL_WIDTH = DS.FULL_WIDTH

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# Plotting
# ============================================================================
def plot_spherical_heatmaps(output_path='spherical_heatmap.pdf', kernels=None):
    """
    Plot 3D spherical heatmaps for different kernels.
    
    Args:
        output_path: Path to save the PDF
        kernels: List of kernel configs to visualize
    """
    # Use default kernels if not specified
    if kernels is None:
        # User requested normal softmax to be included
        # We'll get spherical and exact kernels, AND softmax
        kernels = get_default_kernels(include_baselines=True, include_exact=True)
        # Filter to just Softmax, Spherical YAT, and maybe SLAY
        # Let's keep it focused: Softmax, YAT (Sph), SLAY, LAY, YAT (Exact)
        keep_keys = ['softmax', 'spherical_yat', 'slay', 'lay', 'yat']
        kernels = [k for k in kernels if k['key'] in keep_keys]
    
    # Grid for sphere
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Unit sphere coordinates
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    # Query at north pole [0, 0, 1]
    query = torch.tensor([0.0, 0.0, 1.0])
    
    # Keys from sphere coordinates
    keys = torch.tensor(np.stack([x, y, z], axis=-1), dtype=torch.float32)
    keys_flat = keys.reshape(-1, 3)
    
    # Setup figure
    n_kernels = len(kernels)
    n_cols = n_kernels
    n_rows = 1
    
    # Adjust width for number of kernels
    fig = plt.figure(figsize=(FULL_WIDTH, 3.0))
    fig.patch.set_facecolor('white')
    
    for idx, kernel_cfg in enumerate(kernels):
        ax = fig.add_subplot(1, n_cols, idx + 1, projection='3d')
        ax.set_facecolor('white')
        
        # Compute attention scores
        with torch.no_grad():
            weights = kernel_cfg['fn'](query, keys_flat).numpy()
            
        weights = weights.reshape(n_phi, n_theta)
        
        # Normalize for visualization (0 to 1)
        weights = weights / (weights.max() + 1e-10)
        
        # Plot surface
        surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(weights),
                               rstride=1, cstride=1, shade=False)
        
        # Draw wireframe for structure
        ax.plot_wireframe(x, y, z, color='k', alpha=0.05, rstride=10, cstride=20)
        
        # Mark query position (North Pole)
        ax.scatter([0], [0], [1.1], color='r', s=50, marker='*', zorder=10)
        
        ax.set_title(kernel_cfg['name'], fontsize=10)
        ax.axis('off')
        
        # Keep aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # --- Save individual plot ---
        # Sanitize filename: remove LaTeX commands and special chars
        safe_name = kernel_cfg['name'].replace(' ', '_').lower()
        safe_name = safe_name.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        safe_name = safe_name.replace('ⵟ', 'yat').replace('mathcal', '').replace('_', '')
        # Ensure it's clean
        safe_name = "".join([c for c in safe_name if c.isalnum() or c=='_'])
        
        indiv_path = output_path.replace('.pdf', f'_{safe_name}.pdf')
        
        # Create separate figure for individual plot
        fig_indiv = plt.figure(figsize=(4, 4))
        ax_indiv = fig_indiv.add_subplot(111, projection='3d')
        ax_indiv.set_facecolor('white')
        
        surf = ax_indiv.plot_surface(x, y, z, facecolors=plt.cm.viridis(weights),
                               rstride=1, cstride=1, shade=False)
        ax_indiv.plot_wireframe(x, y, z, color='k', alpha=0.05, rstride=10, cstride=20)
        ax_indiv.scatter([0], [0], [1.1], color='r', s=50, marker='*', zorder=10)
        ax_indiv.set_title(kernel_cfg['name'], fontsize=12)
        ax_indiv.axis('off')
        ax_indiv.set_box_aspect([1,1,1])
        
        fig_indiv.tight_layout()
        fig_indiv.savefig(indiv_path, format='pdf', dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close(fig_indiv)
        print(f"  [OK] Saved individual: {indiv_path}")
    
    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'grid_resolution': {'n_theta': n_theta, 'n_phi': n_phi},
        'query_position': 'north pole [0, 0, 1]',
        'kernels_compared': [k['name'] for k in kernels],
    }, 
    description="3D spherical heatmap of attention weights on S^2")
    
    print(f"  [OK] Data log: {log_path}")
    return output_path


def plot_polar_projection(output_path='spherical_polar.pdf', kernels=None):
    """
    2D polar projection showing attention weights.
    Easier to read than 3D for paper.
    """
    # Use default kernels if not specified
    if kernels is None:
         # Same logic: Softmax vs YAT Spherical vs SLAY, LAY, YAT
        kernels = get_default_kernels(include_baselines=True, include_exact=True)
        keep_keys = ['softmax', 'spherical_yat', 'slay', 'lay', 'yat']
        kernels = [k for k in kernels if k['key'] in keep_keys]
    
    n_kernels = len(kernels)
    
    fig, axes = plt.subplots(1, n_kernels, figsize=(FULL_WIDTH, 2.0), 
                              subplot_kw=dict(projection='polar'))
    if n_kernels == 1:
        axes = [axes]
        
    fig.patch.set_facecolor('white')
    
    # Fixed query at angle 0
    query = torch.tensor([1.0, 0.0, 0.0])
    
    # Generate circle of keys at different angles
    n_angles = 100
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    
    # Keys on 2D unit circle embedded in 3D
    keys = torch.tensor(np.stack([np.cos(angles), np.sin(angles), np.zeros(n_angles)], axis=-1), 
                        dtype=torch.float32)
    
    for ax, kernel_cfg in zip(axes, kernels):
        ax.set_facecolor('white')
        
        kernel_fn = kernel_cfg['fn']
        color = kernel_cfg['color']
        title = kernel_cfg['name']
        
        with torch.no_grad():
            weights = kernel_fn(query, keys).numpy()
        
        # Scale for visualization (normalize max to 1)
        weights = weights / (weights.max() + 1e-10)
        
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
        
        # --- Save individual plot ---
        safe_name = kernel_cfg['name'].replace(' ', '_').lower()
        safe_name = safe_name.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        safe_name = safe_name.replace('ⵟ', 'yat').replace('mathcal', '').replace('_', '')
        safe_name = "".join([c for c in safe_name if c.isalnum() or c=='_'])
        
        indiv_path = output_path.replace('.pdf', f'_{safe_name}.pdf')
        
        fig_indiv = plt.figure(figsize=(3, 3))
        ax_indiv = fig_indiv.add_subplot(111, projection='polar')
        ax_indiv.set_facecolor('white')
        
        ax_indiv.fill(np.append(angles, angles[0]), np.append(weights, weights[0]), color=color, alpha=0.5)
        ax_indiv.plot(np.append(angles, angles[0]), np.append(weights, weights[0]), color=color, linewidth=1.5)
        ax_indiv.scatter([0], [1.1], color='red', s=40, marker='*', zorder=10)
        
        ax_indiv.set_title(title, fontsize=10, pad=10)
        ax_indiv.set_ylim(0, 1.2)
        ax_indiv.set_yticks([0.5, 1.0])
        ax_indiv.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax_indiv.grid(True, alpha=0.3)
        
        fig_indiv.tight_layout()
        fig_indiv.savefig(indiv_path, format='pdf', dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
        plt.close(fig_indiv)
        print(f"  [OK] Saved individual: {indiv_path}")
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'num_angles': n_angles,
        'angles_radians': angles,
        'kernels_compared': [k['name'] for k in kernels],
    }, 
    description="2D polar projection of attention weights")
    
    print(f"  [OK] Data log: {log_path}")
    return output_path


def main():
    print("=" * 60)
    print(" ICML Figure: Spherical Heatmap Visualization")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    # We rely on defaults in function to select softmax/spherical/slay
    print("\n[1/2] Generating 3D spherical heatmaps...")
    path1 = plot_spherical_heatmaps('assets/spherical_heatmap.pdf')
    print(f"  [OK] Saved: {path1}")
    
    print("\n[2/2] Generating 2D polar projection...")
    path2 = plot_polar_projection('assets/spherical_polar.pdf')
    print(f"  [OK] Saved: {path2}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}")
    print(f"  • {path2}")
    print("=" * 60)


if __name__ == "__main__":
    main()
