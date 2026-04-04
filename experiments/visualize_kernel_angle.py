"""
Kernel Angle Visualization - ICML Paper Figure

Compares kernel behavior as a function of cosine similarity x and angle theta.
Demonstrates bounded nature of Spherical YAT vs unbounded Softmax.

Outputs PDF with vector graphics.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from viz_utils import (
    COLORS, DS, setup_icml_style, log_data,
    attention_softmax, attention_yat, attention_spherical_yat,
    attention_slay, attention_lay, DEFAULT_KERNELS, get_default_kernels
)

# Apply unified publication settings
setup_icml_style()

# Use design system constants
COLUMN_WIDTH = DS.COLUMN_WIDTH
FULL_WIDTH = DS.FULL_WIDTH

# Add additional colors for this visualization if needed
COLORS['x_squared'] = COLORS['polynomial']
COLORS['cosine'] = '#33BBEE'  # Cyan

np.random.seed(42)

# ============================================================================
# Kernel Functions (Local wrappers for simple 1D plotting if needed, 
# or use viz_utils ones with dummy dimensions)
# ============================================================================
# For 1D plots against x, it's easier to define the mathematical form directly
def softmax_1d(x): return np.exp(x)
def poly_1d(x): return x**2
def linear_1d(x): return x
def spherical_yat_1d(x, epsilon=1e-2): return (x**2) / (2 - 2*x + epsilon)
def yat_1d(x, epsilon=1e-2): return (x**2) / (2 - 2*x + epsilon) # Same as spherical for unit vectors

# ============================================================================
# Plotting
# ============================================================================
def plot_kernel_comparison(output_path='kernel_comparison.pdf', kernels=None):
    """
    Create publication-quality kernel comparison figure.
    
    Args:
        output_path: Path to save PDF
        kernels: List of kernel configs to visualize
    """
    # Use default kernels if not specified
    if kernels is None:
        kernels = DEFAULT_KERNELS
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    
    x = np.linspace(-0.95, 0.95, 300)
    
    # ------------------------------------------------------------------------
    # Panel (a): Kernel Value vs Cosine Similarity (Linear Scale)
    # ------------------------------------------------------------------------
    ax = axes[0]
    ax.set_facecolor('white')
    
    # Always show baselines for context
    ax.plot(x, linear_1d(x), color=COLORS['cosine'], 
            label='Linear $x$', linestyle=':', alpha=0.6)
    ax.plot(x, poly_1d(x), color=COLORS['x_squared'], 
            label='Poly $x^2$', linestyle=':', alpha=0.6)
    
    # Plot selected kernels
    # Specifically ensuring LAY, SLAY, YAT, Spherical YAT are present
    # These match the exact forms for unit vectors anyway, but we plot distinct lines
    # to show they are conceptually different (even if overlapping).
    
    # Manual plotting order for clarity
    ax.plot(x, softmax_1d(x), color=COLORS['softmax'], label='Softmax', linestyle='-')
    ax.plot(x, yat_1d(x), color=COLORS['yat_exact'], label='ⵟ (YAT)', linestyle='--')
    ax.plot(x, spherical_yat_1d(x), color=COLORS['yat_spherical'], label='ⵟ$_{sph}$', linestyle='-.')
    
    # LAY and SLAY approximations (idealized) match the exact forms in expectation
    # We can plot them as overlapping or just mention in caption.
    # User requested adding them. We will add them with markers to distinguish.
    ax.plot(x[::15], yat_1d(x[::15]), color=COLORS['lay'], label='LAY', linestyle='', marker='o', markersize=3, alpha=0.7)
    ax.plot(x[::15], spherical_yat_1d(x[::15]), color=COLORS['slay'], label='SLAY', linestyle='', marker='s', markersize=3, alpha=0.7)

    ax.set_xlabel('Cosine similarity $x$')
    ax.set_ylabel('Kernel shape $k(x)$')
    ax.set_title('(a) Kernel Decay Profile')
    ax.set_ylim(-0.2, 5.0)
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, fontsize=6, loc='upper left', ncol=2)

    
    # ------------------------------------------------------------------------
    # Panel (b): Log Scale (Positive x) - Growth Rate
    # ------------------------------------------------------------------------
    ax = axes[1]
    ax.set_facecolor('white')
    
    x_pos = np.linspace(0.01, 0.99, 300)
    
    ax.semilogy(x_pos, softmax_1d(x_pos), color=COLORS['softmax'], label='Softmax', linestyle='-')
    ax.semilogy(x_pos, poly_1d(x_pos), color=COLORS['x_squared'], label='$x^2$', linestyle=':')
    ax.semilogy(x_pos, yat_1d(x_pos), color=COLORS['yat_exact'], label='ⵟ (YAT)', linestyle='--')
    ax.semilogy(x_pos, spherical_yat_1d(x_pos), color=COLORS['yat_spherical'], label='ⵟ$_{sph}$', linestyle='-.')
    
    # LAY / SLAY markers
    ax.semilogy(x_pos[::15], yat_1d(x_pos[::15]), color=COLORS['lay'], label='LAY', linestyle='', marker='o', markersize=3, alpha=0.7)
    ax.semilogy(x_pos[::15], spherical_yat_1d(x_pos[::15]), color=COLORS['slay'], label='SLAY', linestyle='', marker='s', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Cosine similarity $x$')
    ax.set_ylabel('Kernel value (log scale)')
    ax.set_title('(b) Growth Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-4, 100)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(frameon=False, fontsize=6, loc='lower right', ncol=1)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {'kernels': [k['name'] for k in kernels]}, 
             description="Kernel profile comparison")
    
    return output_path


def plot_angle_based_comparison(output_path='kernel_angle.pdf'):
    """Plot kernels as a function of angle θ."""
    
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    
    theta = np.linspace(0, 180, 500)
    theta_rad = np.deg2rad(theta)
    x = np.cos(theta_rad)
    
    # -------------------- Panel (a): Linear Scale --------------------
    ax = axes[0]
    ax.set_facecolor('white')
    
    # Baselines (make them slightly recessed)
    ax.plot(theta, softmax_1d(x), color=COLORS['softmax'], label='Softmax', 
            linestyle='-', linewidth=1.5, alpha=0.9)
    ax.plot(theta, poly_1d(x), color=COLORS['x_squared'], label='$x^2$', 
            linestyle=':', linewidth=1.2, alpha=0.7)
    
    # Targets (make them thicker)
    ax.plot(theta, yat_1d(x), color=COLORS['yat_exact'], label='ⵟ (YAT)', 
            linestyle='--', linewidth=2.0)
    ax.plot(theta, spherical_yat_1d(x), color=COLORS['yat_spherical'], label='ⵟ$_{sph}$', 
            linestyle='-.', linewidth=2.0)
    
    # Approximations (bigger markers, closer spacing)
    # Reduced stride to 20 for more visibility
    mark_idx = np.arange(0, len(theta), 20)
    ax.plot(theta[mark_idx], yat_1d(x[mark_idx]), color=COLORS['lay'], label='LAY', 
            linestyle='', marker='o', markersize=4.5, markeredgewidth=0.5, markeredgecolor='white', alpha=0.9)
    ax.plot(theta[mark_idx], spherical_yat_1d(x[mark_idx]), color=COLORS['slay'], label='SLAY', 
            linestyle='', marker='s', markersize=4.5, markeredgewidth=0.5, markeredgecolor='white', alpha=0.9)
    
    ax.set_xlabel('Angle $\\theta$ (degrees)')
    ax.set_ylabel('Kernel value $k(\\theta)$')
    ax.set_title('(a) Kernel Profile (Linear)')
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.1, 4.0) 
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=6)
    
    # -------------------- Panel (b): Log Scale --------------------
    ax = axes[1]
    ax.set_facecolor('white')
    
    # Use log scale on Y to show the peak at theta=0
    ax.semilogy(theta, softmax_1d(x), color=COLORS['softmax'], label='Softmax', 
                linestyle='-', linewidth=1.5, alpha=0.9)
    ax.semilogy(theta, poly_1d(x), color=COLORS['x_squared'], label='$x^2$', 
                linestyle=':', linewidth=1.2, alpha=0.7)
    ax.semilogy(theta, yat_1d(x), color=COLORS['yat_exact'], label='ⵟ', 
                linestyle='--', linewidth=2.0)
    ax.semilogy(theta, spherical_yat_1d(x), color=COLORS['yat_spherical'], label='ⵟ$_{sph}$', 
                linestyle='-.', linewidth=2.0)
    
    # Markers
    ax.semilogy(theta[mark_idx], yat_1d(x[mark_idx]), color=COLORS['lay'], label='LAY', 
                linestyle='', marker='o', markersize=4.5, markeredgewidth=0.5, markeredgecolor='white', alpha=0.9)
    ax.semilogy(theta[mark_idx], spherical_yat_1d(x[mark_idx]), color=COLORS['slay'], label='SLAY', 
                linestyle='', marker='s', markersize=4.5, markeredgewidth=0.5, markeredgecolor='white', alpha=0.9)
    
    ax.set_xlabel('Angle $\\theta$ (degrees)')
    ax.set_ylabel('Kernel value (log scale)')
    ax.set_title('(b) Peak Behavior (Log)')
    ax.set_xlim(0, 180)
    ax.set_ylim(1e-2, 200)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {'theta': theta}, description="Angle-based kernel comparison")
    return output_path


def plot_kernel_derivatives(output_path='kernel_derivatives.pdf'):
    """Plot kernel gradients (self-regularization)."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    x = np.linspace(-0.9, 0.9, 500)
    dx = x[1] - x[0]
    
    # Compute numerical derivatives
    softmax_grad = np.gradient(softmax_1d(x), dx)
    sph_yat_grad = np.gradient(spherical_yat_1d(x), dx)
    yat_grad = np.gradient(yat_1d(x), dx)
    poly_grad = np.gradient(poly_1d(x), dx)
    
    ax.plot(x, softmax_grad, color=COLORS['softmax'], label='Softmax', linestyle='-')
    ax.plot(x, poly_grad, color=COLORS['x_squared'], label='$x^2$', linestyle=':')
    ax.plot(x, yat_grad, color=COLORS['yat_exact'], label='ⵟ (YAT)', linestyle='--')
    ax.plot(x, sph_yat_grad, color=COLORS['yat_spherical'], label='ⵟ$_{sph}$', linestyle='-.')
    
    # LAY / SLAY markers for derivative
    ax.plot(x[::20], yat_grad[::20], color=COLORS['lay'], label='LAY', linestyle='', marker='o', markersize=3, alpha=0.7)
    ax.plot(x[::20], sph_yat_grad[::20], color=COLORS['slay'], label='SLAY', linestyle='', marker='s', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Cosine similarity $x$')
    ax.set_ylabel('Gradient $dk/dx$')
    ax.set_xlim(-1, 1)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=6, ncol=2)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_title('Kernel gradients (self-regularization)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {'x': x}, description="Kernel gradients")
    return output_path

def plot_individual_kernels(output_dir='assets'):
    """Generate separate plots for each kernel to allow detailed inspection."""
    
    # Define kernels to plot individually
    kernels = [
        {'name': 'Softmax', 'fn': softmax_1d, 'color': COLORS['softmax'], 'style': '-'},
        {'name': 'Poly x²', 'fn': poly_1d, 'color': COLORS['x_squared'], 'style': ':'},
        {'name': 'ⵟ (YAT)', 'fn': yat_1d, 'color': COLORS['yat_exact'], 'style': '--'},
        {'name': 'ⵟ_sph', 'fn': spherical_yat_1d, 'color': COLORS['yat_spherical'], 'style': '-.'}
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    
    # Common X axis
    x = np.linspace(-0.95, 0.95, 300)
    theta = np.linspace(0, 180, 500)
    theta_rad = np.deg2rad(theta)
    x_theta = np.cos(theta_rad)
    
    for k in kernels:
        # File name safe
        safe_name = k['name'].replace(' ', '_').replace('ⵟ', 'YAT').replace('$x^2$', 'Poly').replace('_sph', '_Sph').lower()
        fname = f"{output_dir}/kernel_profile_{safe_name}.pdf"
        
        fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))
        fig.patch.set_facecolor('white')
        
        # Panel 1: Linear scale vs Cosine Sim
        ax = axes[0]
        ax.set_facecolor('white')
        y_lin = k['fn'](x)
        ax.plot(x, y_lin, color=k['color'], linestyle=k['style'], linewidth=2.0)
        
        ax.set_xlabel('Cosine similarity $x$')
        ax.set_ylabel(f'{k["name"]} Value')
        ax.set_title(f'(a) {k["name"]} Profile (Linear)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5)
        
        # Panel 2: Log scale vs Angle
        ax = axes[1]
        ax.set_facecolor('white')
        y_log = k['fn'](x_theta)
        ax.semilogy(theta, y_log, color=k['color'], linestyle=k['style'], linewidth=2.0)
        
        ax.set_xlabel('Angle $\\theta$ (degrees)')
        ax.set_ylabel('Value (Log Scale)')
        ax.set_title(f'(b) {k["name"]} Peak (Log)')
        ax.set_xlim(0, 180)
        ax.set_ylim(1e-2, max(y_log.max(), 10) * 1.5)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()
        paths.append(fname)
        
    return paths


def main():
    print("=" * 60)
    print(" ICML Figure: Kernel vs Angle Comparison")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/4] Generating kernel comparison plot...")
    path1 = plot_kernel_comparison('assets/kernel_comparison.pdf')
    print(f"  [OK] Saved: {path1}")
    
    print("\n[2/4] Generating angle-based plot...")
    path2 = plot_angle_based_comparison('assets/kernel_angle.pdf')
    print(f"  [OK] Saved: {path2}")
    
    print("\n[3/4] Generating kernel derivatives plot...")
    path3 = plot_kernel_derivatives('assets/kernel_derivatives.pdf')
    print(f"  [OK] Saved: {path3}")
    
    print("\n[4/4] Generating individual kernel profiles...")
    paths = plot_individual_kernels('assets')
    for p in paths:
        print(f"  [OK] Saved: {p}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}")
    print(f"  • {path2}")
    print(f"  • {path3}")
    for p in paths:
        print(f"  • {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
