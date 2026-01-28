"""
Kernel vs Angle Visualization - ICML Paper Figure

Publication-quality figure showing how different kernel functions behave
as a function of angle/cosine similarity. Key visualization to show the
bounded, self-regularizing nature of the spherical YAT kernel.

Outputs PDF with vector graphics for camera-ready quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from datetime import datetime

# ============================================================================
# Publication Settings (ICML)
# ============================================================================
COLUMN_WIDTH = 3.25  # Single column width
FULL_WIDTH = 6.75    # Full page width

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['lines.linewidth'] = 1.5

# Colorblind-safe palette (Tableau)
COLORS = {
    'softmax': '#0077BB',      # Blue
    'cosine': '#33BBEE',       # Cyan
    'x_squared': '#009988',    # Teal
    'yat': '#EE7733',          # Orange
    'spherical_yat': '#CC3311', # Red
    'slay': '#EE3377',         # Magenta
}

np.random.seed(42)


# ============================================================================
# Kernel Functions
# ============================================================================
def softmax_kernel(x, scale=1.0):
    """Softmax-style exponential kernel: k(x) = exp(scale * x)"""
    return np.exp(scale * x)


def cosine_kernel(x):
    """Simple cosine/linear kernel: k(x) = x"""
    return x


def pure_polynomial_kernel(x):
    """Pure squared kernel: k(x) = x²"""
    return x ** 2


def yat_kernel(x, epsilon=1e-2, norm_q=1.0, norm_k=1.0):
    """
    YAT kernel (unnormalized vectors):
    k(q, k) = (q·k)² / (||q-k||² + ε)
    
    For vectors of norms ||q|| and ||k|| with dot product x = q·k:
    ||q-k||² = ||q||² + ||k||² - 2·q·k
    
    When plotting vs cosine similarity x/||q||||k||, we fix norms.
    """
    dot = x * norm_q * norm_k
    dist_sq = norm_q**2 + norm_k**2 - 2 * dot
    return (dot ** 2) / (dist_sq + epsilon)


def spherical_yat_kernel(x, epsilon=1e-2):
    """
    Spherical YAT kernel (normalized to unit sphere):
    k(x) = x² / (C - 2x) where x = q̂·k̂ and C = 2 + ε
    """
    C = 2.0 + epsilon
    return (x ** 2) / (C - 2 * x)


def relu_kernel(x):
    """
    ReLU-based kernel (like FAVOR+):
    k(x) = max(0, x)²
    """
    return np.maximum(0, x) ** 2


def elu_plus_one_kernel(x):
    """
    ELU+1 kernel (linear attention):
    k(x) = (elu(x) + 1)² approximately
    """
    elu_x = np.where(x > 0, x, np.exp(x) - 1)
    return (elu_x + 1) ** 2 / 4  # Normalized for visualization


# ============================================================================
# Data Logging
# ============================================================================
def log_data(filename, data_dict, description=""):
    """
    Log plot data to a text file for LLM analysis.
    
    Args:
        filename: Output txt file path
        data_dict: Dictionary of arrays/values to log
        description: Optional description of the data
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else 'assets', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# SLAY Visualization Data Log\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Description: {description}\n")
        f.write(f"#" + "="*60 + "\n\n")
        
        for key, value in data_dict.items():
            f.write(f"## {key}\n")
            if isinstance(value, np.ndarray):
                f.write(f"# Shape: {value.shape}, dtype: {value.dtype}\n")
                if value.ndim == 1 and len(value) <= 100:
                    f.write(f"# Values: {value.tolist()}\n")
                elif value.ndim == 1:
                    f.write(f"# First 20: {value[:20].tolist()}\n")
                    f.write(f"# Last 20: {value[-20:].tolist()}\n")
                    f.write(f"# Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}\n")
                else:
                    f.write(f"# Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"# Length: {len(value)}\n")
                f.write(f"# Values: {value[:50]}...\n" if len(value) > 50 else f"# Values: {value}\n")
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, np.ndarray):
                        f.write(f"#   {sub_key}: shape={sub_val.shape}, min={sub_val.min():.4f}, max={sub_val.max():.4f}\n")
                    else:
                        f.write(f"#   {sub_key}: {sub_val}\n")
            else:
                f.write(f"# Value: {value}\n")
            f.write("\n")
    
    return filename


# ============================================================================
# Plotting
# ============================================================================
def plot_kernel_comparison(output_path='kernel_comparison.pdf'):
    """Create publication-quality kernel comparison figure."""
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    fig.patch.set_facecolor('white')
    
    # X values: cosine similarity from -1 to 1
    x = np.linspace(-0.99, 0.99, 500)
    
    # -------------------- Panel (a): Linear scale --------------------
    ax1.set_facecolor('white')
    
    # Plot kernels
    ax1.plot(x, softmax_kernel(x, scale=1.0), 
             color=COLORS['softmax'], label='Softmax $e^x$', linestyle='-')
    ax1.plot(x, cosine_kernel(x), 
             color=COLORS['cosine'], label='Cosine $x$', linestyle='--')
    ax1.plot(x, pure_polynomial_kernel(x), 
             color=COLORS['x_squared'], label='Polynomial $x^2$', linestyle='-.')
    ax1.plot(x, yat_kernel(x, norm_q=1.5, norm_k=1.2), 
             color=COLORS['yat'], label='ⵟ (YAT)', linestyle=':')
    ax1.plot(x, spherical_yat_kernel(x), 
             color=COLORS['spherical_yat'], label='ⵟ$_{sph}$ (Sph. YAT)', linestyle='-', linewidth=2)
    
    ax1.set_xlabel('Cosine similarity $x = \\hat{q}^\\top \\hat{k}$')
    ax1.set_ylabel('Kernel value $k(x)$')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-0.5, 5)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=7)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
    ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
    ax1.set_title('(a) Kernel functions (linear scale)')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add insight annotation
    ax1.annotate('Bounded\n& self-\nregularizing', 
                 xy=(0.8, spherical_yat_kernel(np.array([0.8]))[0]), 
                 xytext=(0.4, 4),
                 fontsize=7, ha='center',
                 arrowprops=dict(arrowstyle='->', color=COLORS['spherical_yat'], lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=COLORS['spherical_yat']))
    
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)
    
    # -------------------- Panel (b): Log scale (positive x only) --------------------
    ax2.set_facecolor('white')
    
    x_pos = np.linspace(0.01, 0.99, 500)
    
    ax2.semilogy(x_pos, softmax_kernel(x_pos, scale=1.0), 
                 color=COLORS['softmax'], label='Softmax $e^x$', linestyle='-')
    ax2.semilogy(x_pos, pure_polynomial_kernel(x_pos), 
                 color=COLORS['x_squared'], label='Polynomial $x^2$', linestyle='-.')
    ax2.semilogy(x_pos, yat_kernel(x_pos, norm_q=1.5, norm_k=1.2), 
                 color=COLORS['yat'], label='ⵟ (YAT)', linestyle=':')
    ax2.semilogy(x_pos, spherical_yat_kernel(x_pos), 
                 color=COLORS['spherical_yat'], label='ⵟ$_{sph}$ (Sph. YAT)', linestyle='-', linewidth=2)
    
    ax2.set_xlabel('Cosine similarity $x$')
    ax2.set_ylabel('Kernel value $k(x)$ (log scale)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1e-4, 1e2)
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=7)
    ax2.set_title('(b) Log scale (aligned tokens)')
    ax2.grid(True, alpha=0.3, linewidth=0.5, which='both')
    
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', dpi=300, 
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'x_values': x,
        'softmax': softmax_kernel(x, scale=1.0),
        'cosine': cosine_kernel(x),
        'x_squared': pure_polynomial_kernel(x),
        'yat': yat_kernel(x, norm_q=1.5, norm_k=1.2),
        'spherical_yat': spherical_yat_kernel(x),
        'parameters': {'epsilon': 1e-2, 'yat_norm_q': 1.5, 'yat_norm_k': 1.2}
    }, description="Kernel functions comparison: k(x) for different attention mechanisms")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_angle_based_comparison(output_path='kernel_angle.pdf'):
    """Plot kernels as a function of angle θ instead of x."""
    
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Angle from 0 to 180 degrees
    theta = np.linspace(1, 179, 500)  # Avoid exact 0 and 180
    theta_rad = np.deg2rad(theta)
    x = np.cos(theta_rad)
    
    # Plot kernels
    ax.plot(theta, softmax_kernel(x, scale=1.0), 
            color=COLORS['softmax'], label='Softmax', linestyle='-')
    ax.plot(theta, pure_polynomial_kernel(x), 
            color=COLORS['x_squared'], label='$x^2$', linestyle='-.')
    ax.plot(theta, spherical_yat_kernel(x), 
            color=COLORS['spherical_yat'], label='ⵟ$_{sph}$', linestyle='-', linewidth=2)
    
    ax.set_xlabel('Angle $\\theta$ (degrees)')
    ax.set_ylabel('Kernel value $k(\\cos\\theta)$')
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 3)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Mark key angles
    ax.axvline(x=90, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    ax.annotate('orthogonal', xy=(90, 0.1), fontsize=7, ha='center')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'theta_degrees': theta,
        'cosine_similarity': x,
        'softmax': softmax_kernel(x, scale=1.0),
        'x_squared': pure_polynomial_kernel(x),
        'spherical_yat': spherical_yat_kernel(x),
    }, description="Kernel functions as function of angle theta")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_kernel_derivatives(output_path='kernel_derivatives.pdf'):
    """
    Plot kernel gradients to show self-regularization behavior.
    Shows how spherical YAT has bounded gradient unlike softmax.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    x = np.linspace(-0.9, 0.9, 500)
    dx = x[1] - x[0]
    
    # Compute numerical derivatives
    softmax_vals = softmax_kernel(x)
    sph_yat_vals = spherical_yat_kernel(x)
    poly_vals = pure_polynomial_kernel(x)
    
    softmax_grad = np.gradient(softmax_vals, dx)
    sph_yat_grad = np.gradient(sph_yat_vals, dx)
    poly_grad = np.gradient(poly_vals, dx)
    
    ax.plot(x, softmax_grad, color=COLORS['softmax'], label='Softmax', linestyle='-')
    ax.plot(x, poly_grad, color=COLORS['x_squared'], label='$x^2$', linestyle='-.')
    ax.plot(x, sph_yat_grad, color=COLORS['spherical_yat'], label='ⵟ$_{sph}$', linestyle='-', linewidth=2)
    
    ax.set_xlabel('Cosine similarity $x$')
    ax.set_ylabel('Gradient $dk/dx$')
    ax.set_xlim(-1, 1)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_title('Kernel gradients (self-regularization)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'x_values': x,
        'softmax_gradient': softmax_grad,
        'x_squared_gradient': poly_grad,
        'spherical_yat_gradient': sph_yat_grad,
        'statistics': {
            'softmax_max_grad': float(softmax_grad.max()),
            'sph_yat_max_grad': float(sph_yat_grad.max()),
            'x_squared_max_grad': float(poly_grad.max()),
        }
    }, description="Kernel gradients dk/dx showing self-regularization")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print(" ICML Figure: Kernel vs Angle Comparison")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/3] Generating kernel comparison plot...")
    path1 = plot_kernel_comparison('assets/kernel_comparison.pdf')
    print(f"  ✓ Saved: {path1}")
    
    print("\n[2/3] Generating angle-based plot...")
    path2 = plot_angle_based_comparison('assets/kernel_angle.pdf')
    print(f"  ✓ Saved: {path2}")
    
    print("\n[3/3] Generating kernel derivatives plot...")
    path3 = plot_kernel_derivatives('assets/kernel_derivatives.pdf')
    print(f"  ✓ Saved: {path3}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}  (main figure)")
    print(f"  • {path2}  (angle-based)")
    print(f"  • {path3}  (gradients)")
    print("=" * 60)


if __name__ == "__main__":
    main()
