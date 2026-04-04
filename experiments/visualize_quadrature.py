"""
Quadrature Contribution Visualization - ICML Paper Figure

Shows the contribution of each Gauss-Laguerre quadrature node.
Demonstrates that only a few scales matter, justifying small R.

Outputs PDF with vector graphics for camera-ready quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from viz_utils import COLORS, DS, setup_icml_style, log_data

# Apply unified publication settings
setup_icml_style()

# Use design system constants
COLUMN_WIDTH = DS.COLUMN_WIDTH
FULL_WIDTH = DS.FULL_WIDTH

# Node colors
NODE_COLORS = [COLORS['node1'], COLORS['node2'], COLORS['node3'], COLORS['node4'], COLORS['node5']]

np.random.seed(42)


# ============================================================================
# Quadrature Analysis
# ============================================================================
def compute_quadrature_contributions(x_vals, R, epsilon=1e-2):
    """
    Compute the contribution of each Gauss-Laguerre node to the integral.
    """
    C = 2.0 + epsilon
    
    # Get Gauss-Laguerre nodes and weights
    nodes, weights = np.polynomial.laguerre.laggauss(R)
    s_vals = nodes / C
    w_vals = weights / C
    
    # Compute contributions for each x value
    # Shape: (R, len(x_vals))
    contributions = np.zeros((R, len(x_vals)))
    
    for r in range(R):
        contributions[r] = w_vals[r] * (x_vals ** 2) * np.exp(2 * s_vals[r] * x_vals)
    
    return s_vals, w_vals, contributions


def compute_exact_kernel(x_vals, epsilon=1e-2):
    """Compute exact spherical YAT kernel: x² / (C - 2x)"""
    C = 2.0 + epsilon
    return (x_vals ** 2) / (C - 2 * x_vals)


def compute_quadrature_error(R_values, x_vals, epsilon=1e-2):
    """Compute approximation error for different numbers of quadrature nodes."""
    exact = compute_exact_kernel(x_vals, epsilon)
    errors = []
    
    for R in R_values:
        _, w_vals, contributions = compute_quadrature_contributions(x_vals, R, epsilon)
        approx = contributions.sum(axis=0)
        
        # Relative error
        rel_error = np.mean(np.abs(exact - approx) / (np.abs(exact) + 1e-10))
        errors.append(rel_error)
    
    return errors


# ============================================================================
# Plotting
# ============================================================================
def plot_quadrature_contributions(output_path='quadrature_contributions.pdf'):
    """
    Bar plot showing contribution of each quadrature node.
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    
    epsilon = 1e-2
    x_samples = [0.2, 0.5, 0.8]
    
    for idx, x_val in enumerate(x_samples):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        R = 5
        s_vals, w_vals, contributions = compute_quadrature_contributions(
            np.array([x_val]), R, epsilon
        )
        
        contrib_vals = contributions[:, 0]
        total = contrib_vals.sum()
        contrib_pct = 100 * contrib_vals / total
        
        bars = ax.bar(range(R), contrib_pct, color=NODE_COLORS[:R], alpha=0.8, 
                      edgecolor='white', linewidth=0.8)
        
        ax.set_xlabel('Quadrature node $r$')
        if idx == 0:
            ax.set_ylabel('Contribution (%)')
        ax.set_title(f'$x = {x_val}$', fontsize=9)
        ax.set_xticks(range(R))
        ax.set_xticklabels([f'{r+1}' for r in range(R)])
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if contrib_pct[i] > 20: 
                ax.text(bar.get_x() + bar.get_width()/2, height - 5,
                        f'{contrib_pct[i]:.0f}%', ha='center', va='top', fontsize=6, color='white', fontweight='bold')
            elif contrib_pct[i] > 5:
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{contrib_pct[i]:.0f}%', ha='center', va='bottom', fontsize=6)
        
        # --- Save individual plot ---
        indiv_path = output_path.replace('.pdf', f'_x{x_val}.pdf')
        
        fig_indiv, ax_indiv = plt.subplots(figsize=(3, 3))
        fig_indiv.patch.set_facecolor('white')
        
        ax_indiv.bar(range(R), contrib_pct, color=NODE_COLORS[:R], alpha=0.8, 
                     edgecolor='white', linewidth=0.8)
        
        ax_indiv.set_xlabel('Quadrature node $r$')
        ax_indiv.set_ylabel('Contribution (%)')
        ax_indiv.set_title(f'Node Contribution ($x = {x_val}$)', fontsize=10)
        ax_indiv.set_xticks(range(R))
        ax_indiv.set_xticklabels([f'{r+1}' for r in range(R)])
        ax_indiv.set_ylim(0, 110)
        ax_indiv.grid(True, alpha=0.3, axis='y')
        
        fig_indiv.tight_layout()
        fig_indiv.savefig(indiv_path, format='pdf', bbox_inches='tight')
        plt.close(fig_indiv)
        print(f"  [OK] Saved individual: {indiv_path}")
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {'x_samples': x_samples}, description="Quadrature contributions")
    return output_path


def plot_node_locations(output_path='quadrature_nodes.pdf'):
    """Visualize the location of quadrature nodes and their weights."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    R_values = [1, 2, 3, 5, 8]
    epsilon = 1e-2
    C = 2.0 + epsilon
    
    for i, R in enumerate(R_values):
        nodes, weights = np.polynomial.laguerre.laggauss(R)
        s_vals = nodes / C
        w_vals = weights / C
        
        y_offset = i
        for s, w in zip(s_vals, w_vals):
            ax.scatter(s, y_offset, s=w * 500 + 10, c=COLORS['node1'], alpha=0.7, 
                       edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel('Quadrature node $s_r = t_r / C$')
    ax.set_ylabel('Number of nodes $R$')
    ax.set_yticks(range(len(R_values)))
    ax.set_yticklabels([str(r) for r in R_values])
    ax.set_title('Gauss-Laguerre node locations')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {'R_values': R_values}, description="Node locations")
    return output_path


def plot_convergence_vs_nodes(output_path='quadrature_convergence.pdf'):
    """Plot approximation error vs number of quadrature nodes."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    fig.patch.set_facecolor('white')
    
    epsilon = 1e-2
    x_vals = np.linspace(-0.9, 0.9, 100)
    R_values = [1, 2, 3, 5, 8, 10, 15]
    
    # Panel (a): Error vs R
    ax = axes[0]
    ax.set_facecolor('white')
    errors = compute_quadrature_error(R_values, x_vals, epsilon)
    ax.semilogy(R_values, errors, 'o-', color=COLORS['node1'], markersize=6, linewidth=1.5)
    ax.set_xlabel('Number of quadrature nodes $R$')
    ax.set_ylabel('Mean relative error')
    ax.set_title('(a) Quadrature convergence')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 16)
    
    ax.axvline(x=3, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(3.2, errors[2] * 1.5, 'R=3\n(default)', fontsize=7, va='bottom')
    
    # Panel (b): Kernel approximation
    ax = axes[1]
    ax.set_facecolor('white')
    exact = compute_exact_kernel(x_vals, epsilon)
    ax.plot(x_vals, exact, 'k-', linewidth=2, label='Exact')
    
    for R, color in zip([1, 2, 3, 5], NODE_COLORS[:4]):
        _, _, contributions = compute_quadrature_contributions(x_vals, R, epsilon)
        approx = contributions.sum(axis=0)
        ax.plot(x_vals, approx, '--', color=color, linewidth=1.5, label=f'R={R}')
    
    ax.set_xlabel('Cosine similarity $x$')
    ax.set_ylabel('Kernel value')
    ax.set_title('(b) Kernel approximation quality')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {'R_values': R_values, 'errors': errors}, description="Convergence")
    return output_path


def main():
    print("=" * 60)
    print(" ICML Figure: Quadrature Analysis")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/3] Generating quadrature contributions plot...")
    path1 = plot_quadrature_contributions('assets/quadrature_contributions.pdf')
    print(f"  [OK] Saved: {path1}")
    
    print("\n[2/3] Generating node locations plot...")
    path2 = plot_node_locations('assets/quadrature_nodes.pdf')
    print(f"  [OK] Saved: {path2}")
    
    print("\n[3/3] Generating convergence plot...")
    path3 = plot_convergence_vs_nodes('assets/quadrature_convergence.pdf')
    print(f"  [OK] Saved: {path3}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}")
    print(f"  • {path2}")
    print(f"  • {path3}")
    print("=" * 60)


if __name__ == "__main__":
    main()
