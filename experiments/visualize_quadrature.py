"""
Quadrature Contribution Visualization - ICML Paper Figure

Shows the contribution of each Gauss-Laguerre quadrature node.
Demonstrates that only a few scales matter, justifying small R.

Outputs PDF with vector graphics for camera-ready quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.integrate import quad
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

# Colorblind-safe colors
COLORS = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377', '#BBBBBB']

np.random.seed(42)


# ============================================================================
# Quadrature Analysis
# ============================================================================
def integrand(s, x, C):
    """
    The integrand from Eq. 4 in the paper:
    h(s) = x^2 * exp(2*s*x)
    
    The full integral is: ∫₀^∞ e^(-sC) h(s) ds = x² / (C - 2x)
    """
    return (x ** 2) * np.exp(2 * s * x)


def compute_quadrature_contributions(x_vals, R, epsilon=1e-2):
    """
    Compute the contribution of each Gauss-Laguerre node to the integral.
    
    Paper: After change of variables t = Cs:
    s_r = t_r / C,  w_r = α_r / C
    
    Contribution at node r: w_r * x² * exp(2*s_r*x)
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
    """
    Compute approximation error for different numbers of quadrature nodes.
    """
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
    Shows that only a few scales matter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    
    epsilon = 1e-2
    
    # Different x values (cosine similarities)
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
        
        bars = ax.bar(range(R), contrib_pct, color=COLORS[:R], alpha=0.8, 
                      edgecolor='white', linewidth=0.8)
        
        ax.set_xlabel('Quadrature node $r$')
        if idx == 0:
            ax.set_ylabel('Contribution (%)')
        ax.set_title(f'$x = {x_val}$', fontsize=9)
        ax.set_xticks(range(R))
        ax.set_xticklabels([f'{r+1}' for r in range(R)])
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annotate with s values
        for i, bar in enumerate(bars):
            if contrib_pct[i] > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{contrib_pct[i]:.0f}%', ha='center', va='bottom', fontsize=6)
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'x_samples': np.array(x_samples),
        'num_nodes_R': 5,
        'epsilon': epsilon,
    }, description="Quadrature node contributions at different x values")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_node_locations(output_path='quadrature_nodes.pdf'):
    """
    Visualize the location of quadrature nodes and their weights.
    """
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
        
        # Offset for visibility
        y_offset = i
        
        for s, w in zip(s_vals, w_vals):
            ax.scatter(s, y_offset, s=w * 500 + 10, c=COLORS[i], alpha=0.7, 
                       edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel('Quadrature node $s_r = t_r / C$')
    ax.set_ylabel('Number of nodes $R$')
    ax.set_yticks(range(len(R_values)))
    ax.set_yticklabels([str(r) for r in R_values])
    ax.set_xlim(-0.1, 4)
    ax.set_title('Gauss-Laguerre node locations (size ∝ weight)')
    ax.grid(True, alpha=0.3, axis='x')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'R_values_tested': R_values,
        'epsilon': epsilon,
        'C': C,
    }, description="Gauss-Laguerre node locations for different R values")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_convergence_vs_nodes(output_path='quadrature_convergence.pdf'):
    """
    Plot approximation error vs number of quadrature nodes.
    Shows rapid convergence with small R.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    fig.patch.set_facecolor('white')
    
    epsilon = 1e-2
    x_vals = np.linspace(-0.9, 0.9, 100)
    R_values = [1, 2, 3, 5, 8, 10, 15]
    
    # ---- Panel (a): Error vs R ----
    ax = axes[0]
    ax.set_facecolor('white')
    
    errors = compute_quadrature_error(R_values, x_vals, epsilon)
    
    ax.semilogy(R_values, errors, 'o-', color=COLORS[0], markersize=6, linewidth=1.5)
    
    ax.set_xlabel('Number of quadrature nodes $R$')
    ax.set_ylabel('Mean relative error')
    ax.set_title('(a) Quadrature convergence')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 16)
    
    # Mark typical choice
    ax.axvline(x=3, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(3.2, errors[2] * 1.5, 'R=3\n(default)', fontsize=7, va='bottom')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    # ---- Panel (b): Kernel approximation for different R ----
    ax = axes[1]
    ax.set_facecolor('white')
    
    exact = compute_exact_kernel(x_vals, epsilon)
    ax.plot(x_vals, exact, 'k-', linewidth=2, label='Exact')
    
    for R, color in zip([1, 2, 3, 5], COLORS[:4]):
        _, _, contributions = compute_quadrature_contributions(x_vals, R, epsilon)
        approx = contributions.sum(axis=0)
        ax.plot(x_vals, approx, '--', color=color, linewidth=1.5, label=f'R={R}')
    
    ax.set_xlabel('Cosine similarity $x$')
    ax.set_ylabel('Kernel value')
    ax.set_title('(b) Kernel approximation quality')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'R_values': np.array(R_values),
        'mean_relative_errors': np.array(errors),
        'x_values': x_vals,
        'exact_kernel': compute_exact_kernel(x_vals, epsilon),
    }, description="Quadrature convergence: error vs number of nodes")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


def plot_expected_contribution(output_path='expected_contribution.pdf'):
    """
    Bar plot showing: w_r * E[x² e^(2s_r x)]
    Averaged over typical x distribution.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    R = 5
    epsilon = 1e-2
    
    # Sample x values from a reasonable distribution
    # Assuming q·k is roughly uniform on [-1, 1] for random vectors
    n_samples = 10000
    x_samples = np.random.uniform(-0.9, 0.9, n_samples)
    
    s_vals, w_vals, contributions = compute_quadrature_contributions(
        x_samples, R, epsilon
    )
    
    # Expected contribution per node
    expected_contrib = contributions.mean(axis=1)
    total = expected_contrib.sum()
    expected_pct = 100 * expected_contrib / total
    
    bars = ax.bar(range(R), expected_pct, color=COLORS[:R], alpha=0.8,
                  edgecolor='white', linewidth=0.8)
    
    ax.set_xlabel('Quadrature node $r$')
    ax.set_ylabel('Expected contribution (%)')
    ax.set_title(f'$w_r \\cdot \\mathbb{{E}}[x^2 e^{{2s_r x}}]$ (R={R})')
    ax.set_xticks(range(R))
    ax.set_xticklabels([f'{r+1}\n($s={s_vals[r]:.2f}$)' for r in range(R)], fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{expected_pct[i]:.1f}%', ha='center', va='bottom', fontsize=7)
    
    # Add insight annotation
    ax.text(0.95, 0.95, f'First 2 nodes:\n{expected_pct[:2].sum():.0f}% of total',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # Log data
    log_path = output_path.replace('.pdf', '_data.txt')
    log_data(log_path, {
        'num_nodes_R': R,
        's_values': s_vals,
        'w_values': w_vals,
        'expected_contribution_pct': expected_pct,
        'first_2_nodes_pct': float(expected_pct[:2].sum()),
    }, description="Expected quadrature contributions E[w_r * x^2 * exp(2s_r x)]")
    print(f"  ✓ Data log: {log_path}")
    
    return output_path


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print(" ICML Figure: Quadrature Analysis")
    print("=" * 60)
    
    os.makedirs('assets', exist_ok=True)
    
    print("\n[1/4] Generating quadrature contributions plot...")
    path1 = plot_quadrature_contributions('assets/quadrature_contributions.pdf')
    print(f"  ✓ Saved: {path1}")
    
    print("\n[2/4] Generating node locations plot...")
    path2 = plot_node_locations('assets/quadrature_nodes.pdf')
    print(f"  ✓ Saved: {path2}")
    
    print("\n[3/4] Generating convergence plot...")
    path3 = plot_convergence_vs_nodes('assets/quadrature_convergence.pdf')
    print(f"  ✓ Saved: {path3}")
    
    print("\n[4/4] Generating expected contribution plot...")
    path4 = plot_expected_contribution('assets/expected_contribution.pdf')
    print(f"  ✓ Saved: {path4}")
    
    print("\n" + "=" * 60)
    print(" Generated figures:")
    print(f"  • {path1}  (per-x contributions)")
    print(f"  • {path2}  (node positions)")
    print(f"  • {path3}  (convergence, main figure)")
    print(f"  • {path4}  (expected contributions)")
    print("=" * 60)


if __name__ == "__main__":
    main()
