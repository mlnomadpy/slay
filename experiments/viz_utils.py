"""
LAY Visualization Design System
================================

Unified design system for all LAY paper visualizations.
Provides: colors, typography, dimensions, kernel functions, and utilities.

Usage:
    from viz_utils import DesignSystem, COLORS, setup_icml_style, log_data
    
    setup_icml_style()  # Apply ICML publication settings
    fig, ax = plt.subplots(figsize=DesignSystem.FULL_WIDTH_SIZE)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional, List


# ============================================================================
# UNIFIED COLOR PALETTE (Colorblind-Safe)
# ============================================================================
COLORS = {
    # === Quadratic Methods (dashed lines in plots) ===
    'softmax': '#0077BB',        # Blue - Standard attention
    'yat_exact': '#EE7733',      # Orange - Pure Euclidean YAT
    'yat_spherical': '#CC3311',  # Red - Spherical YAT
    
    # === Linear Approximations (solid lines, OUR CONTRIBUTIONS) ===
    'lay': '#44AA99',            # Teal-Green - LAY (MAIN METHOD)
    'slay': '#EE3377',           # Magenta - SLAY (spherical approx)
    
    # === Other Linear Baselines ===
    'performer': '#009988',      # Green - FAVOR+
    'linear': '#332288',         # Deep Purple - Linear (ELU+1)
    'cosformer': '#AA4499',      # Pink - Cosformer
    
    # === Reference/Baseline ===
    'polynomial': '#BBBBBB',     # Grey - x² polynomial
    'reference': '#000000',      # Black - exact/reference lines
    
    # === Stability Comparison ===
    'stable': '#44AA99',         # Teal - stable/positive
    'unstable': '#CC3311',       # Red - unstable/negative
    
    # === Quadrature Nodes ===
    'node1': '#0077BB',          # Blue
    'node2': '#EE7733',          # Orange
    'node3': '#009988',          # Teal
    'node4': '#CC3311',          # Red
    'node5': '#33BBEE',          # Cyan
}

# Backward compatibility aliases
COLORS['yat'] = COLORS['yat_exact']
COLORS['spherical_yat'] = COLORS['yat_spherical']
COLORS['slay_anchor'] = COLORS['slay']
COLORS['lay_rbf'] = COLORS['lay']
COLORS['exact'] = COLORS['reference']
COLORS['favor'] = COLORS['performer']
COLORS['favor_plus'] = COLORS['performer']
COLORS['quadrature_only'] = COLORS['polynomial']
COLORS['laplace_only'] = COLORS['yat_exact']
COLORS['pure_yat'] = COLORS['yat_exact']
COLORS['tensor_sketch'] = '#0077BB'
COLORS['random_maclaurin'] = '#009988'
COLORS['nystrom'] = '#33BBEE'


# ============================================================================
# DESIGN SYSTEM CONSTANTS
# ============================================================================
@dataclass
class DesignSystem:
    """Central design system for LAY paper visualizations."""
    
    # === ICML Figure Dimensions ===
    COLUMN_WIDTH: float = 3.25      # Single column width (inches)
    FULL_WIDTH: float = 6.75        # Full page width (inches)
    
    # Pre-computed sizes for common layouts
    SINGLE_PANEL_SIZE: Tuple[float, float] = (3.25, 2.5)
    FULL_WIDTH_SIZE: Tuple[float, float] = (6.75, 2.8)
    SQUARE_SIZE: Tuple[float, float] = (3.0, 3.0)
    COMPOSITE_SIZE: Tuple[float, float] = (6.75, 2.4)
    HEATMAP_SIZE: Tuple[float, float] = (6.75, 4.5)
    
    # === Typography ===
    FONT_FAMILY: str = 'sans-serif'
    FONT_SANS: List[str] = None  # Set in __post_init__
    FONT_SIZE_BASE: int = 9
    FONT_SIZE_LABEL: int = 9
    FONT_SIZE_TITLE: int = 10
    FONT_SIZE_LEGEND: int = 7
    FONT_SIZE_TICK: int = 8
    FONT_SIZE_ANNOTATION: int = 7
    
    # === Line Styles ===
    LINE_WIDTH_THICK: float = 2.0
    LINE_WIDTH_NORMAL: float = 1.5
    LINE_WIDTH_THIN: float = 1.0
    LINE_WIDTH_AXIS: float = 0.8
    
    # === Markers ===
    MARKER_SIZE: float = 5
    MARKER_SIZE_SMALL: float = 3
    
    # === Grid and Alpha ===
    GRID_ALPHA: float = 0.3
    FILL_ALPHA: float = 0.2
    CONFIDENCE_ALPHA: float = 0.2
    
    def __post_init__(self):
        self.FONT_SANS = ['DejaVu Sans', 'Arial', 'Helvetica']


# Create singleton instance
DS = DesignSystem()


# ============================================================================
# METHOD REGISTRY (Consistent naming and styling)
# ============================================================================
@dataclass
class MethodStyle:
    """Style configuration for a method in plots."""
    name: str                    # Display name
    latex_name: str              # LaTeX-compatible name
    color: str                   # Color from COLORS dict
    linestyle: str = '-'         # Line style
    linewidth: float = 1.5       # Line width
    marker: str = ''             # Marker style
    is_quadratic: bool = False   # True for O(L²) methods
    is_our_method: bool = False  # True for LAY/SLAY


METHOD_STYLES: Dict[str, MethodStyle] = {
    'softmax': MethodStyle(
        name='Softmax', latex_name='Softmax $e^x$',
        color=COLORS['softmax'], linestyle='-', linewidth=1.5,
        is_quadratic=True
    ),
    'yat_exact': MethodStyle(
        name='ⵟ (YAT)', latex_name='ⵟ (YAT)',
        color=COLORS['yat_exact'], linestyle='--', linewidth=1.5,
        is_quadratic=True
    ),
    'yat_spherical': MethodStyle(
        name='ⵟ_sph', latex_name='ⵟ$_{sph}$',
        color=COLORS['yat_spherical'], linestyle='-.', linewidth=1.5,
        is_quadratic=True
    ),
    'lay': MethodStyle(
        name='LAY', latex_name='LAY',
        color=COLORS['lay'], linestyle='-', linewidth=2.0,
        marker='o', is_our_method=True
    ),
    'slay': MethodStyle(
        name='SLAY', latex_name='SLAY',
        color=COLORS['slay'], linestyle='-', linewidth=2.0,
        marker='s', is_our_method=True
    ),
    'performer': MethodStyle(
        name='FAVOR+', latex_name='FAVOR+',
        color=COLORS['performer'], linestyle=':', linewidth=1.5,
        marker='^'
    ),
    'linear': MethodStyle(
        name='Linear (ELU+1)', latex_name='Linear (ELU+1)',
        color=COLORS['linear'], linestyle='-', linewidth=1.2
    ),
    'cosformer': MethodStyle(
        name='Cosformer', latex_name='Cosformer',
        color=COLORS['cosformer'], linestyle='--', linewidth=1.2
    ),
}


def get_method_style(method_key: str) -> MethodStyle:
    """Get style for a method, with fallback defaults."""
    return METHOD_STYLES.get(method_key, MethodStyle(
        name=method_key, latex_name=method_key,
        color=COLORS.get(method_key, '#888888')
    ))


# ============================================================================
# PUBLICATION STYLE SETUP
# ============================================================================
def setup_icml_style():
    """
    Apply ICML publication-quality matplotlib settings.
    Call this at the start of any visualization script.
    """
    # Type 1 fonts for PDF (required by many venues)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    # Font settings
    mpl.rcParams['font.family'] = DS.FONT_FAMILY
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # Font sizes
    mpl.rcParams['font.size'] = DS.FONT_SIZE_BASE
    mpl.rcParams['axes.labelsize'] = DS.FONT_SIZE_LABEL
    mpl.rcParams['axes.titlesize'] = DS.FONT_SIZE_TITLE
    mpl.rcParams['legend.fontsize'] = DS.FONT_SIZE_LEGEND
    mpl.rcParams['xtick.labelsize'] = DS.FONT_SIZE_TICK
    mpl.rcParams['ytick.labelsize'] = DS.FONT_SIZE_TICK
    
    # Line widths
    mpl.rcParams['axes.linewidth'] = DS.LINE_WIDTH_AXIS
    mpl.rcParams['lines.linewidth'] = DS.LINE_WIDTH_NORMAL
    
    # Other settings
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['savefig.edgecolor'] = 'none'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight'


def create_figure(layout: str = 'single', **kwargs) -> Tuple[plt.Figure, any]:
    """
    Create a figure with preset dimensions.
    
    Args:
        layout: One of 'single', 'full', 'square', 'composite', 'heatmap'
        **kwargs: Additional arguments to plt.subplots or plt.figure
    
    Returns:
        (fig, ax) or (fig, axes) tuple
    """
    sizes = {
        'single': DS.SINGLE_PANEL_SIZE,
        'full': DS.FULL_WIDTH_SIZE,
        'square': DS.SQUARE_SIZE,
        'composite': DS.COMPOSITE_SIZE,
        'heatmap': DS.HEATMAP_SIZE,
    }
    
    figsize = sizes.get(layout, DS.FULL_WIDTH_SIZE)
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    fig.patch.set_facecolor('white')
    
    return fig, ax


def style_axis(ax, title: str = '', xlabel: str = '', ylabel: str = '',
               grid: bool = True, box: bool = True):
    """Apply consistent styling to an axis."""
    ax.set_facecolor('white')
    
    if title:
        ax.set_title(title, fontsize=DS.FONT_SIZE_TITLE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=DS.FONT_SIZE_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=DS.FONT_SIZE_LABEL)
    
    if grid:
        ax.grid(True, alpha=DS.GRID_ALPHA, linewidth=0.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(DS.LINE_WIDTH_AXIS)
        if not box:
            spine.set_visible(False)


def save_figure(fig, output_path: str, log_data_dict: Optional[Dict] = None, **log_kwargs):
    """
    Save figure to PDF with optional data logging.
    
    Args:
        fig: matplotlib figure
        output_path: Path to save PDF
        log_data_dict: Optional dict to log as companion data file
        **log_kwargs: Additional args for log_data (description, goal, etc.)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'assets', exist_ok=True)
    
    plt.savefig(output_path, format='pdf', dpi=300,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_path}")
    
    if log_data_dict:
        log_path = output_path.replace('.pdf', '_data.txt')
        log_data(log_path, log_data_dict, **log_kwargs)
        print(f"  ✓ Data log: {log_path}")


# ============================================================================
# KERNEL FUNCTIONS (Unified Implementations)
# ============================================================================
def softmax_kernel(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Softmax kernel: k(x) = exp(scale * x)"""
    return np.exp(scale * x)


def cosine_kernel(x: np.ndarray) -> np.ndarray:
    """Linear/cosine kernel: k(x) = x"""
    return x


def polynomial_kernel(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial kernel: k(x) = x^degree"""
    return x ** degree


def yat_kernel(x: np.ndarray, epsilon: float = 1e-2, 
               norm_q: float = 1.0, norm_k: float = 1.0) -> np.ndarray:
    """
    Pure YAT kernel (Euclidean):
    k(q, k) = (q·k)² / (||q-k||² + ε)
    
    For plotting vs cosine similarity x = q̂·k̂:
    q·k = x * ||q|| * ||k||
    ||q-k||² = ||q||² + ||k||² - 2·q·k
    """
    dot = x * norm_q * norm_k
    dist_sq = norm_q**2 + norm_k**2 - 2 * dot
    return (dot ** 2) / (dist_sq + epsilon)


def spherical_yat_kernel(x: np.ndarray, epsilon: float = 1e-2) -> np.ndarray:
    """
    Spherical YAT kernel (unit norm vectors):
    k(x) = x² / (C - 2x) where x = q̂·k̂, C = 2 + ε
    """
    C = 2.0 + epsilon
    return (x ** 2) / (C - 2 * x)


def relu_kernel(x: np.ndarray) -> np.ndarray:
    """ReLU kernel (FAVOR+ style): k(x) = max(0, x)²"""
    return np.maximum(0, x) ** 2


def elu_plus_one_kernel(x: np.ndarray) -> np.ndarray:
    """ELU+1 kernel (linear attention): k(x) = (elu(x) + 1)²"""
    elu_x = np.where(x > 0, x, np.exp(x) - 1)
    return (elu_x + 1) ** 2 / 4  # Normalized


# Registry for kernel functions
KERNEL_FUNCTIONS: Dict[str, Callable] = {
    'softmax': softmax_kernel,
    'cosine': cosine_kernel,
    'polynomial': polynomial_kernel,
    'yat': yat_kernel,
    'yat_exact': yat_kernel,
    'spherical_yat': spherical_yat_kernel,
    'yat_spherical': spherical_yat_kernel,
    'relu': relu_kernel,
    'elu_plus_one': elu_plus_one_kernel,
}


def get_kernel(name: str) -> Callable:
    """Get kernel function by name."""
    return KERNEL_FUNCTIONS.get(name, polynomial_kernel)


# ============================================================================
# QUADRATURE UTILITIES
# ============================================================================
def gauss_laguerre_nodes(R: int, epsilon: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Gauss-Laguerre quadrature nodes and weights for LAY.
    
    Args:
        R: Number of quadrature nodes
        epsilon: Regularization constant (determines C = 2 + epsilon)
    
    Returns:
        (s_vals, w_vals): Scaled nodes and weights
    """
    C = 2.0 + epsilon
    nodes, weights = np.polynomial.laguerre.laggauss(R)
    s_vals = nodes / C
    w_vals = weights / C
    return s_vals, w_vals


def quadrature_approximation(x: np.ndarray, R: int = 3, epsilon: float = 1e-2) -> np.ndarray:
    """
    Quadrature approximation of spherical YAT kernel.
    
    Args:
        x: Cosine similarity values
        R: Number of quadrature nodes
        epsilon: Regularization constant
    
    Returns:
        Approximated kernel values
    """
    s_vals, w_vals = gauss_laguerre_nodes(R, epsilon)
    
    result = np.zeros_like(x)
    for s, w in zip(s_vals, w_vals):
        result += w * (x ** 2) * np.exp(2 * s * x)
    
    return result


# ============================================================================
# PYTORCH ATTENTION KERNELS (for visualization with actual tensors)
# These compute attention weights given query and keys tensors
# ============================================================================
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def attention_softmax(query, keys, temperature=1.0):
    """
    Softmax attention: exp(q·k / T) / sum(exp(q·k / T))
    Exact, O(L²) complexity.
    """
    dots = torch.matmul(keys, query)
    scores = torch.exp(dots / temperature)
    return scores / scores.sum()


def attention_yat(query, keys, epsilon=1e-2):
    """
    Pure YAT kernel (exact, O(L²)):
    k(q, k) = (q·k)² / (||q-k||² + ε)
    
    This is the EXACT kernel. LAY approximates this.
    """
    dots = torch.matmul(keys, query)
    diffs = keys - query.unsqueeze(0)
    dist_sq = (diffs ** 2).sum(dim=-1)
    scores = (dots ** 2) / (dist_sq + epsilon)
    return scores / scores.sum()


def attention_spherical_yat(query, keys, epsilon=1e-2):
    """
    Spherical YAT kernel (exact, O(L²)):
    k(x) = x² / (C - 2x) where x = q̂·k̂, C = 2 + ε
    
    This is the EXACT spherical kernel. SLAY approximates this.
    """
    q_norm = F.normalize(query.unsqueeze(0), p=2, dim=-1).squeeze(0)
    k_norm = F.normalize(keys, p=2, dim=-1)
    
    dots = torch.matmul(k_norm, q_norm)
    C = 2.0 + epsilon
    scores = (dots ** 2) / (C - 2 * dots)
    scores = torch.clamp(scores, min=0)
    return scores / scores.sum()


def attention_slay(query, keys, num_anchors=32, num_prf=16, num_nodes=3, epsilon=1e-2):
    """
    SLAY: Linear approximation of spherical YAT (O(L) complexity).
    Uses anchor features for polynomial, PRF for exponential.
    """
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
        
        poly_score = torch.matmul(k_poly, q_poly)
        prf_score = torch.matmul(k_prf, q_prf)
        
        total_score += w_r * poly_score * prf_score
    
    total_score = torch.clamp(total_score, min=0)
    return total_score / total_score.sum()


def attention_lay(query, keys, num_rff=64, num_anchors=32, epsilon=1e-2):
    """
    LAY: Linear approximation of pure YAT kernel (O(L) complexity).
    Uses Random Fourier Features (RFF) for the RBF/Laplace component
    and anchor features for the polynomial component.
    
    YAT: k(q,k) = (q·k)² / (||q-k||² + ε)
    LAY approximates the 1/(||q-k||² + ε) via Laplace transform → RBF → RFF
    """
    d = query.shape[-1]
    
    # Anchor features for (q·k)² polynomial component
    anchors = torch.randn(num_anchors, d)
    anchors = F.normalize(anchors, p=2, dim=-1)
    
    q_poly = (torch.matmul(query.unsqueeze(0), anchors.T) ** 2).squeeze(0) / np.sqrt(num_anchors)
    k_poly = (torch.matmul(keys, anchors.T) ** 2) / np.sqrt(num_anchors)
    
    # RFF for the RBF/Laplace component (approximating 1/(d² + ε))
    # Using Gauss-Laguerre quadrature to handle the Laplace transform
    num_nodes = 3
    nodes, weights = np.polynomial.laguerre.laggauss(num_nodes)
    nodes = torch.tensor(nodes, dtype=torch.float32) / epsilon
    weights = torch.tensor(weights, dtype=torch.float32) / epsilon
    
    total_score = torch.zeros(keys.shape[0])
    
    for r in range(num_nodes):
        s_r = nodes[r].item()
        w_r = weights[r].item()
        
        # RFF: exp(-s||q-k||²) ≈ E[cos(ω·q)cos(ω·k) + sin(ω·q)sin(ω·k)]
        omega = torch.randn(d, num_rff) * np.sqrt(2 * s_r)
        
        q_proj = torch.matmul(query.unsqueeze(0), omega).squeeze(0)
        k_proj = torch.matmul(keys, omega)
        
        q_cos = torch.cos(q_proj) / np.sqrt(num_rff)
        q_sin = torch.sin(q_proj) / np.sqrt(num_rff)
        k_cos = torch.cos(k_proj) / np.sqrt(num_rff)
        k_sin = torch.sin(k_proj) / np.sqrt(num_rff)
        
        rff_score = torch.matmul(k_cos, q_cos) + torch.matmul(k_sin, q_sin)
        poly_score = torch.matmul(k_poly, q_poly)
        
        total_score += w_r * poly_score * rff_score
    
    total_score = torch.clamp(total_score, min=0)
    return total_score / (total_score.sum() + 1e-8)


def attention_performer(query, keys, num_features=64):
    """
    FAVOR+ (Performer) attention approximation.
    Uses positive random features.
    """
    d = query.shape[-1]
    omega = torch.randn(d, num_features)
    
    def pos_features(x):
        proj = torch.matmul(x, omega) / np.sqrt(d)
        return torch.exp(proj - proj.max(dim=-1, keepdim=True)[0])
    
    q_feat = pos_features(query.unsqueeze(0)).squeeze(0)
    k_feat = pos_features(keys)
    
    scores = torch.matmul(k_feat, q_feat)
    return scores / scores.sum()


# Attention kernel registry
ATTENTION_KERNELS = {
    'softmax': attention_softmax,
    'yat': attention_yat,
    'yat_exact': attention_yat,
    'spherical_yat': attention_spherical_yat,
    'yat_spherical': attention_spherical_yat,
    'slay': attention_slay,
    'lay': attention_lay,
    'performer': attention_performer,
}


def get_attention_kernel(name: str):
    """Get attention kernel function by name."""
    return ATTENTION_KERNELS.get(name, attention_softmax)


# ============================================================================
# DEFAULT KERNELS FOR VISUALIZATION
# ============================================================================
# Standard set of kernels to test and visualize consistently across all figures
DEFAULT_KERNELS = [
    {
        'name': 'Softmax',
        'key': 'softmax',
        'fn': attention_softmax,
        'color': COLORS['softmax'],
        'linestyle': '-',
        'is_exact': True,
        'is_ours': False,
    },
    {
        'name': r'$\mathcal{E}$ (YAT)',
        'key': 'yat',
        'fn': attention_yat,
        'color': COLORS['yat_exact'],
        'linestyle': '--',
        'is_exact': True,
        'is_ours': False,
    },
    {
        'name': 'LAY',
        'key': 'lay',
        'fn': attention_lay,
        'color': COLORS['lay'],
        'linestyle': '-',
        'is_exact': False,
        'is_ours': True,
    },
    {
        'name': r'$\mathcal{E}_{sph}$',
        'key': 'spherical_yat',
        'fn': attention_spherical_yat,
        'color': COLORS['yat_spherical'],
        'linestyle': '--',
        'is_exact': True,
        'is_ours': False,
    },
    {
        'name': 'SLAY',
        'key': 'slay',
        'fn': attention_slay,
        'color': COLORS['slay'],
        'linestyle': '-',
        'is_exact': False,
        'is_ours': True,
    },
    {
        'name': 'Performer',
        'key': 'performer',
        'fn': attention_performer,
        'color': COLORS['performer'],
        'linestyle': ':',
        'is_exact': False,
        'is_ours': False,
    },
]


def get_default_kernels(include_baselines=True, include_exact=True):
    """
    Get default kernels for visualization.
    
    Args:
        include_baselines: Include non-YAT baselines (Softmax, Performer)
        include_exact: Include exact O(L²) kernels (YAT, Spherical YAT)
    
    Returns:
        List of kernel configs
    """
    kernels = []
    for k in DEFAULT_KERNELS:
        if not include_baselines and k['key'] in ['softmax', 'performer']:
            continue
        if not include_exact and k['is_exact'] and not k['is_ours']:
            continue
        kernels.append(k)
    return kernels


# ============================================================================
# DATA LOGGING
# ============================================================================
def log_data(filename, data_dict, description="", goal="", what_to_look_for="", expected_conclusion=""):
    """
    Log plot data to a text file for LLM analysis.
    
    Args:
        filename: Output txt file path
        data_dict: Dictionary of arrays/values to log
        description: Brief description of the data
        goal: The goal/purpose of this visualization
        what_to_look_for: What patterns/features to analyze
        expected_conclusion: The conclusion that should be drawn
    
    Returns:
        Path to the created log file
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else 'assets', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# LAY Visualization Data Log\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"#" + "="*60 + "\n\n")
        
        # Analysis context section
        if goal or what_to_look_for or expected_conclusion:
            f.write("## ANALYSIS CONTEXT\n\n")
            if goal:
                f.write(f"### Goal\n{goal}\n\n")
            if what_to_look_for:
                f.write(f"### What to Look For\n{what_to_look_for}\n\n")
            if expected_conclusion:
                f.write(f"### Expected Conclusion\n{expected_conclusion}\n\n")
            f.write("#" + "-"*60 + "\n\n")
        
        if description:
            f.write(f"## Description\n{description}\n\n")
        
        f.write("## DATA\n\n")
        
        for key, value in data_dict.items():
            f.write(f"### {key}\n")
            if isinstance(value, np.ndarray):
                f.write(f"# Shape: {value.shape}, dtype: {value.dtype}\n")
                if value.ndim == 1 and len(value) <= 100:
                    f.write(f"# Values: {value.tolist()}\n")
                elif value.ndim == 1:
                    f.write(f"# First 20: {value[:20].tolist()}\n")
                    f.write(f"# Last 20: {value[-20:].tolist()}\n")
                    f.write(f"# Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}, Std: {value.std():.6f}\n")
                else:
                    f.write(f"# Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}\n")
            elif hasattr(value, 'numpy'):  # torch tensor
                arr = value.detach().cpu().numpy()
                f.write(f"# Shape: {arr.shape}\n")
                f.write(f"# Min: {arr.min():.6f}, Max: {arr.max():.6f}, Mean: {arr.mean():.6f}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"# Length: {len(value)}\n")
                if len(value) > 50:
                    f.write(f"# Values (first 50): {list(value[:50])}...\n")
                else:
                    f.write(f"# Values: {list(value)}\n")
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, np.ndarray):
                        f.write(f"#   {sub_key}: shape={sub_val.shape}, min={sub_val.min():.4f}, max={sub_val.max():.4f}\n")
                    elif hasattr(sub_val, 'numpy'):
                        arr = sub_val.detach().cpu().numpy()
                        f.write(f"#   {sub_key}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}\n")
                    else:
                        f.write(f"#   {sub_key}: {sub_val}\n")
            else:
                f.write(f"# Value: {value}\n")
            f.write("\n")
    
    return filename


def log_csv_data(filename, columns_dict, description=""):
    """
    Log plot data as CSV for easy import.
    
    Args:
        filename: Output csv file path
        columns_dict: Dictionary where keys are column names, values are arrays
        description: Optional description
    
    Returns:
        Path to created CSV file
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else 'assets', exist_ok=True)
    
    # Convert to numpy arrays
    arrays = {}
    for key, value in columns_dict.items():
        if hasattr(value, 'numpy'):
            arrays[key] = value.detach().cpu().numpy().flatten()
        elif isinstance(value, np.ndarray):
            arrays[key] = value.flatten()
        else:
            arrays[key] = np.array(value).flatten()
    
    # Check lengths match
    lengths = [len(v) for v in arrays.values()]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        arrays = {k: v[:min_len] for k, v in arrays.items()}
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {description}\n")
        f.write(",".join(arrays.keys()) + "\n")
        for i in range(len(list(arrays.values())[0])):
            row = [f"{arrays[k][i]:.8g}" for k in arrays.keys()]
            f.write(",".join(row) + "\n")
    
    return filename


# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================
# For backward compatibility, export common items at module level
__all__ = [
    # Design system
    'COLORS', 'DS', 'DesignSystem', 'METHOD_STYLES', 'MethodStyle',
    'setup_icml_style', 'create_figure', 'style_axis', 'save_figure',
    'get_method_style',
    
    # Kernel functions
    'softmax_kernel', 'cosine_kernel', 'polynomial_kernel',
    'yat_kernel', 'spherical_yat_kernel', 'relu_kernel', 'elu_plus_one_kernel',
    'KERNEL_FUNCTIONS', 'get_kernel',
    
    # Quadrature utilities
    'gauss_laguerre_nodes', 'quadrature_approximation',
    
    # Logging
    'log_data', 'log_csv_data',
]


# ============================================================================
# QUICK DEMO
# ============================================================================
if __name__ == "__main__":
    print("LAY Visualization Design System")
    print("=" * 40)
    print(f"\n📐 Figure Dimensions:")
    print(f"   Single column: {DS.COLUMN_WIDTH}\" width")
    print(f"   Full width: {DS.FULL_WIDTH}\" width")
    
    print(f"\n🎨 Color Palette ({len(COLORS)} colors):")
    for name, color in list(COLORS.items())[:10]:
        print(f"   {name}: {color}")
    
    print(f"\n📊 Method Styles ({len(METHOD_STYLES)} methods):")
    for key, style in METHOD_STYLES.items():
        marker = f" [{style.marker}]" if style.marker else ""
        print(f"   {style.name}: {style.color} ({style.linestyle}){marker}")
    
    print(f"\n🧮 Kernel Functions ({len(KERNEL_FUNCTIONS)} kernels):")
    for name in KERNEL_FUNCTIONS:
        print(f"   - {name}")
    
    print("\n✅ Design system ready. Import with:")
    print("   from viz_utils import COLORS, setup_icml_style, DS")
