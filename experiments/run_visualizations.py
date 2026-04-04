"""
Run All SLAY Visualizations - ICML Paper Figures

This script runs all visualization scripts to generate publication-quality
figures for the SLAY paper. All outputs are saved to the assets/ directory.

Usage:
    python run_visualizations.py
    python run_visualizations.py --only kernel_angle  # Run specific visualization
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_kernel_angle():
    """Run kernel vs angle visualization."""
    print("\n" + "=" * 60)
    print(" 1. KERNEL VS ANGLE VISUALIZATION")
    print("=" * 60)
    
    from experiments.visualize_kernel_angle import main as kernel_main
    kernel_main()


def run_spherical_heatmap():
    """Run spherical heatmap visualization."""
    print("\n" + "=" * 60)
    print(" 2. SPHERICAL HEATMAP VISUALIZATION")
    print("=" * 60)
    
    from experiments.visualize_spherical_heatmap import main as heatmap_main
    heatmap_main()


def run_approximation():
    """Run approximation faithfulness visualization."""
    print("\n" + "=" * 60)
    print(" 3. APPROXIMATION FAITHFULNESS VISUALIZATION")
    print("=" * 60)
    
    from experiments.visualize_approximation import main as approx_main
    approx_main()


def run_denominator():
    """Run denominator positivity visualization."""
    print("\n" + "=" * 60)
    print(" 4. DENOMINATOR POSITIVITY VISUALIZATION")
    print("=" * 60)
    
    from experiments.visualize_denominator import main as denom_main
    denom_main()


def run_attention_entropy():
    """Run attention entropy visualization."""
    print("\n" + "=" * 60)
    print(" 5. ATTENTION ENTROPY VISUALIZATION")
    print("=" * 60)
    
    from experiments.visualize_attention_entropy import main as entropy_main
    entropy_main()


def run_quadrature():
    """Run quadrature analysis visualization."""
    print("\n" + "=" * 60)
    print(" 6. QUADRATURE ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    from experiments.visualize_quadrature import main as quad_main
    quad_main()


def run_composite():
    """Run composite story figure visualization."""
    print("\n" + "=" * 60)
    print(" 7. COMPOSITE STORY FIGURE")
    print("=" * 60)
    
    from experiments.visualize_composite import main as composite_main
    composite_main()


def list_generated_figures():
    """List all generated PDF figures in assets directory."""
    assets_dir = Path(__file__).parent / 'assets'
    
    if not assets_dir.exists():
        print("\nNo assets directory found.")
        return
    
    pdfs = list(assets_dir.glob('*.pdf'))
    
    print("\n" + "=" * 60)
    print(" GENERATED FIGURES")
    print("=" * 60)
    
    if not pdfs:
        print("No PDF files found in assets/")
        return
    
    total_size = 0
    for pdf in sorted(pdfs):
        size_kb = pdf.stat().st_size / 1024
        total_size += size_kb
        print(f"  • {pdf.name:<35} ({size_kb:.1f} KB)")
    
    print("-" * 60)
    print(f"  Total: {len(pdfs)} files, {total_size:.1f} KB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run SLAY visualization scripts')
    parser.add_argument('--only', type=str, default=None,
                        choices=['kernel_angle', 'spherical_heatmap', 'approximation',
                                 'denominator', 'attention_entropy', 'quadrature',
                                 'composite', 'neurons'],
                        help='Run only a specific visualization')
    parser.add_argument('--list', action='store_true',
                        help='List generated figures and exit')
    
    args = parser.parse_args()
    
    if args.list:
        list_generated_figures()
        return
    
    # Change to experiments directory
    os.chdir(Path(__file__).parent)
    
    print("\n" + "#" * 60)
    print("#")
    print("#  SLAY VISUALIZATION SUITE")
    print("#  Generating ICML publication-quality figures")
    print("#")
    print("#" * 60)
    
    start_time = time.time()
    
    # Run selected visualizations
    # 1. Kernel Angle
    if args.only is None or 'kernel_angle' in args.only:
        try:
            run_kernel_angle()
        except Exception as e:
            print(f"\n  [ERROR] Error in kernel_angle: {e}")
            import traceback
            traceback.print_exc()
            
    # 2. Spherical Heatmap
    if args.only is None or 'spherical_heatmap' in args.only:
        try:
            run_spherical_heatmap()
        except Exception as e:
            print(f"\n  [ERROR] Error in spherical_heatmap: {e}")
            import traceback
            traceback.print_exc()

    # 3. Approximation Quality
    if args.only is None or 'approximation' in args.only:
        try:
            run_approximation()
        except Exception as e:
            print(f"\n  [ERROR] Error in approximation: {e}")
            import traceback
            traceback.print_exc()

    # 4. Denominator Stability
    if args.only is None or 'denominator' in args.only:
        try:
            run_denominator()
        except Exception as e:
            print(f"\n  [ERROR] Error in denominator: {e}")
            import traceback
            traceback.print_exc()

    # 5. Attention Entropy
    if args.only is None or 'attention_entropy' in args.only: # Changed 'entropy' to 'attention_entropy' to match choices
        try:
            run_attention_entropy()
        except Exception as e:
            print(f"\n  [ERROR] Error in attention_entropy: {e}") # Changed 'entropy' to 'attention_entropy'
            import traceback
            traceback.print_exc()

    # 6. Quadrature Analysis
    if args.only is None or 'quadrature' in args.only:
        try:
            run_quadrature()
        except Exception as e:
            print(f"\n  [ERROR] Error in quadrature: {e}")
            import traceback
            traceback.print_exc()

    # 7. Composite Figure
    if args.only is None or 'composite' in args.only:
        try:
            run_composite()
        except Exception as e:
            print(f"\n  [ERROR] Error in composite: {e}")
            import traceback
            traceback.print_exc()
    
    # 8. Neurons (New!)
    if args.only is None or 'neurons' in args.only:
        try:
            print("\n" + "=" * 60)
            print(" 8. NEURON BOUNDARIES VISUALIZATION")
            print("=" * 60)
            from experiments.visualize_neurons import main as neurons_main
            neurons_main()
        except Exception as e:
            print(f"\n  [ERROR] Error in neurons: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    
    # List generated figures
    list_generated_figures()
    
    print("\n" + "=" * 60)
    print(f"[DONE] All visualizations completed in {elapsed:.1f} seconds")
    print("=" * 60)
    print("\n" + "=" * 60)
    print(" RECOMMENDED FIGURES FOR PAPER:")
    print("=" * 60)
    print("\n  Main Paper:")
    print("    * assets/slay_overview.pdf     - Story figure (Fig 1)")
    print("    * assets/kernel_comparison.pdf - Kernel comparison")
    print("    * assets/attention_patterns.pdf - Attention behavior")
    print("\n  Appendix:")
    print("    • assets/approximation_quality.pdf - Approximation faithfulness")
    print("    • assets/denominator_histogram.pdf - Stability analysis")
    print("    • assets/quadrature_convergence.pdf - Quadrature justification")
    print("    • assets/scaling_with_exact.pdf - Scaling with exact overlay")


if __name__ == "__main__":
    main()
