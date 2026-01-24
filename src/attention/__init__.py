"""
Attention mechanisms registry and implementations.

This subpackage provides various causal attention implementations:
- standard: Standard softmax attention
- linear: ELU+1 linear attention
- cosformer: Cosformer with cos-based reweighting
- rff: Random Fourier Features attention
- yat: Yat-product attention (exact)
- yat-performer: Linearized Yat attention (fast)
- yat-performer-tensor: Linearized Yat attention (TensorSketch)
- yat-spherical: Spherical Yat attention (exact)
- performer: FAVOR+ linear attention
"""

from .standard import StandardCausalAttention
from .linear import LinearCausalAttention
from .cosformer import CosformerCausalAttention
from .rff import RFFCausalAttention
from .yat import YatCausalAttention
from .yat_performer import YatPerformerCausalAttention
from .yat_performer_tensor import YatPerformerTensorCausalAttention
from .yat_performer_laplace import YatPerformerLaplaceCausalAttention
from .yat_performer_poly_alt import (
    YatPerformerRMCausalAttention,
    YatPerformerNystromCausalAttention,
    YatPerformerAnchorCausalAttention,
)
from .yat_spherical import YatSphericalCausalAttention
from .performer import FastAttention

# Registry of all attention types
ATTENTION_CLASSES = {
    'performer': FastAttention,           # Original FAVOR+ (ReLU)
    'standard': StandardCausalAttention,  # Standard softmax attention
    'linear': LinearCausalAttention,      # ELU+1 linear attention
    'cosformer': CosformerCausalAttention,  # Cosformer
    'rff': RFFCausalAttention,            # Random Fourier Features
    'yat': YatCausalAttention,            # Yat-product (exact)
    'yat-performer': YatPerformerCausalAttention,  # Linearized Yat
    'yat-performer-tensor': YatPerformerTensorCausalAttention,  # TensorSketch Yat
    'yat-performer-laplace': YatPerformerLaplaceCausalAttention,  # Laplace-only PRF
    'yat-performer-rm': YatPerformerRMCausalAttention,  # Random Maclaurin
    'yat-performer-nystrom': YatPerformerNystromCausalAttention,  # Nyström
    'yat-performer-anchors': YatPerformerAnchorCausalAttention,  # Anchor features
    'yat-spherical': YatSphericalCausalAttention,  # Exact Spherical Yat

    # Aliases (paper naming / convenience)
    'yat-performer-hadamard': YatPerformerCausalAttention,  # Hadamard fusion (shared ω)
    'yat-performer-anchor': YatPerformerAnchorCausalAttention,
    'yat-performer-laplace-only': YatPerformerLaplaceCausalAttention,
    'yat-performer-tensorsketch': YatPerformerTensorCausalAttention,
    'yat-performer-random-maclaurin': YatPerformerRMCausalAttention,
    'yat-exact-spherical': YatSphericalCausalAttention,
}

# Attention types that use novel activation in the MLP
NOVEL_ACTIVATION_TYPES = {
    'yat',
    'yat-performer',
    'yat-performer-tensor',
    'yat-performer-laplace',
    'yat-performer-rm',
    'yat-performer-nystrom',
    'yat-performer-anchors',
    'yat-spherical',

    # Aliases
    'yat-performer-hadamard',
    'yat-performer-anchor',
    'yat-performer-laplace-only',
    'yat-performer-tensorsketch',
    'yat-performer-random-maclaurin',
    'yat-exact-spherical',
}


def get_attention_class(name: str):
    """
    Get attention class by name.
    
    Args:
        name: Name of attention type (e.g., 'standard', 'performer', 'yat')
        
    Returns:
        Attention class
        
    Raises:
        ValueError: If attention type not found
    """
    if name not in ATTENTION_CLASSES:
        available = list(ATTENTION_CLASSES.keys())
        raise ValueError(f"Unknown attention type '{name}'. Available: {available}")
    return ATTENTION_CLASSES[name]


def register_attention(name: str, cls):
    """
    Register a new attention class.
    
    Args:
        name: Name to register the attention under
        cls: Attention class (must be nn.Module subclass)
    """
    ATTENTION_CLASSES[name] = cls


def list_attention_types():
    """
    List all available attention types.
    
    Returns:
        List of attention type names
    """
    return list(ATTENTION_CLASSES.keys())


__all__ = [
    'ATTENTION_CLASSES',
    'NOVEL_ACTIVATION_TYPES',
    'get_attention_class',
    'register_attention',
    'list_attention_types',
    # Individual classes
    'StandardCausalAttention',
    'LinearCausalAttention',
    'CosformerCausalAttention',
    'RFFCausalAttention',
    'YatCausalAttention',
    'YatPerformerCausalAttention',
    'YatPerformerTensorCausalAttention',
    'YatPerformerLaplaceCausalAttention',
    'YatPerformerRMCausalAttention',
    'YatPerformerNystromCausalAttention',
    'YatPerformerAnchorCausalAttention',
    'YatSphericalCausalAttention',
    'FastAttention',
]
