"""
Slay: Scalable Linear Attention for Your models.

This package provides modular attention mechanisms and model components.
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG
from .activations import novel_activation
from .attention import (
    ATTENTION_CLASSES,
    NOVEL_ACTIVATION_TYPES,
    get_attention_class,
    register_attention,
    list_attention_types,
)

# NOTE: We intentionally avoid importing data/model/training utilities at
# package-import time. Those modules can depend on heavy optional libraries
# (e.g., transformers) and are not required for attention-only usage.
# They are exposed via lazy imports in __getattr__.

__all__ = [
    # Config
    'DEFAULT_CONFIG',
    # Activations
    'novel_activation',
    # Attention
    'ATTENTION_CLASSES',
    'NOVEL_ACTIVATION_TYPES',
    'get_attention_class',
    'register_attention',
    'list_attention_types',
    # Data
    'FineWebStream',
    'get_eval_loader',
    # Models
    'GPT2Block',
    'NovelBlock',
    'TinyGPT',
    # Utils
    'evaluate',
    'log_metrics',
    'LossPlateauDetector',
]


def __getattr__(name: str):
    if name in {'FineWebStream', 'get_eval_loader'}:
        from .data import FineWebStream, get_eval_loader

        return {'FineWebStream': FineWebStream, 'get_eval_loader': get_eval_loader}[name]

    if name in {'GPT2Block', 'NovelBlock', 'TinyGPT'}:
        from .models import GPT2Block, NovelBlock, TinyGPT

        return {'GPT2Block': GPT2Block, 'NovelBlock': NovelBlock, 'TinyGPT': TinyGPT}[name]

    if name in {'evaluate', 'log_metrics', 'LossPlateauDetector'}:
        from .utils import evaluate, log_metrics, LossPlateauDetector

        return {'evaluate': evaluate, 'log_metrics': log_metrics, 'LossPlateauDetector': LossPlateauDetector}[name]

    raise AttributeError(f"module 'src' has no attribute {name!r}")
