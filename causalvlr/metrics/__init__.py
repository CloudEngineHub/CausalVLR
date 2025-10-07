"""
Metrics Module for CausalVLR

This module provides evaluation metrics for both MRG and VQA tasks.

Available Metrics:
    - MRG: BLEU, METEOR, ROUGE, CIDEr
    - VQA: Accuracy, type-specific metrics

Submodules:
    - mrg_metric: Medical report generation metrics (BLEU, METEOR, ROUGE, CIDEr)
"""

# Import MRG metrics submodule
from . import mrg_metric

# For convenience, expose compute_scores from mrg_metric if available
try:
    from .mrg_metric import compute_scores as compute_mrg_scores
except ImportError:
    compute_mrg_scores = None

__all__ = [
    "mrg_metric",
]

if compute_mrg_scores is not None:
    __all__.append("compute_mrg_scores")
