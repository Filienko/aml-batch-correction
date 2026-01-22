"""Evaluation utilities for model assessment."""

from .metrics import compute_metrics, compute_per_class_metrics, compute_confusion_stats
from .visualization import plot_umap, plot_confusion_matrix, plot_comparison

__all__ = [
    'compute_metrics',
    'compute_per_class_metrics',
    'compute_confusion_stats',
    'plot_umap',
    'plot_confusion_matrix',
    'plot_comparison',
]
