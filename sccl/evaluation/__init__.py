"""Evaluation utilities for model assessment."""

from .metrics import compute_metrics
from .visualization import plot_umap, plot_confusion_matrix, plot_comparison

__all__ = ['compute_metrics', 'plot_umap', 'plot_confusion_matrix', 'plot_comparison']
