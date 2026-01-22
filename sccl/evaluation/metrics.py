"""Metrics for evaluating single cell classification."""

import logging
from typing import Optional, List, Dict
import numpy as np
from anndata import AnnData
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    adata: Optional[AnnData] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics for classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    adata : AnnData, optional
        Data object (needed for some metrics like silhouette score)
    metrics : list of str, optional
        Metrics to compute. If None, computes all available metrics.
        Options: 'accuracy', 'ari', 'nmi', 'f1', 'precision', 'recall',
                'silhouette'

    Returns
    -------
    results : dict
        Dictionary mapping metric names to values

    Examples
    --------
    >>> results = compute_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {results['accuracy']:.3f}")
    >>> print(f"ARI: {results['ari']:.3f}")
    """
    if metrics is None:
        metrics = ['accuracy', 'ari', 'nmi', 'f1', 'precision', 'recall']

    results = {}

    # Convert to string for sklearn compatibility
    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)

    # Accuracy
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true_str, y_pred_str)

    # Adjusted Rand Index (clustering metric)
    if 'ari' in metrics:
        results['ari'] = adjusted_rand_score(y_true_str, y_pred_str)

    # Normalized Mutual Information (clustering metric)
    if 'nmi' in metrics:
        results['nmi'] = normalized_mutual_info_score(y_true_str, y_pred_str)

    # F1 score
    if 'f1' in metrics:
        results['f1_macro'] = f1_score(y_true_str, y_pred_str, average='macro')
        results['f1_weighted'] = f1_score(y_true_str, y_pred_str, average='weighted')

    # Precision
    if 'precision' in metrics:
        results['precision_macro'] = precision_score(
            y_true_str, y_pred_str, average='macro', zero_division=0
        )

    # Recall
    if 'recall' in metrics:
        results['recall_macro'] = recall_score(
            y_true_str, y_pred_str, average='macro', zero_division=0
        )

    # Silhouette score (requires embedding)
    if 'silhouette' in metrics and adata is not None:
        if 'X_pca' in adata.obsm:
            X = adata.obsm['X_pca']
        elif 'X_umap' in adata.obsm:
            X = adata.obsm['X_umap']
        else:
            X = adata.X

        try:
            # Subsample if too large (silhouette is slow)
            if X.shape[0] > 10000:
                idx = np.random.choice(X.shape[0], size=10000, replace=False)
                X_sub = X[idx]
                y_sub = y_true_str[idx]
            else:
                X_sub = X
                y_sub = y_true_str

            # Convert to dense if sparse
            if hasattr(X_sub, 'toarray'):
                X_sub = X_sub.toarray()

            results['silhouette'] = silhouette_score(X_sub, y_sub)
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")

    return results


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    results : dict
        Dictionary mapping class names to their metrics

    Examples
    --------
    >>> per_class = compute_per_class_metrics(y_true, y_pred)
    >>> print(per_class['T_cell']['f1'])
    """
    from sklearn.metrics import classification_report

    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)

    report = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)

    # Extract per-class metrics (exclude averages and support)
    per_class = {}
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            per_class[class_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1-score'],
                'support': metrics['support'],
            }

    return per_class


def compute_confusion_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, any]:
    """Compute confusion matrix and related statistics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    stats : dict
        Dictionary with:
        - 'confusion_matrix': Confusion matrix
        - 'labels': Class labels
        - 'misclassification_rate': Overall misclassification rate
        - 'per_class_error': Error rate per class

    Examples
    --------
    >>> stats = compute_confusion_stats(y_true, y_pred)
    >>> print(stats['confusion_matrix'])
    """
    from sklearn.metrics import confusion_matrix

    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)

    labels = np.unique(np.concatenate([y_true_str, y_pred_str]))
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)

    # Compute per-class error rates
    per_class_error = {}
    for i, label in enumerate(labels):
        total = cm[i, :].sum()
        if total > 0:
            correct = cm[i, i]
            error_rate = 1 - (correct / total)
            per_class_error[label] = error_rate
        else:
            per_class_error[label] = np.nan

    # Overall misclassification rate
    total_correct = np.diag(cm).sum()
    total = cm.sum()
    misclassification_rate = 1 - (total_correct / total) if total > 0 else np.nan

    return {
        'confusion_matrix': cm,
        'labels': labels,
        'misclassification_rate': misclassification_rate,
        'per_class_error': per_class_error,
    }
