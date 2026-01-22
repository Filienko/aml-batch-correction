"""Visualization utilities for single cell analysis."""

import logging
from typing import Optional, List, Union
import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

logger = logging.getLogger(__name__)


def plot_umap(
    adata: AnnData,
    color: Union[str, List[str]],
    save: Optional[str] = None,
    **kwargs
) -> None:
    """Plot UMAP visualization.

    Parameters
    ----------
    adata : AnnData
        Data with UMAP embedding in .obsm['X_umap']
    color : str or list of str
        Variable(s) to color by
    save : str, optional
        Path to save figure
    **kwargs
        Additional arguments passed to sc.pl.umap

    Examples
    --------
    >>> plot_umap(adata, color='cell_type')
    >>> plot_umap(adata, color=['cell_type', 'batch'], save='umap.pdf')
    """
    # Compute UMAP if not present
    if 'X_umap' not in adata.obsm:
        logger.info("Computing UMAP...")
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    # Plot
    sc.pl.umap(adata, color=color, save=save, **kwargs)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = True,
    save: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    labels : list of str, optional
        Class labels (for ordering)
    normalize : bool, default=True
        Whether to normalize by row (true class counts)
    save : str, optional
        Path to save figure
    figsize : tuple, default=(10, 8)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> fig = plot_confusion_matrix(y_true, y_pred)
    >>> fig.savefig('confusion_matrix.pdf')
    """
    from sklearn.metrics import confusion_matrix

    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)

    if labels is None:
        labels = np.unique(np.concatenate([y_true_str, y_pred_str]))

    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = None

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=vmax,
        ax=ax,
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save}")

    return fig


def plot_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    save: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot model comparison as bar charts.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with models as index and metrics as columns
    metrics : list of str, optional
        Metrics to plot (if None, plots all)
    save : str, optional
        Path to save figure
    figsize : tuple, default=(12, 6)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> fig = plot_comparison(comparison_df, metrics=['accuracy', 'ari', 'nmi'])
    """
    if metrics is None:
        metrics = comparison_df.columns.tolist()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        if metric not in comparison_df.columns:
            logger.warning(f"Metric '{metric}' not in comparison DataFrame")
            continue

        ax = axes[i]

        # Plot bars
        comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue')

        ax.set_title(metric.upper())
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save}")

    return fig


def plot_embedding_comparison(
    adata: AnnData,
    embeddings: dict,
    color: str,
    ncols: int = 3,
    save: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple embeddings side by side.

    Parameters
    ----------
    adata : AnnData
        Data object
    embeddings : dict
        Dictionary mapping names to embedding matrices
    color : str
        Variable to color by
    ncols : int, default=3
        Number of columns in subplot grid
    save : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> embeddings = {
    ...     'PCA': adata.obsm['X_pca'],
    ...     'SCimilarity': scimilarity_embedding,
    ...     'scVI': scvi_embedding,
    ... }
    >>> fig = plot_embedding_comparison(adata, embeddings, color='cell_type')
    """
    n_embeddings = len(embeddings)
    nrows = int(np.ceil(n_embeddings / ncols))

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for i, (name, embedding) in enumerate(embeddings.items(), 1):
        # Compute UMAP of this embedding
        adata_tmp = adata.copy()
        adata_tmp.obsm['X_temp'] = embedding

        sc.pp.neighbors(adata_tmp, use_rep='X_temp')
        sc.tl.umap(adata_tmp)

        # Plot
        ax = fig.add_subplot(nrows, ncols, i)
        sc.pl.umap(adata_tmp, color=color, ax=ax, show=False, title=name)

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        logger.info(f"Saved embedding comparison to {save}")

    return fig


def plot_batch_effect(
    adata: AnnData,
    batch_key: str,
    label_key: str,
    save: Optional[str] = None,
) -> plt.Figure:
    """Plot UMAP colored by batch and cell type to visualize batch effects.

    Parameters
    ----------
    adata : AnnData
        Data with UMAP embedding
    batch_key : str
        Column for batch information
    label_key : str
        Column for cell type labels
    save : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> fig = plot_batch_effect(adata, batch_key='study', label_key='cell_type')
    """
    # Compute UMAP if needed
    if 'X_umap' not in adata.obsm:
        logger.info("Computing UMAP...")
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot by batch
    sc.pl.umap(adata, color=batch_key, ax=axes[0], show=False, title='Colored by Batch')

    # Plot by cell type
    sc.pl.umap(adata, color=label_key, ax=axes[1], show=False, title='Colored by Cell Type')

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        logger.info(f"Saved batch effect plot to {save}")

    return fig
