"""Data loading utilities."""

import logging
from pathlib import Path
from typing import Union, Optional
from anndata import AnnData
import scanpy as sc

logger = logging.getLogger(__name__)


def load_data(
    path: Union[str, Path],
    backed: Optional[str] = None,
) -> AnnData:
    """Load single cell data from file.

    Supports .h5ad, .loom, .csv, .mtx formats.

    Parameters
    ----------
    path : str or Path
        Path to data file
    backed : str, optional
        If 'r', load in backed mode (memory efficient for large files)

    Returns
    -------
    adata : AnnData
        Loaded annotated data matrix

    Examples
    --------
    >>> adata = load_data("data/pbmc.h5ad")
    >>> adata = load_data("data/large_dataset.h5ad", backed='r')
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading data from {path}")

    # Load based on file extension
    suffix = path.suffix.lower()

    if suffix == '.h5ad':
        adata = sc.read_h5ad(path, backed=backed)
    elif suffix == '.loom':
        adata = sc.read_loom(path)
    elif suffix == '.csv':
        adata = sc.read_csv(path)
    elif suffix == '.mtx':
        adata = sc.read_mtx(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: .h5ad, .loom, .csv, .mtx"
        )

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    return adata


def detect_batch_key(adata: AnnData) -> Optional[str]:
    """Auto-detect batch/study column in metadata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix

    Returns
    -------
    batch_key : str or None
        Name of batch column, or None if not found
    """
    # Common batch column names
    candidates = [
        'batch', 'study', 'dataset', 'sample', 'donor',
        'patient', 'replicate', 'run', 'experiment'
    ]

    for col in candidates:
        if col in adata.obs.columns:
            n_batches = adata.obs[col].nunique()
            logger.info(f"Detected batch key: '{col}' ({n_batches} batches)")
            return col

    logger.warning("No batch key detected")
    return None


def detect_label_key(adata: AnnData) -> Optional[str]:
    """Auto-detect cell type/label column in metadata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix

    Returns
    -------
    label_key : str or None
        Name of label column, or None if not found
    """
    # Common label column names
    candidates = [
        'cell_type', 'celltype', 'cell_label', 'label',
        'annotation', 'cluster', 'leiden', 'louvain'
    ]

    for col in candidates:
        if col in adata.obs.columns:
            n_types = adata.obs[col].nunique()
            logger.info(f"Detected label key: '{col}' ({n_types} types)")
            return col

    logger.warning("No label key detected")
    return None
