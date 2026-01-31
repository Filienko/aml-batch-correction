"""Data preprocessing utilities."""

import logging
from typing import Optional, Union, List
import numpy as np
from anndata import AnnData
import scanpy as sc

logger = logging.getLogger(__name__)


def preprocess_data(
    adata: AnnData,
    batch_key: Optional[str] = None,
    n_top_genes: int = 2000,
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: float = 1e4,
    copy: bool = True,
) -> AnnData:
    """Standard preprocessing pipeline for single cell data.

    Steps:
    1. Filter cells and genes
    2. Normalize counts
    3. Log transform
    4. Select highly variable genes
    5. Scale and center
    6. PCA

    Parameters
    ----------
    adata : AnnData
        Input data
    batch_key : str, optional
        Batch key for batch-aware HVG selection
    n_top_genes : int, default=2000
        Number of highly variable genes to select
    min_genes : int, default=200
        Minimum number of genes per cell
    min_cells : int, default=3
        Minimum number of cells per gene
    target_sum : float, default=1e4
        Target sum for normalization
    copy : bool, default=True
        Whether to return a copy

    Returns
    -------
    adata : AnnData
        Preprocessed data
    """
    if adata.isbacked:
        adata = adata.to_memory()
    elif copy:
        adata = adata.copy()

    logger.info("Starting preprocessing...")

    # Store raw counts
    if adata.raw is None:
        adata.raw = adata

    # Basic filtering
    logger.info(f"Initial: {adata.n_obs} cells x {adata.n_vars} genes")

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    logger.info(f"After filtering: {adata.n_obs} cells x {adata.n_vars} genes")

    # Normalize
    logger.info("Normalizing counts...")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Store normalized data
    adata.uns['log1p'] = True

    # Select highly variable genes
    logger.info(f"Selecting {n_top_genes} highly variable genes...")
    if batch_key is not None and batch_key in adata.obs:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            flavor='seurat_v3',
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat_v3',
        )

    # Subset to HVGs
    adata = adata[:, adata.var['highly_variable']].copy()
    logger.info(f"Selected {adata.n_vars} HVGs")

    # Scale
    logger.info("Scaling data...")
    sc.pp.scale(adata, max_value=10)

    # PCA
    logger.info("Computing PCA...")
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

    logger.info("Preprocessing completed")

    return adata


def subset_data(
    adata: AnnData,
    obs_filter: Optional[dict] = None,
    var_filter: Optional[dict] = None,
    studies: Optional[List[str]] = None,
    cell_types: Optional[List[str]] = None,
    n_cells: Optional[int] = None,
    study_col: Optional[str] = None,
    cell_type_col: Optional[str] = None,
    copy: bool = True,
) -> AnnData:
    """Subset single cell data based on various criteria.

    Parameters
    ----------
    adata : AnnData
        Input data
    obs_filter : dict, optional
        Dictionary mapping obs columns to values to keep.
        Example: {'study': ['study1', 'study2'], 'tissue': 'bone_marrow'}
    var_filter : dict, optional
        Dictionary for filtering genes
    studies : list of str, optional
        List of studies to keep (shortcut for obs_filter)
    cell_types : list of str, optional
        List of cell types to keep (shortcut for obs_filter)
    n_cells : int, optional
        Randomly downsample to this many cells
    study_col : str, optional
        Name of the study/batch column. If not provided, will try to auto-detect
        from common names: 'study', 'Study', 'dataset', 'batch'
    cell_type_col : str, optional
        Name of the cell type column. If not provided, will try to auto-detect
        from common names: 'cell_type', 'Cell Type', 'celltype', 'cell_label', 'annotation'
    copy : bool, default=True
        Whether to return a copy

    Returns
    -------
    adata : AnnData
        Subsetted data

    Examples
    --------
    >>> # Keep specific studies (auto-detect column)
    >>> adata_sub = subset_data(adata, studies=['study1', 'study2'])
    >>>
    >>> # Keep specific studies (custom column name)
    >>> adata_sub = subset_data(adata, studies=['study1', 'study2'], study_col='batch_id')
    >>>
    >>> # Keep specific cell types
    >>> adata_sub = subset_data(adata, cell_types=['T cell', 'B cell'])
    >>>
    >>> # Custom filter
    >>> adata_sub = subset_data(
    ...     adata,
    ...     obs_filter={'donor': ['D1', 'D2'], 'condition': 'healthy'}
    ... )
    """
    if adata.isbacked:
        adata = adata.to_memory()
    elif copy:
        adata = adata.copy()
    logger.info(f"Starting subset. Initial: {adata.n_obs} cells")

    # Build obs filter from shortcuts
    if obs_filter is None:
        obs_filter = {}

    if studies is not None:
        # Use provided column name or try to auto-detect
        if study_col is not None:
            if study_col not in adata.obs.columns:
                raise ValueError(f"Specified study column '{study_col}' not found in obs. Columns: {adata.obs.columns.tolist()}")
            detected_col = study_col
        else:
            # Try common study column names
            detected_col = None
            for col in ['study', 'Study', 'dataset', 'batch']:
                if col in adata.obs.columns:
                    detected_col = col
                    break
            if detected_col is None:
                raise ValueError(f"No study column found in obs. Columns: {adata.obs.columns.tolist()}. Specify study_col parameter.")

        obs_filter[detected_col] = studies
        logger.info(f"Using study column: '{detected_col}'")

    if cell_types is not None:
        # Use provided column name or try to auto-detect
        if cell_type_col is not None:
            if cell_type_col not in adata.obs.columns:
                raise ValueError(f"Specified cell type column '{cell_type_col}' not found in obs. Columns: {adata.obs.columns.tolist()}")
            detected_col = cell_type_col
        else:
            # Try common cell type column names
            detected_col = None
            for col in ['cell_type', 'Cell Type', 'celltype', 'cell_label', 'annotation']:
                if col in adata.obs.columns:
                    detected_col = col
                    break
            if detected_col is None:
                raise ValueError(f"No cell type column found in obs. Columns: {adata.obs.columns.tolist()}. Specify cell_type_col parameter.")

        obs_filter[detected_col] = cell_types
        logger.info(f"Using cell type column: '{detected_col}'")

    # Apply obs filter
    if obs_filter:
        mask = np.ones(adata.n_obs, dtype=bool)

        for col, values in obs_filter.items():
            if col not in adata.obs.columns:
                raise ValueError(f"Column '{col}' not found in obs")

            # Handle single value or list
            if not isinstance(values, (list, tuple, set)):
                values = [values]

            col_mask = adata.obs[col].isin(values)
            mask &= col_mask

            logger.info(f"Filter '{col}': {col_mask.sum()} cells match")

        adata = adata[mask].copy()
        logger.info(f"After obs filter: {adata.n_obs} cells")

    # Apply var filter
    if var_filter:
        mask = np.ones(adata.n_vars, dtype=bool)

        for col, values in var_filter.items():
            if col not in adata.var.columns:
                raise ValueError(f"Column '{col}' not found in var")

            if not isinstance(values, (list, tuple, set)):
                values = [values]

            mask &= adata.var[col].isin(values)

        adata = adata[:, mask].copy()
        logger.info(f"After var filter: {adata.n_vars} genes")

    # Downsample
    if n_cells is not None and n_cells < adata.n_obs:
        logger.info(f"Downsampling to {n_cells} cells...")
        sc.pp.subsample(adata, n_obs=n_cells, random_state=42)

    logger.info(f"Final: {adata.n_obs} cells x {adata.n_vars} genes")

    return adata
