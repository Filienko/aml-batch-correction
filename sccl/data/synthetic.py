"""Generate synthetic single cell data for testing."""

import logging
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_cells: int = 1000,
    n_genes: int = 2000,
    n_cell_types: int = 5,
    n_batches: int = 3,
    batch_effect_strength: float = 0.3,
    noise_level: float = 0.5,
    seed: int = 42,
) -> AnnData:
    """Generate synthetic single cell RNA-seq data with known structure.

    Creates data with:
    - Multiple cell types with distinct expression profiles
    - Batch effects
    - Realistic count distributions

    Parameters
    ----------
    n_cells : int, default=1000
        Number of cells to generate
    n_genes : int, default=2000
        Number of genes
    n_cell_types : int, default=5
        Number of distinct cell types
    n_batches : int, default=3
        Number of batches
    batch_effect_strength : float, default=0.3
        Strength of batch effects (0-1)
    noise_level : float, default=0.5
        Amount of random noise (0-1)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Synthetic annotated data matrix with:
        - .obs['cell_type']: True cell type labels
        - .obs['batch']: Batch assignments
        - .obs['cell_id']: Cell identifiers
        - .var['gene_name']: Gene names
        - .var['marker_for']: Cell type(s) this gene is a marker for (if any)

    Examples
    --------
    >>> # Generate small dataset for quick testing
    >>> adata = generate_synthetic_data(n_cells=500, n_genes=1000)
    >>>
    >>> # Generate dataset with strong batch effects
    >>> adata = generate_synthetic_data(
    ...     n_cells=2000,
    ...     n_batches=5,
    ...     batch_effect_strength=0.7
    ... )
    """
    np.random.seed(seed)

    logger.info(f"Generating synthetic data: {n_cells} cells x {n_genes} genes")

    # Generate cell type assignments
    cell_types = [f"CellType_{i+1}" for i in range(n_cell_types)]
    cell_type_labels = np.random.choice(cell_types, size=n_cells)

    # Generate batch assignments
    batches = [f"Batch_{i+1}" for i in range(n_batches)]
    batch_labels = np.random.choice(batches, size=n_cells)

    # Create cell type-specific expression profiles
    # Each cell type has elevated expression for a subset of genes
    genes_per_type = n_genes // n_cell_types
    expression_profiles = np.zeros((n_cell_types, n_genes))

    for i in range(n_cell_types):
        # Marker genes for this cell type
        start_idx = i * genes_per_type
        end_idx = min((i + 1) * genes_per_type, n_genes)
        expression_profiles[i, start_idx:end_idx] = 3.0  # High expression

        # Some genes are shared between adjacent cell types (biological reality)
        if i > 0:
            expression_profiles[i, (start_idx - genes_per_type//4):start_idx] = 1.5

    # Create batch effect profiles
    batch_profiles = np.random.randn(n_batches, n_genes) * batch_effect_strength

    # Generate expression matrix
    X = np.zeros((n_cells, n_genes))

    for i in range(n_cells):
        # Get cell type and batch
        cell_type_idx = cell_types.index(cell_type_labels[i])
        batch_idx = batches.index(batch_labels[i])

        # Base expression: cell type profile + batch effect + noise
        base_expression = (
            expression_profiles[cell_type_idx, :]
            + batch_profiles[batch_idx, :]
            + np.random.randn(n_genes) * noise_level
        )

        # Convert to counts (simulate sequencing depth variation)
        sequencing_depth = np.random.lognormal(mean=10, sigma=0.5)
        counts = np.random.poisson(np.exp(base_expression) * sequencing_depth)

        X[i, :] = counts

    # Create gene metadata
    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
    marker_info = []

    for i in range(n_genes):
        # Determine which cell type(s) this gene is a marker for
        markers_for = []
        for ct_idx in range(n_cell_types):
            if expression_profiles[ct_idx, i] > 2.0:
                markers_for.append(cell_types[ct_idx])

        marker_info.append(",".join(markers_for) if markers_for else "")

    var_df = pd.DataFrame({
        'gene_name': gene_names,
        'marker_for': marker_info,
    })
    var_df.index = gene_names

    # Create cell metadata
    obs_df = pd.DataFrame({
        'cell_type': cell_type_labels,
        'batch': batch_labels,
        'cell_id': [f"Cell_{i+1}" for i in range(n_cells)],
    })
    obs_df.index = obs_df['cell_id']

    # Create AnnData object
    adata = AnnData(
        X=X,
        obs=obs_df,
        var=var_df,
    )

    # Add some statistics
    adata.uns['synthetic'] = True
    adata.uns['n_cell_types'] = n_cell_types
    adata.uns['n_batches'] = n_batches
    adata.uns['batch_effect_strength'] = batch_effect_strength

    logger.info(f"Generated data with {n_cell_types} cell types and {n_batches} batches")
    logger.info(f"Cell type distribution:\n{pd.Series(cell_type_labels).value_counts()}")

    return adata


def generate_synthetic_with_hierarchy(
    n_cells: int = 1000,
    n_genes: int = 2000,
    hierarchy: Optional[dict] = None,
    n_batches: int = 3,
    seed: int = 42,
) -> AnnData:
    """Generate synthetic data with hierarchical cell type structure.

    Useful for testing models' ability to capture cell type relationships.

    Parameters
    ----------
    n_cells : int, default=1000
        Number of cells
    n_genes : int, default=2000
        Number of genes
    hierarchy : dict, optional
        Cell type hierarchy. Example:
        {
            'Myeloid': ['Monocyte', 'Macrophage', 'DC'],
            'Lymphoid': ['T_cell', 'B_cell', 'NK']
        }
        If None, uses default hierarchy
    n_batches : int, default=3
        Number of batches
    seed : int, default=42
        Random seed

    Returns
    -------
    adata : AnnData
        Synthetic data with:
        - .obs['cell_type']: Fine-grained cell type
        - .obs['cell_lineage']: High-level lineage
        - .obs['batch']: Batch assignment

    Examples
    --------
    >>> # Generate data with default hierarchy
    >>> adata = generate_synthetic_with_hierarchy(n_cells=1000)
    >>>
    >>> # Custom hierarchy
    >>> hierarchy = {
    ...     'Stem': ['HSC', 'MPP'],
    ...     'Progenitor': ['CMP', 'GMP', 'MEP']
    ... }
    >>> adata = generate_synthetic_with_hierarchy(hierarchy=hierarchy)
    """
    if hierarchy is None:
        hierarchy = {
            'Myeloid': ['Monocyte', 'Macrophage', 'DC'],
            'Lymphoid': ['T_cell', 'B_cell', 'NK'],
            'Stem': ['HSC', 'MPP'],
        }

    # Flatten hierarchy
    cell_types = []
    lineage_map = {}
    for lineage, types in hierarchy.items():
        cell_types.extend(types)
        for ct in types:
            lineage_map[ct] = lineage

    n_cell_types = len(cell_types)

    # Generate base data
    adata = generate_synthetic_data(
        n_cells=n_cells,
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        n_batches=n_batches,
        seed=seed,
    )

    # Replace generic cell type names with hierarchy names
    old_types = adata.obs['cell_type'].unique()
    type_mapping = dict(zip(old_types, cell_types[:len(old_types)]))

    adata.obs['cell_type'] = adata.obs['cell_type'].map(type_mapping)
    adata.obs['cell_lineage'] = adata.obs['cell_type'].map(lineage_map)

    logger.info(f"Generated hierarchical data with {len(hierarchy)} lineages")

    return adata
