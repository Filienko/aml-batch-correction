"""Column name configuration and detection utilities."""

import logging
from typing import Optional, List
from anndata import AnnData

logger = logging.getLogger(__name__)


# Default column name aliases
DEFAULT_STUDY_COLUMNS = ['study', 'Study', 'dataset', 'batch', 'sample']
DEFAULT_CELL_TYPE_COLUMNS = ['cell_type', 'Cell Type', 'celltype', 'cell_label', 'annotation', 'cluster']


def detect_column(
    adata: AnnData,
    candidates: List[str],
    column_type: str = "column"
) -> Optional[str]:
    """Auto-detect a column from a list of candidates.

    Parameters
    ----------
    adata : AnnData
        Data object
    candidates : list of str
        List of candidate column names to try
    column_type : str
        Type of column for error messages (e.g., "study", "cell type")

    Returns
    -------
    column_name : str or None
        Detected column name, or None if not found
    """
    for col in candidates:
        if col in adata.obs.columns:
            logger.info(f"Auto-detected {column_type} column: '{col}'")
            return col

    logger.warning(f"No {column_type} column detected from candidates: {candidates}")
    return None


def get_study_column(adata: AnnData, study_col: Optional[str] = None) -> str:
    """Get or detect the study/batch column name.

    Parameters
    ----------
    adata : AnnData
        Data object
    study_col : str, optional
        Explicit column name. If None, will auto-detect.

    Returns
    -------
    column_name : str
        Study column name

    Raises
    ------
    ValueError
        If column not found

    Examples
    --------
    >>> # Auto-detect
    >>> study_col = get_study_column(adata)
    >>>
    >>> # Use custom column
    >>> study_col = get_study_column(adata, study_col='batch_id')
    """
    if study_col is not None:
        if study_col not in adata.obs.columns:
            raise ValueError(
                f"Specified study column '{study_col}' not found. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return study_col

    detected = detect_column(adata, DEFAULT_STUDY_COLUMNS, "study")
    if detected is None:
        raise ValueError(
            f"No study column found. Available columns: {list(adata.obs.columns)}. "
            f"Specify study_col parameter explicitly."
        )

    return detected


def get_cell_type_column(adata: AnnData, cell_type_col: Optional[str] = None) -> str:
    """Get or detect the cell type/annotation column name.

    Parameters
    ----------
    adata : AnnData
        Data object
    cell_type_col : str, optional
        Explicit column name. If None, will auto-detect.

    Returns
    -------
    column_name : str
        Cell type column name

    Raises
    ------
    ValueError
        If column not found

    Examples
    --------
    >>> # Auto-detect
    >>> cell_type_col = get_cell_type_column(adata)
    >>>
    >>> # Use custom column
    >>> cell_type_col = get_cell_type_column(adata, cell_type_col='annotation')
    """
    if cell_type_col is not None:
        if cell_type_col not in adata.obs.columns:
            raise ValueError(
                f"Specified cell type column '{cell_type_col}' not found. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return cell_type_col

    detected = detect_column(adata, DEFAULT_CELL_TYPE_COLUMNS, "cell type")
    if detected is None:
        raise ValueError(
            f"No cell type column found. Available columns: {list(adata.obs.columns)}. "
            f"Specify cell_type_col parameter explicitly."
        )

    return detected


def list_columns(adata: AnnData, verbose: bool = True) -> dict:
    """List and detect key columns in the data.

    Parameters
    ----------
    adata : AnnData
        Data object
    verbose : bool, default=True
        Whether to print information

    Returns
    -------
    columns : dict
        Dictionary with detected column names

    Examples
    --------
    >>> cols = list_columns(adata)
    >>> print(cols['study'])  # 'Study'
    >>> print(cols['cell_type'])  # 'cell_type'
    """
    result = {}

    # Detect study column
    result['study'] = detect_column(adata, DEFAULT_STUDY_COLUMNS, "study")

    # Detect cell type column
    result['cell_type'] = detect_column(adata, DEFAULT_CELL_TYPE_COLUMNS, "cell type")

    if verbose:
        print("="*60)
        print("COLUMN DETECTION")
        print("="*60)
        print(f"Total columns: {len(adata.obs.columns)}")
        print(f"\nDetected columns:")
        print(f"  Study column:     {result['study']}")
        print(f"  Cell type column: {result['cell_type']}")
        print(f"\nAll columns: {list(adata.obs.columns)}")
        print("="*60)

    return result
