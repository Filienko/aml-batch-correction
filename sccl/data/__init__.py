"""Data utilities for loading, preprocessing, and generating data."""

from .loader import load_data, detect_batch_key, detect_label_key
from .preprocessing import preprocess_data, subset_data
from .synthetic import generate_synthetic_data
from .column_config import (
    get_study_column,
    get_cell_type_column,
    list_columns,
    DEFAULT_STUDY_COLUMNS,
    DEFAULT_CELL_TYPE_COLUMNS,
)

__all__ = [
    'load_data',
    'detect_batch_key',
    'detect_label_key',
    'preprocess_data',
    'subset_data',
    'generate_synthetic_data',
    'get_study_column',
    'get_cell_type_column',
    'list_columns',
    'DEFAULT_STUDY_COLUMNS',
    'DEFAULT_CELL_TYPE_COLUMNS',
]
