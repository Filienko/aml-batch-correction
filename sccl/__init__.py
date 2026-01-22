"""SCCL: Single Cell Classification Library

A flexible tool for single cell RNA-seq classification and batch correction.
"""

from .pipeline import Pipeline
from .models import AVAILABLE_MODELS

__version__ = "0.1.0"
__all__ = ["Pipeline", "AVAILABLE_MODELS"]
