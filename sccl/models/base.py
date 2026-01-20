"""Base model interface for single cell classification."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from anndata import AnnData


class BaseModel(ABC):
    """Abstract base class for single cell classification models.

    All models should inherit from this class and implement the required methods.
    """

    def __init__(self, **kwargs):
        """Initialize the model with optional parameters."""
        self.is_trained = False
        self.params = kwargs

    @abstractmethod
    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict labels for the given data.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        target_column : str, optional
            Target column name. If provided and model requires training,
            can be used to inform prediction.
        batch_key : str, optional
            Batch key for batch correction

        Returns
        -------
        predictions : np.ndarray
            Predicted labels
        """
        pass

    def fit(
        self,
        adata: AnnData,
        target_column: str,
        batch_key: Optional[str] = None,
    ) -> None:
        """Train the model on labeled data.

        Not all models require training (e.g., SCimilarity).

        Parameters
        ----------
        adata : AnnData
            Training data
        target_column : str
            Column with ground truth labels
        batch_key : str, optional
            Batch key for batch correction
        """
        # Default: no training required
        self.is_trained = True

    def get_embedding(
        self,
        adata: AnnData,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Get embedding representation of the data.

        Parameters
        ----------
        adata : AnnData
            Data to embed
        batch_key : str, optional
            Batch key for batch correction

        Returns
        -------
        embedding : np.ndarray
            Embedding matrix (n_cells x n_features)
        """
        # Default: return PCA embedding if available
        if 'X_pca' in adata.obsm:
            return adata.obsm['X_pca']
        else:
            raise NotImplementedError("Model does not provide embeddings")

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
