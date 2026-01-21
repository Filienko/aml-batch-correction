"""SCimilarity foundation model implementation."""

import logging
from typing import Optional
import numpy as np
from anndata import AnnData
import scanpy as sc

from .base import BaseModel

logger = logging.getLogger(__name__)


class SCimilarityModel(BaseModel):
    """SCimilarity foundation model for cell type classification.

    SCimilarity is a pre-trained model that projects cells into a learned
    latent space without requiring training. It performs well for cell type
    annotation and batch correction.

    Parameters
    ----------
    model_path : str, optional
        Path to pretrained model. If None, uses default SCimilarity model.
    n_neighbors : int, default=15
        Number of neighbors for clustering
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering
    species : str, default='human'
        Species for gene alignment ('human' or 'mouse')

    Examples
    --------
    >>> model = SCimilarityModel()
    >>> predictions = model.predict(adata)
    >>>
    >>> # For mouse data
    >>> model = SCimilarityModel(species='mouse')
    >>> predictions = model.predict(adata)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        species: str = 'human',
        **kwargs
    ):
        """Initialize SCimilarity model."""
        super().__init__(**kwargs)
        self.model_path = model_path
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.species = species
        self._scimilarity = None
        self._embedding = None

    def _get_scimilarity(self):
        """Lazy load SCimilarity model."""
        if self._scimilarity is None:
            try:
                import scimilarity
                self._scimilarity = scimilarity
                logger.info("SCimilarity loaded successfully")
            except ImportError:
                raise ImportError(
                    "SCimilarity not installed. Install with: pip install scimilarity"
                )
        return self._scimilarity

    def get_embedding(
        self,
        adata: AnnData,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Get SCimilarity embedding.

        Parameters
        ----------
        adata : AnnData
            Data to embed
        batch_key : str, optional
            Not used for SCimilarity

        Returns
        -------
        embedding : np.ndarray
            SCimilarity embedding
        """
        scim = self._get_scimilarity()

        # Ensure raw counts are available
        if adata.raw is not None:
            adata_raw = adata.raw.to_adata()
        else:
            adata_raw = adata.copy()

        logger.info("Computing SCimilarity embeddings...")

        # Align genes to model vocabulary
        adata_aligned = scim.utils.align_dataset(adata_raw, species=self.species)

        # Normalize
        sc.pp.normalize_total(adata_aligned, target_sum=1e4)
        sc.pp.log1p(adata_aligned)

        # Get embeddings
        embeddings = scim.get_embeddings(adata_aligned)

        self._embedding = embeddings
        return embeddings

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict cell types using SCimilarity + clustering.

        Parameters
        ----------
        adata : AnnData
            Data to predict on
        target_column : str, optional
            Not used (SCimilarity doesn't require labels)
        batch_key : str, optional
            Not used (SCimilarity handles batches implicitly)

        Returns
        -------
        predictions : np.ndarray
            Cluster assignments
        """
        # Get embeddings
        embeddings = self.get_embedding(adata, batch_key=batch_key)

        # Create temporary AnnData with embeddings
        adata_emb = AnnData(embeddings)
        adata_emb.obs_names = adata.obs_names

        # Compute neighbors and cluster
        logger.info("Computing neighborhood graph...")
        sc.pp.neighbors(adata_emb, n_neighbors=self.n_neighbors, use_rep='X')

        logger.info("Performing Leiden clustering...")
        sc.tl.leiden(adata_emb, resolution=self.resolution)

        predictions = adata_emb.obs['leiden'].values

        # If target column is provided and we want to do KNN label transfer
        if target_column is not None and target_column in adata.obs:
            from sklearn.neighbors import KNeighborsClassifier

            # Get labeled cells
            mask = ~adata.obs[target_column].isna()
            if mask.sum() > 0:
                logger.info("Performing KNN label transfer in SCimilarity space...")

                X_train = embeddings[mask]
                y_train = adata.obs[target_column].values[mask]

                # Train KNN
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                knn.fit(X_train, y_train)

                # Predict all cells
                predictions = knn.predict(embeddings)

        self.is_trained = True
        return predictions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SCimilarityModel(n_neighbors={self.n_neighbors}, "
            f"resolution={self.resolution})"
        )
