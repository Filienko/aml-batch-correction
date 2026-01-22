"""scVI model implementation for batch correction."""

import logging
from typing import Optional
import numpy as np
from anndata import AnnData
import scanpy as sc

from .base import BaseModel

logger = logging.getLogger(__name__)


class ScVIModel(BaseModel):
    """scVI model for batch correction and cell type prediction.

    scVI is a deep generative model that learns a low-dimensional representation
    of single-cell data while correcting for batch effects.

    Parameters
    ----------
    n_latent : int, default=30
        Dimensionality of latent space
    n_layers : int, default=2
        Number of layers in encoder/decoder
    max_epochs : int, default=400
        Maximum training epochs
    n_neighbors : int, default=15
        Number of neighbors for clustering
    resolution : float, default=1.0
        Resolution for Leiden clustering

    Examples
    --------
    >>> model = ScVIModel(n_latent=30, max_epochs=200)
    >>> predictions = model.predict(adata, batch_key='study')
    """

    def __init__(
        self,
        n_latent: int = 30,
        n_layers: int = 2,
        max_epochs: int = 400,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        **kwargs
    ):
        """Initialize scVI model."""
        super().__init__(**kwargs)
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self._model = None
        self._embedding = None

    def _get_scvi(self):
        """Lazy load scvi-tools."""
        try:
            import scvi
            return scvi
        except ImportError:
            raise ImportError(
                "scvi-tools not installed. Install with: pip install scvi-tools"
            )

    def fit(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> None:
        """Train scVI model.

        Parameters
        ----------
        adata : AnnData
            Training data
        target_column : str, optional
            Not used (scVI is unsupervised)
        batch_key : str, optional
            Batch key for batch correction
        """
        scvi = self._get_scvi()

        logger.info("Training scVI model...")

        # Setup scVI
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            layer=None,  # Use .X
        )

        # Create model
        self._model = scvi.model.SCVI(
            adata,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
        )

        # Train
        self._model.train(max_epochs=self.max_epochs)

        self.is_trained = True
        logger.info("scVI training completed")

    def get_embedding(
        self,
        adata: AnnData,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Get scVI latent representation.

        Parameters
        ----------
        adata : AnnData
            Data to embed
        batch_key : str, optional
            Batch key for batch correction

        Returns
        -------
        embedding : np.ndarray
            scVI latent representation
        """
        if self._model is None:
            # Train if not trained
            self.fit(adata, batch_key=batch_key)

        # Get latent representation
        self._embedding = self._model.get_latent_representation(adata)

        return self._embedding

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict cell types using scVI + clustering.

        Parameters
        ----------
        adata : AnnData
            Data to predict on
        target_column : str, optional
            If provided, performs KNN label transfer
        batch_key : str, optional
            Batch key for batch correction

        Returns
        -------
        predictions : np.ndarray
            Cluster assignments or transferred labels
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

        # If target column provided, do label transfer
        if target_column is not None and target_column in adata.obs:
            from sklearn.neighbors import KNeighborsClassifier

            # Get labeled cells
            mask = ~adata.obs[target_column].isna()
            if mask.sum() > 0:
                logger.info("Performing KNN label transfer in scVI space...")

                X_train = embeddings[mask]
                y_train = adata.obs[target_column].values[mask]

                # Train KNN
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                knn.fit(X_train, y_train)

                # Predict all cells
                predictions = knn.predict(embeddings)

        return predictions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ScVIModel(n_latent={self.n_latent}, n_layers={self.n_layers}, "
            f"max_epochs={self.max_epochs})"
        )
