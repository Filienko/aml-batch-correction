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
        self._ca_model = None
        self._embedding = None

        # For label transfer
        self._reference_embeddings = None
        self._reference_labels = None
        self._knn_classifier = None

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

        # Load CellAnnotation model if not already loaded
        if not hasattr(self, '_ca_model') or self._ca_model is None:
            logger.info(f"Loading SCimilarity CellAnnotation model (species: {self.species})...")
            # Load model with specified path or default
            model_path = self.model_path if self.model_path else "default"
            try:
                self._ca_model = scim.CellAnnotation(model_path=model_path)
            except Exception as e:
                logger.warning(f"Error loading model with path '{model_path}': {e}")
                logger.info("Trying with default model path...")
                self._ca_model = scim.CellAnnotation(model_path="default")

        # Get gene order from the model
        target_gene_order = self._ca_model.gene_order
        logger.info(f"Using gene order with {len(target_gene_order)} genes")

        # Align genes to model vocabulary
        adata_aligned = scim.utils.align_dataset(adata_raw, target_gene_order)

        # Normalize
        sc.pp.normalize_total(adata_aligned, target_sum=1e4)
        sc.pp.log1p(adata_aligned)

        # Get embeddings using CellAnnotation model
        # get_embeddings expects the expression matrix (X), not the full AnnData
        embeddings = self._ca_model.get_embeddings(adata_aligned.X)

        self._embedding = embeddings
        return embeddings

    def fit(
        self,
        adata: AnnData,
        target_column: str,
        batch_key: Optional[str] = None,
    ) -> None:
        """Train SCimilarity for label transfer.

        Computes embeddings for reference data and trains a KNN classifier
        for label transfer to new data.

        Parameters
        ----------
        adata : AnnData
            Reference data with labels
        target_column : str
            Column in adata.obs containing cell type labels
        batch_key : str, optional
            Not used for SCimilarity
        """
        from sklearn.neighbors import KNeighborsClassifier
        import pandas as pd

        logger.info("Training SCimilarity for label transfer...")

        if target_column not in adata.obs.columns:
            raise ValueError(f"Column '{target_column}' not found in adata.obs")

        # Get embeddings for reference data
        self._reference_embeddings = self.get_embedding(adata, batch_key=batch_key)

        # Get labels, removing any NaN values
        labels = adata.obs[target_column].values
        valid_mask = pd.notna(labels)

        if valid_mask.sum() == 0:
            raise ValueError(f"No valid labels found in column '{target_column}'")

        self._reference_embeddings = self._reference_embeddings[valid_mask]
        self._reference_labels = labels[valid_mask]

        # Train KNN classifier in embedding space
        logger.info(f"Training KNN classifier with {len(self._reference_labels)} reference cells...")
        self._knn_classifier = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=-1
        )
        self._knn_classifier.fit(self._reference_embeddings, self._reference_labels)

        self.is_trained = True
        logger.info("✓ SCimilarity trained for label transfer")

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict cell types using SCimilarity.

        If fit() was called, uses KNN label transfer from reference data.
        Otherwise, performs clustering.

        Parameters
        ----------
        adata : AnnData
            Data to predict on
        target_column : str, optional
            If provided and exists in adata, performs within-data KNN transfer
        batch_key : str, optional
            Not used (SCimilarity handles batches implicitly)

        Returns
        -------
        predictions : np.ndarray
            Predicted labels (if trained) or cluster assignments
        """
        # Get embeddings
        embeddings = self.get_embedding(adata, batch_key=batch_key)

        # If we have a trained KNN classifier, use it for label transfer
        if self._knn_classifier is not None:
            logger.info("Using trained KNN classifier for label transfer...")
            predictions = self._knn_classifier.predict(embeddings)
            return predictions

        # Otherwise, fall back to clustering or within-data transfer
        # Create temporary AnnData with embeddings
        adata_emb = AnnData(embeddings)
        adata_emb.obs_names = adata.obs_names

        # If target column is provided and we want to do KNN label transfer
        if target_column is not None and target_column in adata.obs:
            from sklearn.neighbors import KNeighborsClassifier

            # Get labeled cells
            mask = ~adata.obs[target_column].isna()
            if mask.sum() > 0:
                logger.info("Performing within-data KNN label transfer in SCimilarity space...")

                X_train = embeddings[mask]
                y_train = adata.obs[target_column].values[mask]

                # Train KNN
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1)
                knn.fit(X_train, y_train)

                # Predict all cells
                predictions = knn.predict(embeddings)
                return predictions

        # No labels available - do clustering
        logger.info("Computing neighborhood graph...")
        sc.pp.neighbors(adata_emb, n_neighbors=self.n_neighbors, use_rep='X')

        logger.info("Performing Leiden clustering...")
        sc.tl.leiden(adata_emb, resolution=self.resolution)

        predictions = adata_emb.obs['leiden'].values

        return predictions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SCimilarityModel(n_neighbors={self.n_neighbors}, "
            f"resolution={self.resolution})"
        )
