"""CellTypist model implementation."""

import logging
from typing import Optional
import numpy as np
from anndata import AnnData
import scanpy as sc

from .base import BaseModel

logger = logging.getLogger(__name__)


class CellTypistModel(BaseModel):
    """CellTypist automated cell type annotation.

    CellTypist uses pre-trained logistic regression models for cell type
    prediction. It provides several built-in models trained on large
    reference datasets.

    Parameters
    ----------
    model : str, default='Immune_All_Low.pkl'
        Pre-trained model to use. Options:
        - 'Immune_All_Low.pkl': All immune cells (27 types)
        - 'Immune_All_High.pkl': All immune cells (59 types)
        - 'Healthy_COVID19_PBMC.pkl': PBMC focused
        - Path to custom trained model
    majority_voting : bool, default=True
        Whether to use majority voting for cell type refinement
    over_clustering : str, optional
        Key in adata.obs for over-clustering (for majority voting)

    Examples
    --------
    >>> # Use pre-trained model
    >>> model = CellTypistModel(model='Immune_All_Low.pkl')
    >>> predictions = model.predict(adata)
    >>>
    >>> # Custom model
    >>> model = CellTypistModel(model='path/to/my_model.pkl')
    >>> predictions = model.predict(adata)
    """

    def __init__(
        self,
        model: str = 'Immune_All_Low.pkl',
        majority_voting: bool = True,
        over_clustering: Optional[str] = None,
        **kwargs
    ):
        """Initialize CellTypist model."""
        super().__init__(**kwargs)
        self.model_name = model
        self.majority_voting = majority_voting
        self.over_clustering = over_clustering
        self._celltypist = None
        self._model = None

    def _get_celltypist(self):
        """Lazy load CellTypist."""
        if self._celltypist is None:
            try:
                import celltypist
                self._celltypist = celltypist
                logger.info("CellTypist loaded successfully")
            except ImportError:
                raise ImportError(
                    "CellTypist not installed. Install with: pip install celltypist"
                )
        return self._celltypist

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict cell types using CellTypist.

        Parameters
        ----------
        adata : AnnData
            Data to predict on
        target_column : str, optional
            Not used (CellTypist doesn't require labels)
        batch_key : str, optional
            Not used

        Returns
        -------
        predictions : np.ndarray
            Predicted cell type labels
        """
        ct = self._get_celltypist()

        logger.info(f"Running CellTypist with model: {self.model_name}")

        # CellTypist expects normalized, log-transformed data
        adata_proc = adata.copy()

        # Check if already normalized
        if 'log1p' not in adata_proc.uns:
            logger.info("Normalizing and log-transforming data...")
            sc.pp.normalize_total(adata_proc, target_sum=1e4)
            sc.pp.log1p(adata_proc)

        # Load or download model
        try:
            if self.model_name.endswith('.pkl'):
                # Check if it's a built-in model or custom path
                try:
                    model = ct.models.Model.load(self.model_name)
                except:
                    # Try downloading from CellTypist repository
                    logger.info(f"Downloading model: {self.model_name}")
                    model = ct.models.Model.load(model=self.model_name)
            else:
                # Assume it's a model name that needs downloading
                logger.info(f"Downloading model: {self.model_name}")
                model = ct.models.Model.load(model=self.model_name)

            self._model = model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Available models: https://www.celltypist.org/models")
            raise

        # Run prediction
        logger.info("Predicting cell types...")
        predictions = ct.annotate(
            adata_proc,
            model=self._model,
            majority_voting=self.majority_voting,
            over_clustering=self.over_clustering
        )

        # Get predicted labels
        if self.majority_voting and 'majority_voting' in predictions.predicted_labels.columns:
            predicted_labels = predictions.predicted_labels['majority_voting'].values
        else:
            predicted_labels = predictions.predicted_labels['predicted_labels'].values

        self.is_trained = True
        return predicted_labels

    def fit(
        self,
        adata: AnnData,
        target_column: str,
        batch_key: Optional[str] = None,
    ) -> None:
        """Train CellTypist model on reference data.

        Parameters
        ----------
        adata : AnnData
            Reference data with labels
        target_column : str
            Column in adata.obs containing cell type labels
        batch_key : str, optional
            Not used
        """
        ct = self._get_celltypist()

        logger.info("Training CellTypist model...")

        if target_column not in adata.obs.columns:
            raise ValueError(f"Column '{target_column}' not found in adata.obs")

        # Prepare data
        adata_proc = adata.copy()

        # Normalize if needed
        if 'log1p' not in adata_proc.uns:
            logger.info("Normalizing and log-transforming data...")
            sc.pp.normalize_total(adata_proc, target_sum=1e4)
            sc.pp.log1p(adata_proc)

        # Get labels
        labels = adata_proc.obs[target_column].values

        # Remove cells with missing labels
        import pandas as pd
        valid_mask = pd.notna(labels)
        if valid_mask.sum() < len(labels):
            logger.info(f"Removing {(~valid_mask).sum()} cells with missing labels")
            adata_proc = adata_proc[valid_mask].copy()

        # Train model
        logger.info(f"Training on {adata_proc.n_obs:,} cells...")
        self._model = ct.train(
            adata_proc,
            labels=target_column,
            n_jobs=-1,
            feature_selection=True
        )

        self.is_trained = True
        logger.info("✓ CellTypist model trained")

    def __repr__(self) -> str:
        """String representation."""
        return f"CellTypistModel(model={self.model_name}, majority_voting={self.majority_voting})"
