"""scTab model implementation for cell type classification."""

import logging
from typing import Optional
import numpy as np
from anndata import AnnData
import pandas as pd
import torch
from collections import OrderedDict
import yaml
from pathlib import Path

from .base import BaseModel

logger = logging.getLogger(__name__)


class ScTabModel(BaseModel):
    """scTab foundation model for cell type classification.

    scTab uses TabNet architecture pre-trained on large-scale single-cell data.

    Parameters
    ----------
    checkpoint_path : str
        Path to the scTab checkpoint file (.ckpt)
    model_dir : str
        Path to directory containing var.parquet, cell_type.parquet, and hparams.yaml
    batch_size : int, default=2048
        Batch size for inference

    Examples
    --------
    >>> model = ScTabModel(
    ...     checkpoint_path='scTab-checkpoints/scTab/run1/val_f1_macro_epoch=46_val_f1_macro=0.848.ckpt',
    ...     model_dir='merlin_cxg_2023_05_15_sf-log1p_minimal'
    ... )
    >>> predictions = model.predict(adata)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_dir: str,
        batch_size: int = 2048,
        **kwargs
    ):
        """Initialize scTab model."""
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size

        self._model = None
        self._genes_from_model = None
        self._cell_type_mapping = None
        self._model_params = None

        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load gene ordering and cell type mapping."""
        # Load gene order
        var_path = self.model_dir / "var.parquet"
        if not var_path.exists():
            raise FileNotFoundError(f"Missing var.parquet in {self.model_dir}")
        self._genes_from_model = pd.read_parquet(var_path)
        logger.info(f"Loaded {len(self._genes_from_model)} genes from model")

        # Load cell type mapping
        cell_type_path = self.model_dir / "categorical_lookup" / "cell_type.parquet"
        if not cell_type_path.exists():
            raise FileNotFoundError(f"Missing cell_type.parquet in {self.model_dir}")
        self._cell_type_mapping = pd.read_parquet(cell_type_path)
        logger.info(f"Loaded {len(self._cell_type_mapping)} cell types")

        # Load model hyperparameters
        hparams_path = self.checkpoint_path.parent / "hparams.yaml"
        if not hparams_path.exists():
            raise FileNotFoundError(f"Missing hparams.yaml at {hparams_path}")
        with open(hparams_path) as f:
            self._model_params = yaml.full_load(f.read())
        logger.info("Loaded model hyperparameters")

    def _load_model(self):
        """Load scTab model from checkpoint."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading scTab model from {self.checkpoint_path}")

        # Load checkpoint
        if torch.cuda.is_available():
            ckpt = torch.load(self.checkpoint_path)
        else:
            ckpt = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))

        # Extract TabNet weights
        tabnet_weights = OrderedDict()
        for name, weight in ckpt['state_dict'].items():
            if 'classifier.' in name:
                tabnet_weights[name.replace('classifier.', '')] = weight

        # Import TabNet
        try:
            from cellnet.tabnet.tab_network import TabNet
        except ImportError:
            raise ImportError(
                "cellnet not installed. Install with: pip install git+https://github.com/theislab/scTab.git"
            )

        # Initialize model
        model = TabNet(
            input_dim=self._model_params['gene_dim'],
            output_dim=self._model_params['type_dim'],
            n_d=self._model_params['n_d'],
            n_a=self._model_params['n_a'],
            n_steps=self._model_params['n_steps'],
            gamma=self._model_params['gamma'],
            n_independent=self._model_params['n_independent'],
            n_shared=self._model_params['n_shared'],
            epsilon=self._model_params['epsilon'],
            virtual_batch_size=self._model_params['virtual_batch_size'],
            momentum=self._model_params['momentum'],
            mask_type=self._model_params['mask_type'],
        )

        # Load weights and set to eval mode
        model.load_state_dict(tabnet_weights)
        model.eval()

        self._model = model
        logger.info("✓ scTab model loaded")
        return model

    def _streamline_count_matrix(self, adata: AnnData):
        """Align gene space to match model's expected genes."""
        from scipy.sparse import csc_matrix

        # Get gene names (try common column names)
        if 'feature_name' in adata.var.columns:
            gene_col = 'feature_name'
        elif 'gene_ids' in adata.var.columns:
            gene_col = 'gene_ids'
        else:
            # Use index
            gene_names = adata.var.index.to_numpy()
            gene_col = None

        if gene_col:
            gene_names = adata.var[gene_col].to_numpy()

        # Subset to genes in model
        genes_in_model = self._genes_from_model.feature_name.to_numpy()
        gene_mask = np.isin(gene_names, genes_in_model)

        if gene_mask.sum() == 0:
            raise ValueError("No genes overlap with model's gene space!")

        logger.info(f"Gene overlap: {gene_mask.sum()}/{len(gene_names)}")

        # Subset and reorder
        adata_subset = adata[:, gene_mask].copy()
        if gene_col:
            subset_genes = adata_subset.var[gene_col].to_numpy()
        else:
            subset_genes = adata_subset.var.index.to_numpy()

        # Reorder to match model's gene order
        gene_to_idx = {g: i for i, g in enumerate(subset_genes)}
        model_gene_indices = [gene_to_idx.get(g, -1) for g in genes_in_model]

        # Create aligned matrix (zero-filled for missing genes)
        n_cells = adata_subset.n_obs
        n_genes = len(genes_in_model)
        aligned_matrix = np.zeros((n_cells, n_genes), dtype=np.float32)

        X = adata_subset.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        for i, idx in enumerate(model_gene_indices):
            if idx >= 0:
                aligned_matrix[:, i] = X[:, idx]

        return aligned_matrix

    def _sf_log1p_norm(self, x):
        """Normalize each cell to have 10000 counts and apply log(x+1) transform."""
        x_tensor = torch.from_numpy(x).float()
        counts = torch.sum(x_tensor, dim=1, keepdim=True)
        counts += counts == 0.  # Avoid zero division
        scaling_factor = 10000. / counts
        return torch.log1p(scaling_factor * x_tensor)

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict cell types using scTab.

        Parameters
        ----------
        adata : AnnData
            Data to predict on (raw counts expected)
        target_column : str, optional
            Not used (scTab doesn't support supervised training)
        batch_key : str, optional
            Not used

        Returns
        -------
        predictions : np.ndarray
            Predicted cell type labels
        """
        # Load model if needed
        model = self._load_model()

        # Get raw counts
        if adata.raw is not None:
            adata_raw = adata.raw.to_adata()
        else:
            adata_raw = adata.copy()

        # Align genes
        logger.info("Aligning gene space...")
        x_streamlined = self._streamline_count_matrix(adata_raw)

        # Create batches
        n_cells = x_streamlined.shape[0]
        preds = []

        logger.info(f"Running scTab inference on {n_cells:,} cells...")
        with torch.no_grad():
            for i in range(0, n_cells, self.batch_size):
                batch_x = x_streamlined[i:i+self.batch_size]

                # Normalize
                x_input = self._sf_log1p_norm(batch_x)

                # Predict
                logits, _ = model(x_input)
                batch_preds = torch.argmax(logits, dim=1).numpy()
                preds.append(batch_preds)

        # Concatenate all predictions
        preds = np.hstack(preds)

        # Map integers to cell type labels
        predictions = self._cell_type_mapping.loc[preds]['label'].to_numpy()

        self.is_trained = True
        logger.info(f"✓ scTab predictions: {len(set(predictions))} unique cell types")
        return predictions

    def fit(
        self,
        adata: AnnData,
        target_column: str,
        batch_key: Optional[str] = None,
    ) -> None:
        """scTab doesn't support custom training - uses pre-trained model only."""
        logger.warning("scTab uses pre-trained model only - fit() does nothing")
        self.is_trained = True

    def __repr__(self) -> str:
        """String representation."""
        return f"ScTabModel(checkpoint={self.checkpoint_path.name})"
