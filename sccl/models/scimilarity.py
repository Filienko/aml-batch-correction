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
    classifier : str, default='knn'
        Classifier to use on embeddings for label transfer.
        Options: 'knn', 'random_forest', 'svm', 'logistic_regression'
    classifier_params : dict, optional
        Additional parameters to pass to the classifier
    label_propagation : bool, default=False
        Whether to use label propagation for semi-supervised refinement.
        Similar to CellTypist's majority voting. Uses query data structure
        to smooth predictions via kNN majority voting in embedding space.
    propagation_neighbors : int, default=15
        Number of neighbors for label propagation (if enabled)

    Examples
    --------
    >>> # Pure supervised: KNN on embeddings
    >>> model = SCimilarityModel()
    >>> model.fit(adata_ref, target_column='cell_type')
    >>> predictions = model.predict(adata_query)
    >>>
    >>> # Semi-supervised: with label propagation
    >>> model = SCimilarityModel(label_propagation=True)
    >>> model.fit(adata_ref, target_column='cell_type')
    >>> predictions = model.predict(adata_query)  # Uses query structure
    >>>
    >>> # Random Forest on embeddings
    >>> model = SCimilarityModel(classifier='random_forest')
    >>> model.fit(adata_ref, target_column='cell_type')
    >>> predictions = model.predict(adata_query)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        species: str = 'human',
        classifier: Optional[str] = 'knn',
        classifier_params: Optional[dict] = None,
        label_propagation: bool = False,
        propagation_neighbors: int = 15,
        **kwargs
    ):
        """Initialize SCimilarity model."""
        super().__init__(**kwargs)
        self.model_path = model_path
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.species = species
        self.classifier = classifier
        self.classifier_params = classifier_params or {}
        self.label_propagation = label_propagation
        self.propagation_neighbors = propagation_neighbors
        self._scimilarity = None
        self._ca_model = None
        self._embedding = None

        # For label transfer
        self._reference_embeddings = None
        self._reference_labels = None
        self._trained_classifier = None

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

        # Store counts in layers if not present (required by lognorm_counts)
        if 'counts' not in adata_aligned.layers:
            adata_aligned.layers['counts'] = adata_aligned.X.copy()

        # Normalize using SCimilarity's lognorm_counts (critical for proper embeddings!)
        adata_normalized = scim.utils.lognorm_counts(adata_aligned)

        # Get embeddings using CellAnnotation model
        # get_embeddings expects the expression matrix (X), not the full AnnData
        embeddings = self._ca_model.get_embeddings(adata_normalized.X)

        self._embedding = embeddings
        return embeddings

    def fit(
        self,
        adata: AnnData,
        target_column: str,
        batch_key: Optional[str] = None,
    ) -> None:
        """Train SCimilarity for label transfer.

        Computes embeddings for reference data and trains a classifier
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

        # Get classifier based on type
        classifier_obj = self._get_classifier()

        # Train classifier in embedding space
        logger.info(f"Training {self.classifier} classifier with {len(self._reference_labels)} reference cells...")
        self._trained_classifier = classifier_obj
        self._trained_classifier.fit(self._reference_embeddings, self._reference_labels)

        self.is_trained = True
        logger.info(f"✓ SCimilarity trained for label transfer with {self.classifier}")

    def _get_classifier(self):
        """Get classifier instance based on type."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        if self.classifier == 'knn':
            params = {'n_neighbors': self.n_neighbors, 'n_jobs': -1}
            params.update(self.classifier_params)
            return KNeighborsClassifier(**params)
        elif self.classifier == 'random_forest':
            params = {'n_estimators': 100, 'max_depth': 20, 'n_jobs': -1, 'random_state': 42}
            params.update(self.classifier_params)
            return RandomForestClassifier(**params)
        elif self.classifier == 'svm':
            params = {'kernel': 'rbf', 'probability': True, 'random_state': 42}
            params.update(self.classifier_params)
            return SVC(**params)
        elif self.classifier == 'logistic_regression':
            params = {'max_iter': 1000, 'n_jobs': -1, 'random_state': 42}
            params.update(self.classifier_params)
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier}")

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

        # If we have a trained classifier, use it for label transfer
        if self._trained_classifier is not None:
            logger.info(f"Using trained {self.classifier} classifier for label transfer...")
            predictions = self._trained_classifier.predict(embeddings)

            # Apply label propagation if enabled (semi-supervised refinement)
            if self.label_propagation:
                logger.info(f"Applying label propagation with k={self.propagation_neighbors}...")
                predictions = self._refine_with_label_propagation(embeddings, predictions)

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

    def _refine_with_label_propagation(
        self,
        embeddings: np.ndarray,
        initial_predictions: np.ndarray
    ) -> np.ndarray:
        """Refine predictions using label propagation on query data.

        This is semi-supervised: uses query cell-cell similarity to smooth predictions.
        Similar to CellTypist's majority voting.

        Parameters
        ----------
        embeddings : np.ndarray
            Query embeddings
        initial_predictions : np.ndarray
            Initial predictions from classifier

        Returns
        -------
        refined_predictions : np.ndarray
            Refined predictions after label propagation
        """
        from sklearn.neighbors import NearestNeighbors
        from collections import Counter

        # Build kNN graph in embedding space
        nn = NearestNeighbors(n_neighbors=self.propagation_neighbors + 1, n_jobs=-1)
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        # For each cell, do majority voting over neighbors (including self)
        refined_predictions = np.empty_like(initial_predictions)

        for i in range(len(embeddings)):
            neighbor_indices = indices[i]  # includes self
            neighbor_labels = initial_predictions[neighbor_indices]

            # Majority vote using Counter (handles strings properly)
            label_counts = Counter(neighbor_labels)
            most_common_label = label_counts.most_common(1)[0][0]
            refined_predictions[i] = most_common_label

        # Count how many predictions changed
        n_changed = (initial_predictions != refined_predictions).sum()
        pct_changed = n_changed / len(initial_predictions) * 100
        logger.info(f"Label propagation: {n_changed}/{len(initial_predictions)} ({pct_changed:.1f}%) predictions changed")

        return refined_predictions

    # ------------------------------------------------------------------
    # Hyperparameter optimisation
    # ------------------------------------------------------------------

    def optimize_hyperparameters(
        self,
        adata: AnnData,
        target_column: str,
        cv: int = 3,
        n_trials: int = 20,
    ) -> dict:
        """Optimise the classifier head on SCimilarity embeddings.

        Computes reference embeddings, runs cross-validated hyperparameter
        search over the classifier head, and fits the best classifier on
        all data.  After this call the model is ready for ``predict()`` —
        no separate ``fit()`` is needed.

        Parameters
        ----------
        adata : AnnData
            Labeled reference data.
        target_column : str
            Cell-type label column in ``adata.obs``.
        cv : int, default=3
            Cross-validation folds.
        n_trials : int, default=20
            Max combinations for ``RandomizedSearchCV``.

        Returns
        -------
        best_params : dict
        """
        import pandas as pd
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        _PARAM_GRIDS = {
            'knn': {'n_neighbors': [5, 10, 15, 30, 50]},
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
            },
            'logistic_regression': {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
            'svm': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
        }

        param_grid = _PARAM_GRIDS.get(self.classifier, {})

        # Compute and cache reference embeddings
        embeddings = self.get_embedding(adata)
        labels = adata.obs[target_column].values
        valid = pd.notna(labels)
        X = embeddings[valid]
        y = labels[valid]

        # Store so predict() works after this call (no separate fit() needed)
        self._reference_embeddings = X
        self._reference_labels = y

        if not param_grid:
            logger.info(
                f"No HPO grid for classifier '{self.classifier}', "
                "fitting with current defaults."
            )
            clf = self._get_classifier()
            clf.fit(X, y)
            self._trained_classifier = clf
            self.is_trained = True
            return {}

        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)

        base_clf = self._get_classifier()
        if n_combos <= n_trials:
            search = GridSearchCV(
                base_clf, param_grid, cv=cv, n_jobs=-1,
                scoring='f1_macro', refit=True,
            )
        else:
            search = RandomizedSearchCV(
                base_clf, param_grid, n_iter=n_trials, cv=cv,
                n_jobs=-1, random_state=42,
                scoring='f1_macro', refit=True,
            )

        search.fit(X, y)
        # best_estimator_ is already fitted on all data (refit=True)
        self._trained_classifier = search.best_estimator_
        self.is_trained = True

        logger.info(
            f"SCimilarity HPO best params: {search.best_params_} "
            f"(CV F1: {search.best_score_:.4f})"
        )
        return search.best_params_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained classifier head to disk.

        Saves two files::

            <path>/classifier.joblib   — fitted sklearn classifier head
            <path>/config.json         — model config (paths, species, …)

        .. note::
            The SCimilarity **foundation model weights** at ``self.model_path``
            are *not* copied.  They must remain accessible at the original
            path when loading.

        Parameters
        ----------
        path : str
            Directory to create and populate.

        Examples
        --------
        >>> model = SCimilarityModel(model_path='/data/model_v1.1')
        >>> model.fit(adata_ref, target_column='cell_type')
        >>> model.save('/tmp/scim_knn')
        >>> # Later session (foundation weights still at /data/model_v1.1):
        >>> model = SCimilarityModel.load('/tmp/scim_knn')
        >>> predictions = model.predict(adata_query)
        """
        import joblib
        import json
        from pathlib import Path

        if not self.is_trained or self._trained_classifier is None:
            raise RuntimeError(
                "Model must be trained before saving. "
                "Call fit() or optimize_hyperparameters() first."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._trained_classifier, path / 'classifier.joblib')

        config = {
            'model_class': 'SCimilarityModel',
            'model_path': self.model_path,
            'n_neighbors': self.n_neighbors,
            'resolution': self.resolution,
            'species': self.species,
            'classifier': self.classifier,
            'classifier_params': self.classifier_params,
            'label_propagation': self.label_propagation,
            'propagation_neighbors': self.propagation_neighbors,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved SCimilarity classifier head to {path}")
        logger.info(
            f"  Foundation model weights must remain at: {self.model_path}"
        )

    @classmethod
    def load(cls, path: str) -> 'SCimilarityModel':
        """Load a saved SCimilarity classifier from disk.

        Restores the fitted classifier head.  The SCimilarity foundation
        model is lazy-loaded from the original ``model_path`` on the first
        call to ``predict()`` or ``get_embedding()``.

        Parameters
        ----------
        path : str
            Directory written by ``save()``.

        Returns
        -------
        model : SCimilarityModel
            Ready-to-predict model instance.
        """
        import joblib
        import json
        from pathlib import Path

        path = Path(path)
        with open(path / 'config.json') as f:
            config = json.load(f)

        config.pop('model_class', None)
        obj = cls(**config)
        obj._trained_classifier = joblib.load(path / 'classifier.joblib')
        obj.is_trained = True

        logger.info(f"Loaded SCimilarity classifier from {path}")
        return obj

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SCimilarityModel(classifier={self.classifier}, "
            f"n_neighbors={self.n_neighbors}, resolution={self.resolution})"
        )
