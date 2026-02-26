"""Traditional machine learning models using scikit-learn."""

import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .base import BaseModel

logger = logging.getLogger(__name__)

# Auto-populated by __init_subclass__ — maps class name → class object.
_SKLEARN_SUBCLASSES: dict = {}


class SklearnModel(BaseModel):
    """Base class for scikit-learn models."""

    def __init_subclass__(cls, **kwargs):
        """Register every concrete subclass for save/load dispatch."""
        super().__init_subclass__(**kwargs)
        _SKLEARN_SUBCLASSES[cls.__name__] = cls

    def __init__(self, **kwargs):
        """Initialize sklearn model."""
        super().__init__(**kwargs)
        self.classifier = None
        self.train_genes: list = []   # populated in fit(); used to align query data

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_data_matrix(self, adata: AnnData) -> np.ndarray:
        """Extract data matrix for training/prediction.

        Uses log-normalized counts if available, otherwise raw counts.
        """
        if 'log1p' in adata.uns:
            return adata.X
        elif adata.raw is not None:
            import scanpy as sc
            adata_tmp = adata.raw.to_adata()
            sc.pp.normalize_total(adata_tmp, target_sum=1e4)
            sc.pp.log1p(adata_tmp)
            return adata_tmp.X
        else:
            return adata.X

    def _to_dense(self, X) -> np.ndarray:
        if hasattr(X, 'toarray'):
            return X.toarray()
        return np.asarray(X)

    def _get_var_names(self, adata: AnnData) -> list:
        """Return gene names matching what _get_data_matrix() will return."""
        if 'log1p' in adata.uns:
            return adata.var_names.tolist()
        elif adata.raw is not None:
            return adata.raw.var_names.tolist()
        return adata.var_names.tolist()

    def _align_to_train_genes(self, X: np.ndarray, query_genes: list) -> np.ndarray:
        """Reorder/fill X columns to match the training gene order.

        If there is no gene overlap the datasets likely use incompatible naming
        conventions and the matrix is returned unchanged (the caller owns the
        decision of whether to proceed).  If overlap exists but order differs,
        columns are reordered and missing genes are filled with zeros.
        """
        if not self.train_genes:
            return X

        overlap = set(self.train_genes) & set(query_genes)
        if not overlap:
            return X  # different naming conventions — proceed as-is

        # Fast path: identical gene lists in the same order
        if self.train_genes == query_genes:
            return X

        query_idx = {g: i for i, g in enumerate(query_genes)}
        X_aligned = np.zeros((X.shape[0], len(self.train_genes)), dtype=X.dtype)
        for j, gene in enumerate(self.train_genes):
            if gene in query_idx:
                X_aligned[:, j] = X[:, query_idx[gene]]
        return X_aligned

    # ------------------------------------------------------------------
    # Hyperparameter grid (overridden by each concrete class)
    # ------------------------------------------------------------------

    def _param_grid(self) -> dict:
        """Return the hyperparameter search grid for this model.

        Each value must be a list of candidate values.
        Returns an empty dict if this model has no tunable parameters.
        """
        return {}

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        adata: AnnData,
        target_column: str,
        batch_key: Optional[str] = None,
    ) -> None:
        """Train the classifier."""
        if self.classifier is None:
            raise ValueError("Classifier not initialized")

        logger.info(f"Training {self.__class__.__name__}...")

        self.train_genes = self._get_var_names(adata)
        X = self._to_dense(self._get_data_matrix(adata))
        y = adata.obs[target_column].values
        self.classifier.fit(X, y)
        self.is_trained = True
        logger.info("Training completed")

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict labels."""
        if not self.is_trained and target_column is not None:
            self.fit(adata, target_column=target_column, batch_key=batch_key)

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X = self._to_dense(self._get_data_matrix(adata))
        X = self._align_to_train_genes(X, self._get_var_names(adata))
        return self.classifier.predict(X)

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
        """Search for best hyperparameters, then fit on all data.

        Uses ``GridSearchCV`` when the total grid size ≤ ``n_trials``,
        otherwise ``RandomizedSearchCV``.  After this call the model is
        fitted with the best found parameters and ``predict()`` can be
        called directly — no separate ``fit()`` is needed.

        Parameters
        ----------
        adata : AnnData
            Labeled training data.
        target_column : str
            Cell-type label column.
        cv : int, default=3
            Cross-validation folds.
        n_trials : int, default=20
            Maximum combinations for ``RandomizedSearchCV``.

        Returns
        -------
        best_params : dict
        """
        import pandas as pd
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.base import clone

        param_grid = self._param_grid()
        if not param_grid:
            logger.info(
                f"No hyperparameter grid defined for {self.__class__.__name__}, "
                "falling back to fit() with defaults."
            )
            self.fit(adata, target_column)
            return {}

        self.train_genes = self._get_var_names(adata)
        X = self._to_dense(self._get_data_matrix(adata))
        y = adata.obs[target_column].values

        # Drop NaN labels
        valid = pd.notna(y)
        X, y = X[valid], y[valid]

        # Count total grid combinations
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)

        base = clone(self.classifier)
        if n_combos <= n_trials:
            search = GridSearchCV(
                base, param_grid, cv=cv, n_jobs=-1,
                scoring='f1_macro', refit=True,
            )
        else:
            search = RandomizedSearchCV(
                base, param_grid, n_iter=n_trials, cv=cv,
                n_jobs=-1, random_state=42,
                scoring='f1_macro', refit=True,
            )

        search.fit(X, y)

        # best_estimator_ is already fitted on the full data (refit=True)
        self.classifier = search.best_estimator_
        self.is_trained = True

        # Keep self.params consistent with actual params used
        self.params.update(search.best_params_)

        logger.info(
            f"HPO best params: {search.best_params_} "
            f"(CV F1 macro: {search.best_score_:.4f})"
        )
        return search.best_params_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained model to disk using joblib + JSON.

        Creates two files inside *path*::

            <path>/classifier.joblib   — fitted sklearn estimator
            <path>/meta.json           — constructor params and class name

        Parameters
        ----------
        path : str
            Directory to create and write model files into.

        Examples
        --------
        >>> model = RandomForestModel()
        >>> model.fit(adata_ref, target_column='cell_type')
        >>> model.save('/tmp/my_rf')
        >>> # Later session:
        >>> model = RandomForestModel.load('/tmp/my_rf')
        >>> predictions = model.predict(adata_query)
        """
        import joblib

        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before saving. Call fit() or "
                "optimize_hyperparameters() first."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.classifier, path / 'classifier.joblib')

        meta = {
            'model_class': self.__class__.__name__,
            'params': self.params,
            'is_trained': self.is_trained,
            'train_genes': self.train_genes,
        }
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved {self.__class__.__name__} to {path}")

    @classmethod
    def load(cls, path: str) -> 'SklearnModel':
        """Load a saved model from disk.

        Can be called on ``SklearnModel`` (auto-dispatches to the correct
        subclass) or directly on any concrete subclass.

        Parameters
        ----------
        path : str
            Directory written by ``save()``.

        Returns
        -------
        model : SklearnModel
            Fitted model ready for ``predict()``.

        Examples
        --------
        >>> model = SklearnModel.load('/tmp/my_rf')          # auto-dispatch
        >>> model = RandomForestModel.load('/tmp/my_rf')     # explicit
        """
        import joblib

        path = Path(path)
        with open(path / 'meta.json') as f:
            meta = json.load(f)

        class_name = meta['model_class']
        model_cls = _SKLEARN_SUBCLASSES.get(class_name)
        if model_cls is None:
            raise ValueError(
                f"Unknown model class '{class_name}'. "
                f"Known sklearn models: {sorted(_SKLEARN_SUBCLASSES)}"
            )

        obj = model_cls(**meta['params'])
        obj.classifier = joblib.load(path / 'classifier.joblib')
        obj.is_trained = meta['is_trained']
        obj.train_genes = meta.get('train_genes', [])

        logger.info(f"Loaded {class_name} from {path}")
        return obj


# ======================================================================
# Concrete sklearn models
# ======================================================================

class RandomForestModel(SklearnModel):
    """Random Forest classifier.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    random_state : int, default=42
        Random seed
    **kwargs
        Additional parameters for RandomForestClassifier

    Examples
    --------
    >>> model = RandomForestModel(n_estimators=200, max_depth=10)
    >>> model.fit(adata_train, target_column='cell_type')
    >>> predictions = model.predict(adata_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

    def _param_grid(self) -> dict:
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
        }


class SVMModel(SklearnModel):
    """Support Vector Machine classifier.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    C : float, default=1.0
        Regularization parameter
    random_state : int, default=42
        Random seed
    **kwargs
        Additional parameters for SVC

    Examples
    --------
    >>> model = SVMModel(kernel='linear', C=0.1)
    >>> model.fit(adata_train, target_column='cell_type')
    >>> predictions = model.predict(adata_test)
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(kernel=kernel, C=C, random_state=random_state, **kwargs)
        self.classifier = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            **kwargs
        )

    def _param_grid(self) -> dict:
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto'],
        }


class LogisticRegressionModel(SklearnModel):
    """Logistic Regression classifier.

    Parameters
    ----------
    penalty : str, default='l2'
        Regularization type ('l1', 'l2', 'elasticnet', 'none')
    C : float, default=1.0
        Inverse of regularization strength
    max_iter : int, default=1000
        Maximum iterations
    random_state : int, default=42
        Random seed
    **kwargs
        Additional parameters for LogisticRegression

    Examples
    --------
    >>> model = LogisticRegressionModel(penalty='l1', solver='liblinear')
    >>> model.fit(adata_train, target_column='cell_type')
    >>> predictions = model.predict(adata_test)
    """

    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            penalty=penalty, C=C, max_iter=max_iter, random_state=random_state, **kwargs
        )
        self.classifier = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

    def _param_grid(self) -> dict:
        return {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}


class KNNModel(SklearnModel):
    """K-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors
    weights : str, default='uniform'
        Weight function ('uniform', 'distance')
    metric : str, default='euclidean'
        Distance metric
    **kwargs
        Additional parameters for KNeighborsClassifier

    Examples
    --------
    >>> model = KNNModel(n_neighbors=10, weights='distance')
    >>> model.fit(adata_train, target_column='cell_type')
    >>> predictions = model.predict(adata_test)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'euclidean',
        **kwargs
    ):
        super().__init__(
            n_neighbors=n_neighbors, weights=weights, metric=metric, **kwargs
        )
        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1,
            **kwargs
        )

    def _param_grid(self) -> dict:
        return {'n_neighbors': [3, 5, 10, 15, 30, 50]}
