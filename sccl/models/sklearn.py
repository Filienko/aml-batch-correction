"""Traditional machine learning models using scikit-learn."""

import logging
from typing import Optional
import numpy as np
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .base import BaseModel

logger = logging.getLogger(__name__)


class SklearnModel(BaseModel):
    """Base class for scikit-learn models."""

    def __init__(self, **kwargs):
        """Initialize sklearn model."""
        super().__init__(**kwargs)
        self.classifier = None

    def _get_data_matrix(self, adata: AnnData) -> np.ndarray:
        """Extract data matrix for training/prediction.

        Uses log-normalized counts if available, otherwise raw counts.

        Parameters
        ----------
        adata : AnnData
            Input data

        Returns
        -------
        X : np.ndarray
            Data matrix
        """
        # Prefer normalized data in .X
        if 'log1p' in adata.uns:
            return adata.X
        # Otherwise normalize on the fly
        elif adata.raw is not None:
            import scanpy as sc
            adata_tmp = adata.raw.to_adata()
            sc.pp.normalize_total(adata_tmp, target_sum=1e4)
            sc.pp.log1p(adata_tmp)
            return adata_tmp.X
        else:
            return adata.X

    def fit(
        self,
        adata: AnnData,
        target_column: str,
        batch_key: Optional[str] = None,
    ) -> None:
        """Train the classifier.

        Parameters
        ----------
        adata : AnnData
            Training data
        target_column : str
            Column with ground truth labels
        batch_key : str, optional
            Not used for basic sklearn models
        """
        if self.classifier is None:
            raise ValueError("Classifier not initialized")

        logger.info(f"Training {self.__class__.__name__}...")

        # Get data matrix
        X = self._get_data_matrix(adata)
        y = adata.obs[target_column].values

        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Train
        self.classifier.fit(X, y)
        self.is_trained = True

        logger.info("Training completed")

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        batch_key: Optional[str] = None,
    ) -> np.ndarray:
        """Predict labels.

        Parameters
        ----------
        adata : AnnData
            Data to predict on
        target_column : str, optional
            If provided and model not trained, will train first
        batch_key : str, optional
            Not used

        Returns
        -------
        predictions : np.ndarray
            Predicted labels
        """
        # Train if not trained and labels provided
        if not self.is_trained and target_column is not None:
            self.fit(adata, target_column=target_column, batch_key=batch_key)

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Get data matrix
        X = self._get_data_matrix(adata)

        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Predict
        predictions = self.classifier.predict(X)

        return predictions


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
        """Initialize Random Forest model."""
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
            n_jobs=-1,  # Use all cores
            **kwargs
        )


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
        """Initialize SVM model."""
        super().__init__(kernel=kernel, C=C, random_state=random_state, **kwargs)
        self.classifier = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            **kwargs
        )


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
        """Initialize Logistic Regression model."""
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
        """Initialize KNN model."""
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
