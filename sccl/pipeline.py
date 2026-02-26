"""Main Pipeline class for single cell classification."""

import logging
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

from .models import get_model, AVAILABLE_MODELS
from .data.preprocessing import preprocess_data, subset_data
from .evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline for single cell classification and batch correction.

    This class provides a unified interface for:
    - Loading and preprocessing single cell data
    - Running different classification models
    - Evaluating predictions
    - Comparing models

    Parameters
    ----------
    model : str
        Model name (e.g., 'scimilarity', 'random_forest', 'scvi')
    batch_key : str, optional
        Column name for batch information (for batch correction)
    preprocess : bool, default=True
        Whether to apply standard preprocessing
    model_params : dict, optional
        Additional parameters to pass to the model

    Examples
    --------
    >>> from sccl import Pipeline
    >>> import scanpy as sc
    >>>
    >>> # Load data
    >>> adata = sc.read_h5ad("data.h5ad")
    >>>
    >>> # Create pipeline
    >>> pipeline = Pipeline(model="scimilarity")
    >>>
    >>> # Predict
    >>> predictions = pipeline.predict(adata, target_column="cell_type")
    >>>
    >>> # Evaluate
    >>> metrics = pipeline.evaluate(adata, target_column="cell_type", test_size=0.2)
    """

    def __init__(
        self,
        model: str,
        batch_key: Optional[str] = None,
        preprocess: bool = True,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the pipeline."""
        if model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model
        self.batch_key = batch_key
        self.preprocess = preprocess
        self.model_params = model_params or {}
        self.random_state = random_state

        # Initialize model
        self.model = get_model(model, **self.model_params)

        logger.info(f"Initialized pipeline with model: {model}")

    def predict(
        self,
        adata: AnnData,
        target_column: Optional[str] = None,
        subset_params: Optional[Dict[str, Any]] = None,
        return_embedding: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Predict target labels for single cell data.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        target_column : str, optional
            Column to predict. If provided and data is labeled, will use for training.
            If None, performs unsupervised clustering.
        subset_params : dict, optional
            Parameters for subsetting data (e.g., {'studies': ['study1', 'study2']})
        return_embedding : bool, default=False
            Whether to return embeddings in addition to predictions

        Returns
        -------
        predictions : np.ndarray or dict
            Predicted labels, or dict with 'predictions' and 'embedding' if return_embedding=True
        """
        logger.info("Starting prediction...")

        # Subset data if requested
        if subset_params:
            adata = subset_data(adata, **subset_params)
            logger.info(f"Subset to {adata.n_obs} cells")

        # Preprocess
        if self.preprocess:
            adata = preprocess_data(adata, batch_key=self.batch_key)
            logger.info("Preprocessing completed")

        # Predict
        result = self.model.predict(adata, target_column=target_column, batch_key=self.batch_key)

        if return_embedding and hasattr(self.model, 'get_embedding'):
            embedding = self.model.get_embedding(adata, batch_key=self.batch_key)
            return {'predictions': result, 'embedding': embedding}

        logger.info("Prediction completed")
        return result

    def evaluate(
        self,
        adata: AnnData,
        target_column: str,
        test_size: float = 0.2,
        subset_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate model performance with train/test split.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with ground truth labels
        target_column : str
            Column containing ground truth labels
        test_size : float, default=0.2
            Fraction of data to use for testing
        subset_params : dict, optional
            Parameters for subsetting data
        metrics : list of str, optional
            Metrics to compute. Default: ['accuracy', 'ari', 'nmi', 'f1']
        random_state : int, optional
            Seed for the train/test split.  Falls back to the value passed to
            ``Pipeline.__init__`` (itself defaulting to ``None``, which lets
            numpy's global seed govern the split — important for multi-run
            statistical evaluation).

        Returns
        -------
        results : dict
            Dictionary mapping metric names to values
        """
        logger.info("Starting evaluation...")

        # Subset if requested
        if subset_params:
            adata = subset_data(adata, **subset_params)

        # Resolve random state: explicit arg > instance default > None
        rs = random_state if random_state is not None else self.random_state

        # Split data
        from sklearn.model_selection import train_test_split

        indices = np.arange(adata.n_obs)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=rs,
            stratify=adata.obs[target_column] if target_column in adata.obs else None
        )

        # Train on training set
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        # Preprocess
        if self.preprocess:
            adata_train = preprocess_data(adata_train, batch_key=self.batch_key)
            adata_test = preprocess_data(adata_test, batch_key=self.batch_key)

        # Train (if model requires training)
        if hasattr(self.model, 'fit'):
            logger.info("Training model...")
            self.model.fit(adata_train, target_column=target_column, batch_key=self.batch_key)

        # Predict on test set
        predictions = self.model.predict(adata_test, target_column=None, batch_key=self.batch_key)

        # Get ground truth
        y_true = adata_test.obs[target_column].values

        # Compute metrics
        results = compute_metrics(
            y_true=y_true,
            y_pred=predictions,
            adata=adata_test,
            metrics=metrics,
        )

        logger.info("Evaluation completed")
        logger.info(f"Results: {results}")

        return results

    def optimize_hyperparameters(
        self,
        adata: AnnData,
        target_column: str,
        cv: int = 3,
        n_trials: int = 20,
    ) -> tuple:
        """Find the best hyperparameters and fit on all provided data.

        Runs a cross-validated hyperparameter search on the labelled reference
        data.  After this call the internal model is fitted with the best found
        parameters and ``predict()`` can be called directly — no separate
        ``fit()`` is needed.

        Parameters
        ----------
        adata : AnnData
            Labeled reference data.
        target_column : str
            Cell-type label column in ``adata.obs``.
        cv : int, default=3
            Number of cross-validation folds.
        n_trials : int, default=20
            Maximum number of hyperparameter combinations to try.

        Returns
        -------
        best_params : dict
            Best hyperparameters found by the search.
        elapsed_sec : float
            Wall-clock time taken by the search (seconds).

        Examples
        --------
        >>> pipeline = Pipeline(model='random_forest')
        >>> best_params, hpo_time = pipeline.optimize_hyperparameters(
        ...     adata_ref, target_column='cell_type'
        ... )
        >>> predictions = pipeline.model.predict(adata_query)
        """
        import time

        if self.preprocess:
            adata = preprocess_data(adata.copy(), batch_key=self.batch_key)

        t0 = time.time()
        best_params = self.model.optimize_hyperparameters(
            adata, target_column, cv=cv, n_trials=n_trials,
        )
        elapsed = time.time() - t0

        logger.info(f"HPO completed in {elapsed:.1f}s. Best params: {best_params}")
        return best_params, elapsed

    def compare_models(
        self,
        adata: AnnData,
        target_column: str,
        models: List[str],
        test_size: float = 0.2,
        subset_params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compare multiple models on the same data.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        target_column : str
            Column to predict
        models : list of str
            List of model names to compare
        test_size : float, default=0.2
            Fraction of data for testing
        subset_params : dict, optional
            Parameters for subsetting data

        Returns
        -------
        comparison : pd.DataFrame
            DataFrame with models as rows and metrics as columns
        """
        logger.info(f"Comparing models: {models}")

        results = []
        for model_name in models:
            logger.info(f"\nEvaluating {model_name}...")

            # Create pipeline for this model
            pipeline = Pipeline(
                model=model_name,
                batch_key=self.batch_key,
                preprocess=self.preprocess,
            )

            # Evaluate
            metrics = pipeline.evaluate(
                adata=adata.copy(),
                target_column=target_column,
                test_size=test_size,
                subset_params=subset_params,
            )

            metrics['model'] = model_name
            results.append(metrics)

        # Create comparison dataframe
        df = pd.DataFrame(results)
        df = df.set_index('model')

        logger.info("\n" + "="*50)
        logger.info("COMPARISON RESULTS")
        logger.info("="*50)
        logger.info("\n" + str(df))

        return df


def run_pipeline_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run pipeline from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - 'data': data loading parameters
        - 'models': list of models to run
        - 'evaluation': evaluation parameters

    Returns
    -------
    results : dict
        Dictionary with results for each model
    """
    logger.info("Running pipeline from configuration")

    # Load data
    data_config = config['data']
    adata = sc.read_h5ad(data_config['path'])
    logger.info(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")

    # Get parameters
    target_column = data_config['target_column']
    batch_column = data_config.get('batch_column')
    subset_params = data_config.get('subset')

    eval_config = config.get('evaluation', {})
    test_size = eval_config.get('test_size', 0.2)

    # Get models
    models_config = config['models']
    if isinstance(models_config, list):
        model_names = [m if isinstance(m, str) else list(m.keys())[0] for m in models_config]
    else:
        model_names = [models_config]

    # Run comparison
    pipeline = Pipeline(model=model_names[0], batch_key=batch_column)

    comparison = pipeline.compare_models(
        adata=adata,
        target_column=target_column,
        models=model_names,
        test_size=test_size,
        subset_params=subset_params,
    )

    return {
        'comparison': comparison,
        'config': config,
    }
