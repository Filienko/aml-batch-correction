"""Model implementations for single cell classification."""

from .base import BaseModel
from .scimilarity import SCimilarityModel
from .sklearn import RandomForestModel, SVMModel, LogisticRegressionModel, KNNModel

# Model registry
AVAILABLE_MODELS = {
    'scimilarity': SCimilarityModel,
    'random_forest': RandomForestModel,
    'svm': SVMModel,
    'logistic_regression': LogisticRegressionModel,
    'knn': KNNModel,
}

# Try to import optional models
try:
    from .scvi import ScVIModel
    AVAILABLE_MODELS['scvi'] = ScVIModel
except ImportError:
    pass

try:
    from .celltypist import CellTypistModel
    AVAILABLE_MODELS['celltypist'] = CellTypistModel
except ImportError:
    pass

try:
    from .sctab import ScTabModel
    AVAILABLE_MODELS['sctab'] = ScTabModel
except ImportError:
    pass


def get_model(model_name: str, **kwargs) -> BaseModel:
    """Get a model instance by name.

    Parameters
    ----------
    model_name : str
        Name of the model
    **kwargs
        Additional parameters for the model

    Returns
    -------
    model : BaseModel
        Model instance
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    model_class = AVAILABLE_MODELS[model_name]
    return model_class(**kwargs)


def load_model(path: str) -> BaseModel:
    """Load a saved SCCL model from disk (auto-detects model type).

    Parameters
    ----------
    path : str
        Directory created by ``model.save()``.

    Returns
    -------
    model : BaseModel
        Ready-to-predict model instance.

    Examples
    --------
    >>> from sccl.models import load_model
    >>> model = load_model('/tmp/my_rf_model')
    >>> predictions = model.predict(adata_query)
    """
    import json
    from pathlib import Path
    from .sklearn import SklearnModel
    from .scimilarity import SCimilarityModel

    path = Path(path)

    if (path / 'meta.json').exists():
        # Sklearn-family model
        with open(path / 'meta.json') as f:
            meta = json.load(f)
        class_name = meta.get('model_class', '')
        if class_name == 'SCimilarityModel':
            return SCimilarityModel.load(path)
        return SklearnModel.load(path)

    if (path / 'config.json').exists():
        # SCimilarity model
        return SCimilarityModel.load(path)

    raise ValueError(
        f"No recognised model files found in '{path}'. "
        "Expected 'meta.json' (sklearn models) or 'config.json' (SCimilarity)."
    )


__all__ = ['BaseModel', 'get_model', 'load_model', 'AVAILABLE_MODELS']
