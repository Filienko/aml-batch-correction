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


__all__ = ['BaseModel', 'get_model', 'AVAILABLE_MODELS']
