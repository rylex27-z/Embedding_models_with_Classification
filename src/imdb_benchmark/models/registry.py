"""
Model Registry

Maps model configuration names to model classes and parameters.
"""

from typing import Dict, Any, Optional
from .sklearn_models import (
    get_logistic_regression,
    get_random_forest,
    get_adaboost,
    get_default_params as get_sklearn_default_params,
    get_hyperparameter_grid as get_sklearn_grid,
)
from .lstm import (
    LSTMTextClassifier,
    get_default_params as get_lstm_default_params,
    get_hyperparameter_grid as get_lstm_grid,
)


MODEL_CONSTRUCTORS = {
    "logreg": get_logistic_regression,
    "rf": get_random_forest,
    "adaboost": get_adaboost,
    "lstm": LSTMTextClassifier,
}


def create_model(model_name: str, random_state: Optional[int] = None, **params) -> Any:
    """
    Create a model instance.
    
    Args:
        model_name: Name of the model (logreg, rf, adaboost, lstm)
        random_state: Random seed for reproducibility
        **params: Model hyperparameters
    
    Returns:
        Model instance
    
    Example:
        >>> model = create_model("logreg", C=1.0, random_state=42)
        >>> model = create_model("lstm", hidden_dim=128, random_state=42)
    """
    if model_name not in MODEL_CONSTRUCTORS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_CONSTRUCTORS.keys())}"
        )
    
    constructor = MODEL_CONSTRUCTORS[model_name]
    
    # Add random_state if not provided
    if random_state is not None and "random_state" not in params:
        params["random_state"] = random_state
    
    return constructor(**params)


def get_default_params(model_name: str) -> Dict[str, Any]:
    """
    Get default parameters for a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary of default parameters
    """
    if model_name in ["logreg", "rf", "adaboost"]:
        return get_sklearn_default_params(model_name)
    elif model_name == "lstm":
        return get_lstm_default_params()
    else:
        return {}


def get_hyperparameter_grid(model_name: str) -> Dict[str, list]:
    """
    Get hyperparameter search space for a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary of hyperparameter grids
    """
    if model_name in ["logreg", "rf", "adaboost"]:
        return get_sklearn_grid(model_name)
    elif model_name == "lstm":
        return get_lstm_grid()
    else:
        return {}


def list_models() -> Dict[str, list]:
    """
    List all available models grouped by type.
    
    Returns:
        Dictionary mapping model types to model names
    """
    return {
        "sklearn": ["logreg", "rf", "adaboost"],
        "pytorch": ["lstm"],
    }
