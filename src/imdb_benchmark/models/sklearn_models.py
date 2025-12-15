"""
Scikit-learn Models

Logistic Regression, Random Forest, and AdaBoost classifiers with hyperparameter search spaces.
"""

from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)


def get_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    **kwargs
) -> LogisticRegression:
    """
    Create Logistic Regression classifier.
    
    Args:
        C: Inverse of regularization strength
        max_iter: Maximum number of iterations
        random_state: Random seed
        **kwargs: Additional arguments
    
    Returns:
        LogisticRegression instance
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        **kwargs
    )


def get_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: Optional[int] = None,
    **kwargs
) -> RandomForestClassifier:
    """
    Create Random Forest classifier.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        random_state: Random seed
        **kwargs: Additional arguments
    
    Returns:
        RandomForestClassifier instance
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
        **kwargs
    )


def get_adaboost(
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    random_state: Optional[int] = None,
    **kwargs
) -> AdaBoostClassifier:
    """
    Create AdaBoost classifier.
    
    Args:
        n_estimators: Number of boosting stages
        learning_rate: Weight applied to each classifier
        random_state: Random seed
        **kwargs: Additional arguments
    
    Returns:
        AdaBoostClassifier instance
    """
    return AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        **kwargs
    )


def get_hyperparameter_grid(model_name: str) -> Dict[str, list]:
    """
    Get hyperparameter search space for a model.
    
    Args:
        model_name: Name of the model (logreg, rf, adaboost)
    
    Returns:
        Dictionary of hyperparameter grids
    """
    grids = {
        "logreg": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "rf": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "adaboost": {
            "n_estimators": [25, 50, 100],
            "learning_rate": [0.5, 1.0, 1.5],
        },
    }
    
    return grids.get(model_name, {})


def get_default_params(model_name: str) -> Dict[str, Any]:
    """
    Get default parameters for a model.
    
    Args:
        model_name: Name of the model (logreg, rf, adaboost)
    
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        "logreg": {
            "C": 1.0,
            "max_iter": 1000,
        },
        "rf": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
        },
        "adaboost": {
            "n_estimators": 50,
            "learning_rate": 1.0,
        },
    }
    
    return defaults.get(model_name, {})
