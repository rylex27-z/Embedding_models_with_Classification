"""
Cross-Validation and Hyperparameter Tuning

Implements 5-fold StratifiedKFold repeated across multiple seeds (default: 4 seeds = 20 evaluations).
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class CVEvaluator:
    """Cross-validation evaluator with multiple seeds."""
    
    def __init__(
        self,
        n_folds: int = 5,
        seeds: List[int] = None,
        tune_hyperparams: bool = False,
        tuning_method: str = "grid",  # "grid" or "random"
        n_iter: int = 10,  # For random search
    ):
        """
        Initialize CV evaluator.
        
        Args:
            n_folds: Number of CV folds
            seeds: List of random seeds (default: [42, 123, 456, 789])
            tune_hyperparams: Whether to perform hyperparameter tuning
            tuning_method: "grid" for GridSearchCV or "random" for RandomizedSearchCV
            n_iter: Number of iterations for random search
        """
        self.n_folds = n_folds
        self.seeds = seeds if seeds is not None else [42, 123, 456, 789]
        self.tune_hyperparams = tune_hyperparams
        self.tuning_method = tuning_method
        self.n_iter = n_iter
    
    def evaluate(
        self,
        embedding_fn,
        model_fn,
        X_train: List[str],
        y_train: List[int],
        X_test: List[str],
        y_test: List[int],
        embedding_name: str = "unknown",
        model_name: str = "unknown",
        param_grid: Optional[Dict[str, list]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate embedding + model combination with cross-validation.
        
        Args:
            embedding_fn: Function to create embedding instance
            model_fn: Function to create model instance
            X_train: Training texts
            y_train: Training labels
            X_test: Test texts
            y_test: Test labels
            embedding_name: Name for reporting
            model_name: Name for reporting
            param_grid: Hyperparameter grid for tuning (optional)
        
        Returns:
            DataFrame with results for each fold/seed combination
        """
        results = []
        
        for seed in self.seeds:
            logger.info(f"Running CV for seed={seed}")
            
            # Create stratified k-fold
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                logger.info(f"  Fold {fold_idx + 1}/{self.n_folds}")
                
                # Split data
                X_tr = [X_train[i] for i in train_idx]
                y_tr = [y_train[i] for i in train_idx]
                X_val = [X_train[i] for i in val_idx]
                y_val = [y_train[i] for i in val_idx]
                
                # Generate embeddings
                embed_start = time.time()
                embedding = embedding_fn()
                X_tr_emb = embedding.fit_transform(X_tr)
                X_val_emb = embedding.transform(X_val)
                embed_time = time.time() - embed_start
                
                # Train model
                train_start = time.time()
                
                if self.tune_hyperparams and param_grid:
                    # Hyperparameter tuning
                    model = model_fn()
                    
                    if self.tuning_method == "grid":
                        search = GridSearchCV(
                            model, param_grid, cv=3, scoring="f1", n_jobs=-1
                        )
                    else:
                        search = RandomizedSearchCV(
                            model, param_grid, n_iter=self.n_iter,
                            cv=3, scoring="f1", n_jobs=-1, random_state=seed
                        )
                    
                    search.fit(X_tr_emb, y_tr)
                    model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    # Use default params
                    model = model_fn()
                    model.fit(X_tr_emb, y_tr)
                    best_params = {}
                
                train_time = time.time() - train_start
                
                # Predict on validation set
                infer_start = time.time()
                y_pred = model.predict(X_val_emb)
                infer_time = time.time() - infer_start
                
                # Compute metrics
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average="binary")
                precision = precision_score(y_val, y_pred, average="binary", zero_division=0)
                recall = recall_score(y_val, y_pred, average="binary", zero_division=0)
                
                # Store results
                results.append({
                    "embedding": embedding_name,
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_idx + 1,
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "train_seconds": train_time,
                    "infer_seconds": infer_time,
                    "embed_seconds": embed_time,
                    **best_params,
                })
                
                logger.info(f"    Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return pd.DataFrame(results)
    
    def evaluate_on_test(
        self,
        embedding_fn,
        model_fn,
        X_train: List[str],
        y_train: List[int],
        X_test: List[str],
        y_test: List[int],
        embedding_name: str = "unknown",
        model_name: str = "unknown",
    ) -> Dict[str, float]:
        """
        Train on full training set and evaluate on test set.
        
        Args:
            embedding_fn: Function to create embedding instance
            model_fn: Function to create model instance
            X_train: Training texts
            y_train: Training labels
            X_test: Test texts
            y_test: Test labels
            embedding_name: Name for reporting
            model_name: Name for reporting
        
        Returns:
            Dictionary with test metrics
        """
        logger.info(f"Evaluating {embedding_name} + {model_name} on test set")
        
        # Generate embeddings
        embed_start = time.time()
        embedding = embedding_fn()
        X_train_emb = embedding.fit_transform(X_train)
        X_test_emb = embedding.transform(X_test)
        embed_time = time.time() - embed_start
        
        # Train model
        train_start = time.time()
        model = model_fn()
        model.fit(X_train_emb, y_train)
        train_time = time.time() - train_start
        
        # Predict on test set
        infer_start = time.time()
        y_pred = model.predict(X_test_emb)
        infer_time = time.time() - infer_start
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
        
        logger.info(f"Test - Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return {
            "embedding": embedding_name,
            "model": model_name,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "train_seconds": train_time,
            "infer_seconds": infer_time,
            "embed_seconds": embed_time,
        }


def aggregate_cv_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CV results to get mean and std per embedding+model combination.
    
    Args:
        results_df: DataFrame with individual fold results
    
    Returns:
        DataFrame with aggregated results
    """
    metrics = ["accuracy", "f1", "precision", "recall", "train_seconds", "infer_seconds"]
    
    agg_funcs = {metric: ["mean", "std"] for metric in metrics}
    
    summary = results_df.groupby(["embedding", "model"]).agg(agg_funcs)
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary = summary.reset_index()
    
    return summary
