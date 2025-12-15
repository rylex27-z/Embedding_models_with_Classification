"""
Evaluation module with cross-validation and metrics computation.
"""
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0)
    }


def cross_validate_with_repeats(
    embedding_obj,
    model_obj,
    X_train: List[str],
    y_train: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 4,
    random_seeds: Optional[List[int]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Perform stratified k-fold cross-validation with multiple repeats.
    
    This implements the protocol: StratifiedKFold with n_splits repeated across
    n_repeats different seeds (total n_splits * n_repeats evaluations).
    
    Args:
        embedding_obj: Embedding object with fit_transform method
        model_obj: Model object with fit and predict methods
        X_train: Training texts
        y_train: Training labels
        n_splits: Number of folds (default 5)
        n_repeats: Number of repeats with different seeds (default 4)
        random_seeds: List of random seeds (if None, uses [42, 123, 456, 789])
        verbose: Whether to print progress
    
    Returns:
        (results_df, summary_dict) where results_df contains per-fold results
        and summary_dict contains mean±std statistics
    """
    if random_seeds is None:
        random_seeds = [42, 123, 456, 789][:n_repeats]
    
    if len(random_seeds) < n_repeats:
        raise ValueError(f"Need {n_repeats} random seeds, got {len(random_seeds)}")
    
    total_folds = n_splits * n_repeats
    
    if verbose:
        print(f"Running {n_splits}-fold CV × {n_repeats} repeats = {total_folds} total evaluations")
    
    results = []
    fold_idx = 0
    
    for repeat_idx, seed in enumerate(random_seeds[:n_repeats]):
        if verbose:
            print(f"\n=== Repeat {repeat_idx + 1}/{n_repeats} (seed={seed}) ===")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_idx += 1
            
            if verbose:
                print(f"Fold {fold + 1}/{n_splits} (overall {fold_idx}/{total_folds})")
            
            # Split data
            X_train_fold = [X_train[i] for i in train_idx]
            X_val_fold = [X_train[i] for i in val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            # Time the fold
            start_time = time.time()
            
            # Fit embedding and transform
            X_train_emb = embedding_obj.fit_transform(X_train_fold)
            X_val_emb = embedding_obj.transform(X_val_fold)
            
            # Fit model and predict
            model_obj.fit(X_train_emb, y_train_fold)
            y_pred = model_obj.predict(X_val_emb)
            
            fold_time = time.time() - start_time
            
            # Compute metrics
            metrics = compute_metrics(y_val_fold, y_pred)
            
            # Store results
            result = {
                'repeat': repeat_idx,
                'seed': seed,
                'fold': fold,
                'overall_fold': fold_idx,
                'runtime': fold_time,
                **metrics
            }
            results.append(result)
            
            if verbose:
                print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
                      f"Time: {fold_time:.2f}s")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute summary statistics
    summary = {
        'n_folds': total_folds,
        'accuracy_mean': results_df['accuracy'].mean(),
        'accuracy_std': results_df['accuracy'].std(),
        'f1_mean': results_df['f1'].mean(),
        'f1_std': results_df['f1'].std(),
        'precision_mean': results_df['precision'].mean(),
        'precision_std': results_df['precision'].std(),
        'recall_mean': results_df['recall'].mean(),
        'recall_std': results_df['recall'].std(),
        'runtime_mean': results_df['runtime'].mean(),
        'runtime_std': results_df['runtime'].std(),
    }
    
    if verbose:
        print(f"\n=== Summary ({total_folds} folds) ===")
        print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        print(f"F1 Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        print(f"Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        print(f"Runtime: {summary['runtime_mean']:.2f} ± {summary['runtime_std']:.2f}s")
    
    return results_df, summary


def evaluate_on_test(
    embedding_obj,
    model_obj,
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Train on full training set and evaluate on test set.
    
    Args:
        embedding_obj: Embedding object
        model_obj: Model object
        X_train: Training texts
        y_train: Training labels
        X_test: Test texts
        y_test: Test labels
        verbose: Whether to print progress
    
    Returns:
        Dictionary of test metrics
    """
    if verbose:
        print("Training on full training set and evaluating on test set...")
    
    start_time = time.time()
    
    # Fit embedding and transform
    X_train_emb = embedding_obj.fit_transform(X_train)
    X_test_emb = embedding_obj.transform(X_test)
    
    # Fit model and predict
    model_obj.fit(X_train_emb, y_train)
    y_pred = model_obj.predict(X_test_emb)
    
    runtime = time.time() - start_time
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    metrics['runtime'] = runtime
    
    if verbose:
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1: {metrics['f1']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Runtime: {runtime:.2f}s")
    
    return metrics


def run_full_evaluation(
    embedding_obj,
    model_obj,
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    embedding_type: str,
    embedding_variant: str,
    model_type: str,
    cv_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full evaluation: CV on training set and final evaluation on test set.
    
    Args:
        embedding_obj: Embedding object
        model_obj: Model object
        X_train: Training texts
        y_train: Training labels
        X_test: Test texts
        y_test: Test labels
        embedding_type: Name of embedding type
        embedding_variant: Name of embedding variant
        model_type: Name of model type
        cv_config: Configuration for cross-validation (n_splits, n_repeats, random_seeds)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with all results
    """
    if cv_config is None:
        cv_config = {'n_splits': 5, 'n_repeats': 4}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating: {embedding_type} ({embedding_variant}) + {model_type}")
        print(f"{'='*80}\n")
    
    # Cross-validation
    cv_results_df, cv_summary = cross_validate_with_repeats(
        embedding_obj=embedding_obj,
        model_obj=model_obj,
        X_train=X_train,
        y_train=y_train,
        verbose=verbose,
        **cv_config
    )
    
    # Test evaluation
    test_metrics = evaluate_on_test(
        embedding_obj=embedding_obj,
        model_obj=model_obj,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        verbose=verbose
    )
    
    return {
        'embedding_type': embedding_type,
        'embedding_variant': embedding_variant,
        'model_type': model_type,
        'cv_results': cv_results_df,
        'cv_summary': cv_summary,
        'test_metrics': test_metrics
    }
