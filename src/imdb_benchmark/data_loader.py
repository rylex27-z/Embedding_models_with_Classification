"""
IMDB Dataset Loader

Loads the IMDB sentiment classification dataset from the Stanford IMDB
folder structure (aclImdb/train/{pos,neg} and aclImdb/test/{pos,neg}).
"""

import os
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_imdb_data(
    data_path: str,
    load_train: bool = True,
    load_test: bool = True,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load IMDB dataset from the standard folder structure.
    
    Args:
        data_path: Path to the aclImdb directory
        load_train: Whether to load training data
        load_test: Whether to load test data
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
        - X_train/X_test: List of review texts
        - y_train/y_test: List of labels (0=negative, 1=positive)
    
    Example:
        >>> X_train, y_train, X_test, y_test = load_imdb_data("/path/to/aclImdb")
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data path not found: {data_path}\n"
            f"Please download and extract the IMDB dataset. "
            f"See data/README.md for instructions."
        )
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    if load_train:
        logger.info("Loading training data...")
        X_train, y_train = _load_split(data_path / "train")
        logger.info(f"Loaded {len(X_train)} training samples")
    
    if load_test:
        logger.info("Loading test data...")
        X_test, y_test = _load_split(data_path / "test")
        logger.info(f"Loaded {len(X_test)} test samples")
    
    return X_train, y_train, X_test, y_test


def _load_split(split_path: Path) -> Tuple[List[str], List[int]]:
    """
    Load a single split (train or test) of the IMDB dataset.
    
    Args:
        split_path: Path to train or test directory
    
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    # Load positive reviews
    pos_path = split_path / "pos"
    if pos_path.exists():
        for file_path in sorted(pos_path.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(1)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
    else:
        logger.warning(f"Positive reviews directory not found: {pos_path}")
    
    # Load negative reviews
    neg_path = split_path / "neg"
    if neg_path.exists():
        for file_path in sorted(neg_path.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(0)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
    else:
        logger.warning(f"Negative reviews directory not found: {neg_path}")
    
    if len(texts) == 0:
        raise ValueError(
            f"No data found in {split_path}. "
            f"Please ensure the directory structure is correct: "
            f"{split_path}/pos/ and {split_path}/neg/"
        )
    
    return texts, labels


def get_dataset_info(data_path: str) -> dict:
    """
    Get information about the IMDB dataset.
    
    Args:
        data_path: Path to the aclImdb directory
    
    Returns:
        Dictionary with dataset statistics
    """
    X_train, y_train, X_test, y_test = load_imdb_data(data_path)
    
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_pos": sum(y_train),
        "train_neg": len(y_train) - sum(y_train),
        "test_pos": sum(y_test),
        "test_neg": len(y_test) - sum(y_test),
        "total_size": len(X_train) + len(X_test),
    }
