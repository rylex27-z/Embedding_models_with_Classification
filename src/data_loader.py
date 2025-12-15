"""
Data loading and preprocessing for IMDB dataset.
"""
import os
import zipfile
import random
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


def load_imdb_from_zip(
    zip_path: str,
    extract_to: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load IMDB dataset from local zip file.
    
    Expected structure inside zip:
        aclImdb/
            train/
                pos/  (12500 files)
                neg/  (12500 files)
            test/
                pos/  (12500 files)
                neg/  (12500 files)
    
    Args:
        zip_path: Path to aclImdb.zip file
        extract_to: Directory to extract to. If None, extracts to same directory as zip
        sample_size: Optional sample size per class per split (for testing/debugging)
        random_seed: Random seed for sampling
    
    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    # Extract if needed
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    aclimdb_dir = os.path.join(extract_to, "aclImdb")
    
    if not os.path.exists(aclimdb_dir):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Using existing extracted directory: {aclimdb_dir}")
    
    # Load train and test sets
    train_texts, train_labels = _load_split(
        os.path.join(aclimdb_dir, "train"),
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    test_texts, test_labels = _load_split(
        os.path.join(aclimdb_dir, "test"),
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    print(f"Loaded {len(train_texts)} train samples and {len(test_texts)} test samples")
    return train_texts, train_labels, test_texts, test_labels


def _load_split(
    split_dir: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Load one split (train or test) from directory.
    
    Args:
        split_dir: Path to split directory containing pos/ and neg/
        sample_size: Optional sample size per class
        random_seed: Random seed for sampling
    
    Returns:
        (texts, labels) where labels are 1 for positive, 0 for negative
    """
    texts = []
    labels = []
    
    # Load positive reviews
    pos_dir = os.path.join(split_dir, "pos")
    pos_files = [f for f in os.listdir(pos_dir) if f.endswith('.txt')]
    
    if sample_size is not None and sample_size < len(pos_files):
        random.seed(random_seed)
        pos_files = random.sample(pos_files, sample_size)
    
    for filename in pos_files:
        with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(1)
    
    # Load negative reviews
    neg_dir = os.path.join(split_dir, "neg")
    neg_files = [f for f in os.listdir(neg_dir) if f.endswith('.txt')]
    
    if sample_size is not None and sample_size < len(neg_files):
        random.seed(random_seed)
        neg_files = random.sample(neg_files, sample_size)
    
    for filename in neg_files:
        with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(0)
    
    return texts, labels


def load_imdb_from_directory(
    data_dir: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load IMDB dataset from already extracted directory.
    
    Args:
        data_dir: Path to aclImdb directory
        sample_size: Optional sample size per class per split
        random_seed: Random seed for sampling
    
    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    train_texts, train_labels = _load_split(
        os.path.join(data_dir, "train"),
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    test_texts, test_labels = _load_split(
        os.path.join(data_dir, "test"),
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    print(f"Loaded {len(train_texts)} train samples and {len(test_texts)} test samples")
    return train_texts, train_labels, test_texts, test_labels


def create_train_val_split(
    train_texts: List[str],
    train_labels: List[int],
    val_size: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Split training data into train and validation sets.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_size: Fraction of data to use for validation
        random_seed: Random seed
    
    Returns:
        (train_texts, train_labels, val_texts, val_labels)
    """
    return train_test_split(
        train_texts,
        train_labels,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_labels
    )


def preprocess_text(text: str, lowercase: bool = True, remove_html: bool = True) -> str:
    """
    Basic text preprocessing.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_html: Whether to remove HTML tags
    
    Returns:
        Preprocessed text
    """
    import re
    
    if remove_html:
        text = re.sub(r'<[^>]+>', ' ', text)
    
    if lowercase:
        text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_preprocess_imdb(
    zip_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    preprocess: bool = True
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load and optionally preprocess IMDB dataset.
    
    Args:
        zip_path: Path to zip file (if loading from zip)
        data_dir: Path to extracted directory (if already extracted)
        sample_size: Optional sample size per class per split
        random_seed: Random seed
        preprocess: Whether to apply text preprocessing
    
    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    # Load data
    if zip_path is not None:
        train_texts, train_labels, test_texts, test_labels = load_imdb_from_zip(
            zip_path, sample_size=sample_size, random_seed=random_seed
        )
    elif data_dir is not None:
        train_texts, train_labels, test_texts, test_labels = load_imdb_from_directory(
            data_dir, sample_size=sample_size, random_seed=random_seed
        )
    else:
        raise ValueError("Either zip_path or data_dir must be provided")
    
    # Preprocess if requested
    if preprocess:
        print("Preprocessing texts...")
        train_texts = [preprocess_text(text) for text in train_texts]
        test_texts = [preprocess_text(text) for text in test_texts]
    
    return train_texts, train_labels, test_texts, test_labels
