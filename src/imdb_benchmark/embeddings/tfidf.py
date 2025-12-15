"""
TF-IDF Embedding Generator

Converts text to TF-IDF vectors using scikit-learn.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class TfidfEmbedding:
    """TF-IDF text vectorizer with configurable parameters."""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        **kwargs
    ):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to extract (e.g., (1,2) for unigrams and bigrams)
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
            **kwargs: Additional arguments for TfidfVectorizer
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.kwargs = kwargs
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs
        )
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> 'TfidfEmbedding':
        """
        Fit the TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of text documents
        
        Returns:
            self
        """
        logger.info(f"Fitting TF-IDF with max_features={self.max_features}, "
                   f"ngram_range={self.ngram_range}")
        self.vectorizer.fit(texts)
        self._is_fitted = True
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts: List of text documents
        
        Returns:
            TF-IDF matrix (n_samples, n_features)
        """
        if not self._is_fitted:
            raise ValueError("TfidfEmbedding must be fitted before transform")
        
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform texts in one step.
        
        Args:
            texts: List of text documents
        
        Returns:
            TF-IDF matrix (n_samples, n_features)
        """
        logger.info(f"Fitting and transforming TF-IDF with max_features={self.max_features}, "
                   f"ngram_range={self.ngram_range}")
        vectors = self.vectorizer.fit_transform(texts).toarray()
        self._is_fitted = True
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Output shape: {vectors.shape}")
        return vectors
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)."""
        if not self._is_fitted:
            raise ValueError("TfidfEmbedding must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            **self.kwargs
        }


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get default TF-IDF configurations for experiments.
    
    Returns:
        Dictionary mapping config names to hyperparameters
    """
    return {
        "tfidf_unigram": {
            "max_features": 10000,
            "ngram_range": (1, 1),
        },
        "tfidf_bigram": {
            "max_features": 10000,
            "ngram_range": (1, 2),
        },
        "tfidf_trigram": {
            "max_features": 20000,
            "ngram_range": (1, 3),
        },
    }
