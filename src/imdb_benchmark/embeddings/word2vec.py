"""
Word2Vec Embedding Generator

Trains Word2Vec models using Gensim and generates document vectors via mean pooling.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import logging

logger = logging.getLogger(__name__)


class Word2VecEmbedding:
    """Word2Vec text vectorizer with CBOW and Skip-gram support."""
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        sg: int = 0,  # 0 for CBOW, 1 for Skip-gram
        epochs: int = 10,
        workers: int = 4,
        **kwargs
    ):
        """
        Initialize Word2Vec model.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word frequency
            sg: Training algorithm (0=CBOW, 1=Skip-gram)
            epochs: Number of training epochs
            workers: Number of worker threads
            **kwargs: Additional arguments for Word2Vec
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.workers = workers
        self.kwargs = kwargs
        
        self.model = None
        self._is_fitted = False
    
    def _preprocess(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts into tokenized sentences.
        
        Args:
            texts: List of text documents
        
        Returns:
            List of tokenized documents
        """
        return [simple_preprocess(text) for text in texts]
    
    def fit(self, texts: List[str]) -> 'Word2VecEmbedding':
        """
        Train Word2Vec model on texts.
        
        Args:
            texts: List of text documents
        
        Returns:
            self
        """
        algorithm = "CBOW" if self.sg == 0 else "Skip-gram"
        logger.info(f"Training Word2Vec ({algorithm}) with vector_size={self.vector_size}, "
                   f"window={self.window}, epochs={self.epochs}")
        
        # Preprocess texts
        sentences = self._preprocess(texts)
        
        # Train Word2Vec
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            workers=self.workers,
            **self.kwargs
        )
        
        self._is_fitted = True
        logger.info(f"Vocabulary size: {len(self.model.wv)}")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to document vectors via mean pooling.
        
        Args:
            texts: List of text documents
        
        Returns:
            Document vectors (n_samples, vector_size)
        """
        if not self._is_fitted:
            raise ValueError("Word2VecEmbedding must be fitted before transform")
        
        sentences = self._preprocess(texts)
        vectors = []
        
        for tokens in sentences:
            # Get word vectors for tokens in vocabulary
            word_vecs = [
                self.model.wv[word]
                for word in tokens
                if word in self.model.wv
            ]
            
            # Mean pooling (or zero vector if no words in vocabulary)
            if word_vecs:
                doc_vec = np.mean(word_vecs, axis=0)
            else:
                doc_vec = np.zeros(self.vector_size)
            
            vectors.append(doc_vec)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit Word2Vec model and transform texts in one step.
        
        Args:
            texts: List of text documents
        
        Returns:
            Document vectors (n_samples, vector_size)
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "sg": self.sg,
            "epochs": self.epochs,
            "workers": self.workers,
            **self.kwargs
        }


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get default Word2Vec configurations for experiments.
    
    Returns:
        Dictionary mapping config names to hyperparameters
    """
    return {
        "w2v_cbow": {
            "vector_size": 100,
            "window": 5,
            "sg": 0,  # CBOW
            "epochs": 10,
        },
        "w2v_skipgram": {
            "vector_size": 100,
            "window": 5,
            "sg": 1,  # Skip-gram
            "epochs": 10,
        },
        "w2v_cbow_300d": {
            "vector_size": 300,
            "window": 5,
            "sg": 0,
            "epochs": 10,
        },
    }
