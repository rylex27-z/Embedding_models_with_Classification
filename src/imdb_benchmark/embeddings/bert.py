"""
BERT-based Embedding Generator

Uses Hugging Face Transformers for BERT, RoBERTa, DistilBERT, and ALBERT embeddings.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class BertEmbedding:
    """BERT-based text embedder using transformers library."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
        pooling: str = "cls",  # "cls" or "mean"
        **kwargs
    ):
        """
        Initialize BERT-based embedder.
        
        Args:
            model_name: Hugging Face model name (e.g., 'bert-base-uncased', 
                       'roberta-base', 'distilbert-base-uncased', 'albert-base-v2')
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for encoding
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            pooling: Pooling strategy ('cls' for [CLS] token, 'mean' for mean pooling)
            **kwargs: Additional arguments for model/tokenizer
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.kwargs = kwargs
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing {model_name} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        self._is_fitted = True  # No training needed for feature extraction
    
    def fit(self, texts: List[str]) -> 'BertEmbedding':
        """
        Fit method for compatibility (no-op for pre-trained models).
        
        Args:
            texts: List of text documents
        
        Returns:
            self
        """
        # Pre-trained models don't need fitting for feature extraction
        logger.info(f"Using pre-trained {self.model_name} (no training)")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to BERT embeddings.
        
        Args:
            texts: List of text documents
        
        Returns:
            Document vectors (n_samples, hidden_size)
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                if self.pooling == "cls":
                    # Use [CLS] token representation
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif self.pooling == "mean":
                    # Mean pooling over all tokens (excluding padding)
                    attention_mask = encoded["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    
                    # Mask padding tokens
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    raise ValueError(f"Invalid pooling strategy: {self.pooling}")
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step (fit is a no-op).
        
        Args:
            texts: List of text documents
        
        Returns:
            Document vectors (n_samples, hidden_size)
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "pooling": self.pooling,
            **self.kwargs
        }


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get default BERT configurations for experiments.
    
    Returns:
        Dictionary mapping config names to hyperparameters
    """
    return {
        "bert_base": {
            "model_name": "bert-base-uncased",
            "max_length": 512,
            "pooling": "cls",
        },
        "roberta_base": {
            "model_name": "roberta-base",
            "max_length": 512,
            "pooling": "cls",
        },
        "distilbert_base": {
            "model_name": "distilbert-base-uncased",
            "max_length": 512,
            "pooling": "cls",
        },
        "albert_base": {
            "model_name": "albert-base-v2",
            "max_length": 512,
            "pooling": "cls",
        },
    }
