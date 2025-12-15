"""
Text embedding methods: TF-IDF, Word2Vec (CBOW/Skip-gram), BERT-family.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class TfidfEmbedding:
    """TF-IDF embedding."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2), **kwargs):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (e.g., (1, 2) for unigrams and bigrams)
            **kwargs: Additional arguments for TfidfVectorizer
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            **kwargs
        )
        self.variant = f"max_features_{max_features}_ngram_{ngram_range[0]}_{ngram_range[1]}"
    
    def fit(self, texts: List[str]) -> 'TfidfEmbedding':
        """Fit vectorizer on texts."""
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors."""
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to TF-IDF vectors."""
        return self.vectorizer.fit_transform(texts).toarray()


class Word2VecEmbedding:
    """Word2Vec embedding (CBOW or Skip-gram)."""
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 4,
        sg: int = 0,  # 0 for CBOW, 1 for Skip-gram
        epochs: int = 10,
        **kwargs
    ):
        """
        Initialize Word2Vec model.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Ignores words with frequency lower than this
            workers: Number of worker threads
            sg: Training algorithm: 0 for CBOW, 1 for Skip-gram
            epochs: Number of training epochs
            **kwargs: Additional arguments for Word2Vec
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.model = None
        self.variant = f"{'skipgram' if sg == 1 else 'cbow'}_dim_{vector_size}_window_{window}"
    
    def fit(self, texts: List[str]) -> 'Word2VecEmbedding':
        """
        Fit Word2Vec model on texts.
        
        Args:
            texts: List of text documents
        """
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs
        )
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to document vectors by averaging word vectors.
        
        Args:
            texts: List of text documents
        
        Returns:
            Document vectors (n_documents, vector_size)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        doc_vectors = []
        for text in texts:
            words = text.split()
            word_vecs = []
            for word in words:
                if word in self.model.wv:
                    word_vecs.append(self.model.wv[word])
            
            if word_vecs:
                doc_vectors.append(np.mean(word_vecs, axis=0))
            else:
                # If no words found in vocabulary, use zero vector
                doc_vectors.append(np.zeros(self.vector_size))
        
        return np.array(doc_vectors)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to Word2Vec vectors."""
        self.fit(texts)
        return self.transform(texts)


class BertEmbedding:
    """BERT-family embedding."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 8,
        device: Optional[str] = None,
        pooling: str = "cls"  # "cls" or "mean"
    ):
        """
        Initialize BERT-based model.
        
        Args:
            model_name: HuggingFace model name (e.g., "bert-base-uncased", "roberta-base")
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
            device: Device to use ("cuda" or "cpu"). If None, auto-detect.
            pooling: Pooling strategy: "cls" for [CLS] token, "mean" for mean pooling
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load tokenizer and model
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Extract variant name from model_name
        self.variant = model_name.replace("/", "_")
    
    def fit(self, texts: List[str]) -> 'BertEmbedding':
        """No fitting needed for BERT."""
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to BERT embeddings.
        
        Args:
            texts: List of text documents
        
        Returns:
            Document embeddings (n_documents, hidden_size)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
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
                outputs = self.model(**encoded)
                
                if self.pooling == "cls":
                    # Use [CLS] token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif self.pooling == "mean":
                    # Mean pooling over all tokens
                    attention_mask = encoded['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to BERT embeddings (no fitting needed)."""
        return self.transform(texts)


def get_embedding(
    embedding_type: str,
    variant: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Factory function to get embedding object.
    
    Args:
        embedding_type: Type of embedding ("tfidf", "word2vec", "bert")
        variant: Variant specification (e.g., "cbow", "skipgram", "bert-base-uncased")
        config: Configuration dictionary for embedding parameters
    
    Returns:
        Embedding object
    """
    if config is None:
        config = {}
    
    if embedding_type.lower() == "tfidf":
        return TfidfEmbedding(**config)
    
    elif embedding_type.lower() == "word2vec":
        if variant and "skipgram" in variant.lower():
            config['sg'] = 1
        else:
            config['sg'] = 0
        return Word2VecEmbedding(**config)
    
    elif embedding_type.lower() == "bert":
        if variant:
            config['model_name'] = variant
        return BertEmbedding(**config)
    
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
