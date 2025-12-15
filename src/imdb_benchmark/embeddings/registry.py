"""
Embedding Registry

Maps embedding configuration names to embedding classes and parameters.
"""

from typing import Dict, Any, Type
from .tfidf import TfidfEmbedding, get_default_configs as get_tfidf_configs
from .word2vec import Word2VecEmbedding, get_default_configs as get_w2v_configs
from .bert import BertEmbedding, get_default_configs as get_bert_configs


EMBEDDING_CLASSES = {
    "tfidf": TfidfEmbedding,
    "word2vec": Word2VecEmbedding,
    "bert": BertEmbedding,
}


def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get all default embedding configurations.
    
    Returns:
        Dictionary mapping config names to (class_key, params)
    """
    configs = {}
    
    # TF-IDF configs
    for name, params in get_tfidf_configs().items():
        configs[name] = {"embedding_class": "tfidf", **params}
    
    # Word2Vec configs
    for name, params in get_w2v_configs().items():
        configs[name] = {"embedding_class": "word2vec", **params}
    
    # BERT configs
    for name, params in get_bert_configs().items():
        configs[name] = {"embedding_class": "bert", **params}
    
    return configs


def create_embedding(embedding_name: str, **override_params) -> Any:
    """
    Create an embedding instance from a config name or class name.
    
    Args:
        embedding_name: Name from get_all_configs() or class key (tfidf/word2vec/bert)
        **override_params: Parameters to override defaults
    
    Returns:
        Embedding instance
    
    Example:
        >>> emb = create_embedding("tfidf_bigram")
        >>> emb = create_embedding("bert", model_name="roberta-base")
    """
    all_configs = get_all_configs()
    
    if embedding_name in all_configs:
        # Use predefined config
        config = all_configs[embedding_name].copy()
        embedding_class_key = config.pop("embedding_class")
        
        # Override with custom params
        config.update(override_params)
        
        embedding_class = EMBEDDING_CLASSES[embedding_class_key]
        return embedding_class(**config)
    
    elif embedding_name in EMBEDDING_CLASSES:
        # Use class directly with override params
        embedding_class = EMBEDDING_CLASSES[embedding_name]
        return embedding_class(**override_params)
    
    else:
        available = list(all_configs.keys()) + list(EMBEDDING_CLASSES.keys())
        raise ValueError(
            f"Unknown embedding: {embedding_name}. "
            f"Available options: {available}"
        )


def list_embeddings() -> Dict[str, list]:
    """
    List all available embeddings grouped by type.
    
    Returns:
        Dictionary mapping embedding types to config names
    """
    all_configs = get_all_configs()
    
    grouped = {
        "tfidf": [],
        "word2vec": [],
        "bert": [],
    }
    
    for name, config in all_configs.items():
        embedding_class = config["embedding_class"]
        grouped[embedding_class].append(name)
    
    return grouped
