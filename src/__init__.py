"""
IMDB Sentiment Classification Benchmark Package
"""

__version__ = "0.1.0"

from .data_loader import (
    load_imdb_from_zip,
    load_imdb_from_directory,
    load_and_preprocess_imdb,
    preprocess_text
)

from .embeddings import (
    TfidfEmbedding,
    Word2VecEmbedding,
    BertEmbedding,
    get_embedding
)

from .models import (
    LSTMClassifier,
    LSTMClassifierWrapper,
    get_model
)

from .evaluation import (
    compute_metrics,
    cross_validate_with_repeats,
    evaluate_on_test,
    run_full_evaluation
)

from .utils import (
    load_config,
    save_config,
    get_environment_info,
    save_results_csv,
    load_results_csv,
    format_results_table
)

__all__ = [
    # Data loading
    'load_imdb_from_zip',
    'load_imdb_from_directory',
    'load_and_preprocess_imdb',
    'preprocess_text',
    
    # Embeddings
    'TfidfEmbedding',
    'Word2VecEmbedding',
    'BertEmbedding',
    'get_embedding',
    
    # Models
    'LSTMClassifier',
    'LSTMClassifierWrapper',
    'get_model',
    
    # Evaluation
    'compute_metrics',
    'cross_validate_with_repeats',
    'evaluate_on_test',
    'run_full_evaluation',
    
    # Utils
    'load_config',
    'save_config',
    'get_environment_info',
    'save_results_csv',
    'load_results_csv',
    'format_results_table',
]
