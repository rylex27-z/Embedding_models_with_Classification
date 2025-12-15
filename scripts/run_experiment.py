#!/usr/bin/env python3
"""
CLI script to run experiments with specified embedding and model.

Usage:
    python scripts/run_experiment.py --embedding tfidf --model logreg
    python scripts/run_experiment.py --embedding word2vec --variant skipgram --model randomforest
    python scripts/run_experiment.py --embedding bert --variant bert-base-uncased --model logreg --sample 1000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_and_preprocess_imdb
from src.embeddings import get_embedding
from src.models import get_model
from src.evaluation import run_full_evaluation
from src.utils import (
    load_config,
    save_results_csv,
    get_environment_info,
    get_timestamp,
    ensure_dir
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run IMDB sentiment classification experiment")
    
    # Required arguments
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=["tfidf", "word2vec", "bert"],
        help="Embedding type"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logreg", "randomforest", "adaboost", "lstm"],
        help="Model type"
    )
    
    # Optional arguments
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Embedding variant (e.g., 'skipgram', 'cbow', 'bert-base-uncased', 'roberta-base')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/default_config.yaml)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to IMDB zip or directory (overrides config)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size per class for testing (default: use full dataset)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation (only run test evaluation)"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=4,
        help="Number of CV repeats (default: 4, total=20 folds)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*80)
    print("IMDB Sentiment Classification Experiment")
    print("="*80)
    print(f"Embedding: {args.embedding}")
    print(f"Variant: {args.variant or 'default'}")
    print(f"Model: {args.model}")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data path if provided
    if args.data_path:
        if args.data_path.endswith('.zip'):
            config['data']['zip_path'] = args.data_path
            config['data']['data_dir'] = None
        else:
            config['data']['data_dir'] = args.data_path
            config['data']['zip_path'] = None
    
    # Override sample size if provided
    if args.sample:
        config['data']['sample_size'] = args.sample
    
    # Load data
    print("\nLoading IMDB dataset...")
    train_texts, train_labels, test_texts, test_labels = load_and_preprocess_imdb(
        zip_path=config['data'].get('zip_path'),
        data_dir=config['data'].get('data_dir'),
        sample_size=config['data'].get('sample_size'),
        random_seed=config['data'].get('random_seed', 42),
        preprocess=config['data'].get('preprocess', True)
    )
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    # Get embedding configuration
    if args.embedding == "tfidf":
        emb_config = config['embeddings']['tfidf']
        variant = args.variant or f"max_features_{emb_config['max_features']}"
    elif args.embedding == "word2vec":
        if args.variant and "skipgram" in args.variant.lower():
            emb_config = config['embeddings']['word2vec_skipgram']
            variant = args.variant or "skipgram"
        else:
            emb_config = config['embeddings']['word2vec_cbow']
            variant = args.variant or "cbow"
    elif args.embedding == "bert":
        emb_config = config['embeddings']['bert'].copy()
        variant = args.variant or "bert-base-uncased"
        emb_config['model_name'] = variant
    
    # Create embedding object
    print(f"\nInitializing {args.embedding} embedding ({variant})...")
    embedding_obj = get_embedding(args.embedding, variant, emb_config)
    
    # Get model configuration
    model_config = config['models'].get(args.model, {}).copy()
    
    # Create model object
    print(f"Initializing {args.model} model...")
    if args.model == "lstm":
        # For LSTM, we need to know the input dimension
        # We'll create a temporary embedding to get the dimension
        temp_emb = embedding_obj.fit_transform([train_texts[0]])
        input_dim = temp_emb.shape[1]
        model_obj = get_model(args.model, input_dim=input_dim, config=model_config)
    else:
        model_obj = get_model(args.model, config=model_config)
    
    # Cross-validation configuration
    cv_config = {
        'n_splits': args.cv_splits,
        'n_repeats': args.cv_repeats,
        'random_seeds': config['cross_validation'].get('random_seeds', [42, 123, 456, 789])
    }
    
    # Run evaluation
    if args.no_cv:
        print("\nSkipping cross-validation, running test evaluation only...")
        from src.evaluation import evaluate_on_test
        
        test_metrics = evaluate_on_test(
            embedding_obj=embedding_obj,
            model_obj=model_obj,
            X_train=train_texts,
            y_train=train_labels,
            X_test=test_texts,
            y_test=test_labels,
            verbose=True
        )
        
        results = {
            'embedding_type': args.embedding,
            'embedding_variant': variant,
            'model_type': args.model,
            'test_metrics': test_metrics
        }
    else:
        results = run_full_evaluation(
            embedding_obj=embedding_obj,
            model_obj=model_obj,
            X_train=train_texts,
            y_train=train_labels,
            X_test=test_texts,
            y_test=test_labels,
            embedding_type=args.embedding,
            embedding_variant=variant,
            model_type=args.model,
            cv_config=cv_config,
            verbose=True
        )
    
    # Save results
    timestamp = get_timestamp()
    output_prefix = args.output_prefix or f"{args.embedding}_{variant}_{args.model}_{timestamp}"
    
    results_dir = Path(config['output']['results_dir'])
    ensure_dir(results_dir)
    
    # Save CV results (long format)
    if not args.no_cv and 'cv_results' in results:
        cv_results_long = results['cv_results'].copy()
        cv_results_long['embedding_type'] = args.embedding
        cv_results_long['embedding_variant'] = variant
        cv_results_long['model_type'] = args.model
        
        long_file = results_dir / f"{output_prefix}_cv_long.csv"
        cv_results_long.to_csv(long_file, index=False)
        print(f"\nCV results (long format) saved to: {long_file}")
    
    # Save summary
    summary_data = {
        'embedding_type': args.embedding,
        'embedding_variant': variant,
        'model_type': args.model,
        'timestamp': timestamp,
    }
    
    if not args.no_cv and 'cv_summary' in results:
        summary_data.update(results['cv_summary'])
    
    if 'test_metrics' in results:
        for k, v in results['test_metrics'].items():
            summary_data[f'test_{k}'] = v
    
    # Add environment info
    env_info = get_environment_info()
    summary_data.update({
        'python_version': env_info['python_version'].split()[0],
        'cuda_available': env_info['cuda_available'],
        'gpu_name': env_info['gpu_name']
    })
    
    summary_df = pd.DataFrame([summary_data])
    summary_file = results_dir / f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
