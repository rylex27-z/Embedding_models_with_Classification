"""
Run Experiment CLI

Command-line interface to run IMDB sentiment classification experiments.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imdb_benchmark.data_loader import load_imdb_data
from imdb_benchmark.embeddings.registry import create_embedding, list_embeddings
from imdb_benchmark.models.registry import create_model, get_hyperparameter_grid, list_models
from imdb_benchmark.tuning import CVEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_experiment(
    data_path: str,
    embedding: str,
    model: str,
    output_dir: str = "reports/results",
    n_folds: int = 5,
    seeds: list = None,
    tune_hyperparams: bool = False,
    embedding_variant: str = None,
):
    """
    Run a single experiment with specified embedding and model.
    
    Args:
        data_path: Path to aclImdb directory
        embedding: Embedding name (from registry)
        model: Model name (from registry)
        output_dir: Directory to save results
        n_folds: Number of CV folds
        seeds: List of random seeds
        tune_hyperparams: Whether to tune hyperparameters
        embedding_variant: Variant name for results (optional)
    """
    logger.info(f"Starting experiment: {embedding} + {model}")
    logger.info(f"Data path: {data_path}")
    
    # Load data
    logger.info("Loading IMDB data...")
    X_train, y_train, X_test, y_test = load_imdb_data(data_path)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Set default seeds if not provided
    if seeds is None:
        seeds = [42, 123, 456, 789]
    
    # Create embedding and model factories
    def embedding_fn():
        return create_embedding(embedding)
    
    def model_fn():
        # Use first seed for model initialization
        return create_model(model, random_state=seeds[0])
    
    # Get hyperparameter grid for tuning
    param_grid = get_hyperparameter_grid(model) if tune_hyperparams else None
    
    # Create CV evaluator
    evaluator = CVEvaluator(
        n_folds=n_folds,
        seeds=seeds,
        tune_hyperparams=tune_hyperparams,
    )
    
    # Run cross-validation
    logger.info(f"Running {n_folds}-fold CV across {len(seeds)} seeds ({n_folds * len(seeds)} total evaluations)")
    results_df = evaluator.evaluate(
        embedding_fn=embedding_fn,
        model_fn=model_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        embedding_name=embedding_variant or embedding,
        model_name=model,
        param_grid=param_grid,
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "results_long.csv"
    
    # Append to existing file if it exists
    if results_file.exists():
        existing_df = pd.read_csv(results_file)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        logger.info(f"Appending to existing results file: {results_file}")
    else:
        logger.info(f"Creating new results file: {results_file}")
    
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    print(f"\nEmbedding: {embedding}")
    print(f"Model: {model}")
    print(f"Total evaluations: {len(results_df)}")
    print(f"\nMean metrics across {n_folds * len(seeds)} folds:")
    print(f"  Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"  F1 Score: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    print(f"  Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"  Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print("="*80 + "\n")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run IMDB sentiment classification experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with TF-IDF and Logistic Regression
  python scripts/run_experiment.py --data /path/to/aclImdb --embedding tfidf --model logreg
  
  # Run with BERT and Random Forest
  python scripts/run_experiment.py --data /content/aclImdb --embedding bert_base --model rf
  
  # Run with Word2Vec CBOW and LSTM
  python scripts/run_experiment.py --data /path/to/aclImdb --embedding w2v_cbow --model lstm
  
  # Run with hyperparameter tuning
  python scripts/run_experiment.py --data /path/to/aclImdb --embedding tfidf_bigram --model logreg --tune
  
  # List available embeddings and models
  python scripts/run_experiment.py --list
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to aclImdb directory",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="Embedding name (e.g., tfidf, tfidf_bigram, w2v_cbow, bert_base)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., logreg, rf, adaboost, lstm)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Custom variant name for results (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/results",
        help="Output directory for results (default: reports/results)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789],
        help="Random seeds for CV (default: 42 123 456 789)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available embeddings and models",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "="*80)
        print("AVAILABLE EMBEDDINGS")
        print("="*80)
        embeddings = list_embeddings()
        for emb_type, emb_list in embeddings.items():
            print(f"\n{emb_type.upper()}:")
            for emb_name in emb_list:
                print(f"  - {emb_name}")
        
        print("\n" + "="*80)
        print("AVAILABLE MODELS")
        print("="*80)
        models = list_models()
        for model_type, model_list in models.items():
            print(f"\n{model_type.upper()}:")
            for model_name in model_list:
                print(f"  - {model_name}")
        print("="*80 + "\n")
        return
    
    if not args.data:
        parser.error("--data is required (or use --list to see available options)")
    
    if not args.embedding:
        parser.error("--embedding is required")
    
    if not args.model:
        parser.error("--model is required")
    
    # Run experiment
    run_experiment(
        data_path=args.data,
        embedding=args.embedding,
        model=args.model,
        output_dir=args.output,
        n_folds=args.folds,
        seeds=args.seeds,
        tune_hyperparams=args.tune,
        embedding_variant=args.variant,
    )


if __name__ == "__main__":
    main()
