#!/usr/bin/env python3
"""
Batch experiment runner - runs multiple experiments systematically.

Usage:
    python scripts/run_batch_experiments.py --config configs/experiment_configs/quick_test.yaml
    python scripts/run_batch_experiments.py --embeddings tfidf word2vec --models logreg randomforest
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from itertools import product

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch experiments")
    
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=["tfidf", "word2vec"],
        help="Embeddings to test (default: tfidf word2vec)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "randomforest", "adaboost"],
        help="Models to test (default: logreg randomforest adaboost)"
    )
    parser.add_argument(
        "--bert-models",
        nargs="+",
        default=[],
        help="BERT variants to test (e.g., bert-base-uncased roberta-base)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to IMDB data"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for quick testing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    return parser.parse_args()


def build_command(embedding, variant, model, args):
    """Build experiment command."""
    cmd = [
        "python", "scripts/run_experiment.py",
        "--embedding", embedding,
        "--model", model
    ]
    
    if variant:
        cmd.extend(["--variant", variant])
    
    if args.config:
        cmd.extend(["--config", args.config])
    
    if args.data_path:
        cmd.extend(["--data-path", args.data_path])
    
    if args.sample:
        cmd.extend(["--sample", str(args.sample)])
    
    return cmd


def main():
    """Main function."""
    args = parse_args()
    
    print("="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    
    # Build experiment list
    experiments = []
    
    # Standard embeddings (TF-IDF, Word2Vec)
    for embedding in args.embeddings:
        if embedding == "word2vec":
            variants = ["cbow", "skipgram"]
        else:
            variants = [None]
        
        for variant in variants:
            for model in args.models:
                experiments.append((embedding, variant, model))
    
    # BERT embeddings (usually only with logreg)
    if args.bert_models:
        bert_classifiers = ["logreg"]  # BERT typically used with simple classifier
        for bert_variant in args.bert_models:
            for model in bert_classifiers:
                experiments.append(("bert", bert_variant, model))
    
    print(f"\nTotal experiments to run: {len(experiments)}")
    print("-"*80)
    
    # Run experiments
    failed = []
    for i, (embedding, variant, model) in enumerate(experiments, 1):
        variant_str = f" ({variant})" if variant else ""
        print(f"\n[{i}/{len(experiments)}] Running: {embedding}{variant_str} + {model}")
        print("-"*80)
        
        cmd = build_command(embedding, variant, model, args)
        
        if args.dry_run:
            print(f"Would run: {' '.join(cmd)}")
            continue
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✓ Completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed with error code {e.returncode}")
            failed.append((embedding, variant, model))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
    
    # Summary
    print("\n" + "="*80)
    print("BATCH EXPERIMENTS SUMMARY")
    print("="*80)
    
    if not args.dry_run:
        successful = len(experiments) - len(failed)
        print(f"Successful: {successful}/{len(experiments)}")
        
        if failed:
            print(f"\nFailed experiments:")
            for embedding, variant, model in failed:
                variant_str = f" ({variant})" if variant else ""
                print(f"  - {embedding}{variant_str} + {model}")
        else:
            print("\nAll experiments completed successfully! ✓")
        
        # Suggest next step
        print("\nNext step: Aggregate results")
        print("  python scripts/build_results_table.py")
    else:
        print(f"\nDry run complete. Would run {len(experiments)} experiments.")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
