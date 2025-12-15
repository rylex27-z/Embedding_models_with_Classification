#!/usr/bin/env python3
"""
Quick test script to validate the basic workflow without requiring the full IMDB dataset.
This creates synthetic data for testing purposes.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import TfidfEmbedding, Word2VecEmbedding
from src.models import get_model
from src.evaluation import cross_validate_with_repeats, compute_metrics


def generate_synthetic_data(n_samples=200):
    """Generate synthetic sentiment data for testing."""
    np.random.seed(42)
    
    # Positive sentiment vocabulary
    pos_words = ['excellent', 'great', 'amazing', 'wonderful', 'fantastic', 
                 'love', 'best', 'perfect', 'brilliant', 'outstanding']
    
    # Negative sentiment vocabulary
    neg_words = ['terrible', 'awful', 'horrible', 'worst', 'bad', 
                 'hate', 'poor', 'disappointing', 'waste', 'boring']
    
    # Neutral words
    neutral_words = ['movie', 'film', 'watch', 'actor', 'scene', 'story', 
                     'character', 'plot', 'director', 'cinema']
    
    texts = []
    labels = []
    
    # Generate positive reviews
    for _ in range(n_samples // 2):
        n_words = np.random.randint(10, 30)
        words = (
            np.random.choice(pos_words, size=n_words//2).tolist() +
            np.random.choice(neutral_words, size=n_words//2).tolist()
        )
        np.random.shuffle(words)
        texts.append(' '.join(words))
        labels.append(1)
    
    # Generate negative reviews
    for _ in range(n_samples // 2):
        n_words = np.random.randint(10, 30)
        words = (
            np.random.choice(neg_words, size=n_words//2).tolist() +
            np.random.choice(neutral_words, size=n_words//2).tolist()
        )
        np.random.shuffle(words)
        texts.append(' '.join(words))
        labels.append(0)
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, np.array(labels)


def test_tfidf_logreg():
    """Test TF-IDF + Logistic Regression."""
    print("\n" + "="*80)
    print("TEST 1: TF-IDF + Logistic Regression")
    print("="*80)
    
    # Generate data
    print("Generating synthetic data...")
    texts, labels = generate_synthetic_data(n_samples=200)
    print(f"Generated {len(texts)} samples")
    
    # Create embedding
    print("Creating TF-IDF embedding...")
    embedding = TfidfEmbedding(max_features=100, ngram_range=(1, 2))
    
    # Create model
    print("Creating Logistic Regression model...")
    model = get_model('logreg', config={'max_iter': 100, 'random_state': 42})
    
    # Run cross-validation (small config for testing)
    print("Running 3-fold CV × 2 repeats = 6 evaluations...")
    cv_results, cv_summary = cross_validate_with_repeats(
        embedding_obj=embedding,
        model_obj=model,
        X_train=texts,
        y_train=labels,
        n_splits=3,
        n_repeats=2,
        random_seeds=[42, 123],
        verbose=True
    )
    
    print("\n✅ TF-IDF + LogReg test PASSED")
    print(f"   Accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
    print(f"   F1: {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")
    
    return cv_results, cv_summary


def test_word2vec_randomforest():
    """Test Word2Vec + Random Forest."""
    print("\n" + "="*80)
    print("TEST 2: Word2Vec + Random Forest")
    print("="*80)
    
    # Generate data
    print("Generating synthetic data...")
    texts, labels = generate_synthetic_data(n_samples=200)
    print(f"Generated {len(texts)} samples")
    
    # Create embedding
    print("Creating Word2Vec (CBOW) embedding...")
    embedding = Word2VecEmbedding(vector_size=50, window=3, min_count=1, sg=0, epochs=5)
    
    # Create model
    print("Creating Random Forest model...")
    model = get_model('randomforest', config={'n_estimators': 10, 'random_state': 42})
    
    # Run cross-validation (small config for testing)
    print("Running 3-fold CV × 2 repeats = 6 evaluations...")
    cv_results, cv_summary = cross_validate_with_repeats(
        embedding_obj=embedding,
        model_obj=model,
        X_train=texts,
        y_train=labels,
        n_splits=3,
        n_repeats=2,
        random_seeds=[42, 123],
        verbose=True
    )
    
    print("\n✅ Word2Vec + RandomForest test PASSED")
    print(f"   Accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
    print(f"   F1: {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")
    
    return cv_results, cv_summary


def test_cv_protocol():
    """Test that CV protocol produces correct number of evaluations."""
    print("\n" + "="*80)
    print("TEST 3: CV Protocol (20 evaluations)")
    print("="*80)
    
    # Generate data
    print("Generating synthetic data...")
    texts, labels = generate_synthetic_data(n_samples=200)
    
    # Create simple setup
    embedding = TfidfEmbedding(max_features=50)
    model = get_model('logreg', config={'max_iter': 100, 'random_state': 42})
    
    # Run 5-fold × 4 repeats = 20 evaluations
    print("Running 5-fold CV × 4 repeats = 20 evaluations...")
    cv_results, cv_summary = cross_validate_with_repeats(
        embedding_obj=embedding,
        model_obj=model,
        X_train=texts,
        y_train=labels,
        n_splits=5,
        n_repeats=4,
        random_seeds=[42, 123, 456, 789],
        verbose=False
    )
    
    n_evaluations = len(cv_results)
    print(f"Number of evaluations: {n_evaluations}")
    
    assert n_evaluations == 20, f"Expected 20 evaluations, got {n_evaluations}"
    assert cv_summary['n_folds'] == 20, f"Expected n_folds=20, got {cv_summary['n_folds']}"
    
    print("\n✅ CV protocol test PASSED (20 evaluations confirmed)")
    print(f"   Accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
    
    return cv_results, cv_summary


def test_results_format():
    """Test that results are in correct format."""
    print("\n" + "="*80)
    print("TEST 4: Results Format")
    print("="*80)
    
    texts, labels = generate_synthetic_data(n_samples=100)
    embedding = TfidfEmbedding(max_features=50)
    model = get_model('logreg')
    
    cv_results, cv_summary = cross_validate_with_repeats(
        embedding_obj=embedding,
        model_obj=model,
        X_train=texts,
        y_train=labels,
        n_splits=2,
        n_repeats=2,
        random_seeds=[42, 123],
        verbose=False
    )
    
    # Check CV results DataFrame columns
    expected_cols = ['repeat', 'seed', 'fold', 'overall_fold', 'runtime', 
                     'accuracy', 'f1', 'precision', 'recall']
    for col in expected_cols:
        assert col in cv_results.columns, f"Missing column: {col}"
    
    # Check summary keys
    expected_keys = ['n_folds', 'accuracy_mean', 'accuracy_std', 
                     'f1_mean', 'f1_std', 'precision_mean', 'precision_std',
                     'recall_mean', 'recall_std', 'runtime_mean', 'runtime_std']
    for key in expected_keys:
        assert key in cv_summary, f"Missing summary key: {key}"
    
    print("✅ Results format test PASSED")
    print(f"   CV results shape: {cv_results.shape}")
    print(f"   Summary keys: {len(cv_summary)}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING END-TO-END TESTS")
    print("="*80)
    print("\nNote: Using synthetic data for testing (no IMDB dataset required)")
    
    try:
        # Run tests
        test_tfidf_logreg()
        test_word2vec_randomforest()
        test_cv_protocol()
        test_results_format()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print("\nThe project structure is working correctly!")
        print("Next steps:")
        print("  1. Download IMDB dataset (see data/README.md)")
        print("  2. Run real experiments with scripts/run_experiment.py")
        print("  3. Or use notebooks for interactive exploration")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ❌")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
