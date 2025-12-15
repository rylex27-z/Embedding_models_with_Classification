# Experiment Protocol

This document records the experimental methodology, compute environment, and evaluation procedures used in this benchmark.

---

## Compute Environment

### Primary Platform: Google Colab

All experiments are designed to run on **Google Colab** with the following characteristics:

**Free Tier**:
- **CPU**: Intel Xeon (2 cores @ 2.0-2.3 GHz)
- **RAM**: 12-13 GB
- **GPU**: Optional (T4 with 15 GB VRAM when available)
- **Storage**: ~100 GB temporary
- **Session timeout**: 12 hours idle, 24 hours max

**Colab Pro** (optional):
- **RAM**: Up to 25 GB
- **GPU**: T4, P100, or V100
- **Priority access**: Faster GPU allocation

### Environment Snapshot

Record environment details at runtime using:

```python
from src.utils import get_environment_info
env_info = get_environment_info()
```

Captured information:
- Python version
- Operating system
- CUDA availability and version
- GPU name and memory
- Timestamp

Example output:
```
Python: 3.10.12
Platform: Linux-5.15.0-1045-gcp-x86_64
CUDA Available: True
CUDA Version: 11.8
GPU: Tesla T4
Timestamp: 20241215_120000
```

---

## Dataset

### IMDB Movie Reviews
- **Source**: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size**: 50,000 reviews (25k train, 25k test)
- **Classes**: Binary (positive/negative sentiment)
- **Balance**: Perfectly balanced (12,500 per class per split)
- **Preprocessing**: Basic HTML removal, lowercasing

### Data Loading

```python
from src.data_loader import load_and_preprocess_imdb

train_texts, train_labels, test_texts, test_labels = load_and_preprocess_imdb(
    zip_path="/content/aclImdb.zip",
    preprocess=True,
    random_seed=42
)
```

---

## Cross-Validation Protocol

### Default Configuration: 5×4 Stratified K-Fold

**Total evaluations**: 20 folds

**Method**: StratifiedKFold repeated with different random seeds

**Configuration**:
- `n_splits`: 5 (each repeat has 5 folds)
- `n_repeats`: 4 (repeat with 4 different seeds)
- `random_seeds`: [42, 123, 456, 789]

### Rationale

1. **Stratification**: Maintains class balance in each fold
2. **Multiple seeds**: Reduces variance from random split
3. **20 total folds**: Sufficient for robust mean±std estimates
4. **Practical for Colab**: Balances rigor with computation time

### Alternative: 20-Fold CV

For faster experiments, single 20-fold CV can be used:

```yaml
cross_validation:
  n_splits: 20
  n_repeats: 1
  random_seeds: [42]
```

### Procedure

For each fold:
1. Split training data into train/validation
2. Fit embedding on fold training set
3. Transform fold training and validation sets
4. Fit classifier on fold training embeddings
5. Predict on fold validation embeddings
6. Compute metrics (accuracy, F1, precision, recall)
7. Record runtime

After all folds:
1. Compute mean and standard deviation for each metric
2. Store per-fold results (long format)
3. Store aggregated statistics (summary format)

---

## Evaluation Metrics

All metrics computed using scikit-learn:

### Primary Metrics

1. **Accuracy**: Overall correctness
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **F1 Score**: Harmonic mean of precision and recall
   $$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

3. **Precision**: Positive predictive value
   $$\text{Precision} = \frac{TP}{TP + FP}$$

4. **Recall**: True positive rate (sensitivity)
   $$\text{Recall} = \frac{TP}{TP + FN}$$

### Additional Measurements

- **Runtime**: Wall-clock time for embedding + training + prediction per fold
- **Memory**: Peak memory usage (when available)

### Reporting Format

**Per-fold (long format)**:
```
repeat, seed, fold, accuracy, f1, precision, recall, runtime
0, 42, 0, 0.8524, 0.8501, 0.8612, 0.8394, 12.34
0, 42, 1, 0.8487, 0.8463, 0.8575, 0.8354, 11.98
...
```

**Summary (aggregated)**:
```
embedding_type, embedding_variant, model_type, 
accuracy_mean, accuracy_std, f1_mean, f1_std,
precision_mean, precision_std, recall_mean, recall_std,
runtime_mean, runtime_std
```

---

## Experimental Configurations

### Embedding Configurations

**TF-IDF**:
```yaml
max_features: 5000
ngram_range: [1, 2]
min_df: 2
max_df: 0.95
```

**Word2Vec (CBOW & Skip-gram)**:
```yaml
vector_size: 100
window: 5
min_count: 5
workers: 4
epochs: 10
```

**BERT variants**:
```yaml
max_length: 512  # or 256 for faster processing
batch_size: 8    # adjust based on GPU memory
pooling: "cls"   # or "mean"
```

Models tested:
- `bert-base-uncased`
- `roberta-base`
- `distilbert-base-uncased`
- `albert-base-v2`

### Model Configurations

**Logistic Regression**:
```yaml
C: 1.0
max_iter: 1000
solver: "lbfgs"
```

**Random Forest**:
```yaml
n_estimators: 100
max_depth: null
min_samples_split: 2
```

**AdaBoost**:
```yaml
n_estimators: 50
learning_rate: 1.0
```

**LSTM**:
```yaml
hidden_dim: 128
num_layers: 2
dropout: 0.3
bidirectional: true
learning_rate: 0.001
batch_size: 32
epochs: 10
```

---

## Resource Management

### Memory Constraints

**For BERT embeddings on Colab Free**:
- Use `sample_size` for initial testing (e.g., 1000 per class)
- Reduce `max_length` to 256 if OOM
- Reduce `batch_size` to 4 if needed
- Consider DistilBERT for faster processing

**For LSTM models**:
- Monitor GPU memory with `!nvidia-smi`
- Reduce `batch_size` or `hidden_dim` if OOM
- Use CPU for small experiments

### Sampling Strategy

For rapid prototyping:
```python
# Quick test with 500 samples per class
python scripts/run_experiment.py --sample 500 --embedding tfidf --model logreg

# Full dataset
python scripts/run_experiment.py --embedding tfidf --model logreg
```

---

## Reproducibility

### Random Seeds

All random operations seeded:
- Data sampling: `random_seed=42`
- Cross-validation: `random_seeds=[42, 123, 456, 789]`
- Model training: `random_state=42`
- PyTorch: `torch.manual_seed(42)`

### Version Pinning

Dependencies pinned in `requirements.txt`:
```
scikit-learn==1.3.0
transformers==4.30.2
torch==2.0.1
...
```

### Experiment Tracking

Each experiment saves:
1. Configuration (YAML)
2. Environment info
3. Per-fold results
4. Summary statistics
5. Timestamp

---

## Experimental Workflow

### Standard Procedure

1. **Setup environment**:
   ```python
   !pip install -r requirements.txt
   ```

2. **Upload/mount data**:
   ```python
   # Upload or use Google Drive
   ```

3. **Run experiments**:
   ```bash
   # Run individual experiments
   python scripts/run_experiment.py --embedding tfidf --model logreg
   python scripts/run_experiment.py --embedding word2vec --variant skipgram --model randomforest
   # ... (run all combinations)
   ```

4. **Aggregate results**:
   ```bash
   python scripts/build_results_table.py
   ```

5. **Commit results**:
   ```bash
   git add reports/results/*.csv reports/results/*.md
   git commit -m "Add experiment results"
   git push
   ```

### Batch Experiments

For running multiple experiments systematically, use notebooks:
- `notebooks/full_benchmark.ipynb`

---

## Statistical Significance

With 20-fold CV:
- **Mean**: Point estimate of expected performance
- **Std**: Uncertainty/variance in estimate
- **95% CI**: Approximately $\text{mean} \pm 2 \times \text{std}$

For comparing two methods:
- Non-overlapping confidence intervals suggest significant difference
- Paired t-test on fold-level results for formal testing

---

## Limitations and Considerations

1. **Colab session limits**: Save checkpoints regularly
2. **GPU availability**: Not guaranteed on free tier
3. **Computational cost**: BERT experiments take longer
4. **Memory constraints**: May need sampling for BERT
5. **Randomness**: Despite seeding, some variance expected

---

## References

- **Cross-validation**: Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection." *IJCAI*.
- **Stratified K-Fold**: Ensures balanced class distribution in each fold
- **Statistical Testing**: Dietterich, T. G. (1998). "Approximate statistical tests for comparing supervised classification learning algorithms." *Neural Computation*.

---

## Version History

- **v0.1.0** (2024-12-15): Initial experimental protocol
