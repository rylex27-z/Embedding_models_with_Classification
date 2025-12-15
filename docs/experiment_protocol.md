# Experiment Protocol

This document describes the experimental setup, evaluation protocol, and environment specifications for the IMDB sentiment classification benchmark.

## Dataset

**Name**: Stanford IMDB Movie Review Dataset  
**Source**: http://ai.stanford.edu/~amaas/data/sentiment/  
**Citation**: Maas, A. L., et al. (2011). "Learning Word Vectors for Sentiment Analysis." *ACL*.

### Dataset Statistics
- **Training Set**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Test Set**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Total**: 50,000 labeled reviews
- **Label Balance**: Perfectly balanced (50% positive, 50% negative)
- **Additional Unlabeled**: 50,000 reviews (not used in this benchmark)

### Data Format
- Each review is stored as a separate text file
- Positive reviews: rating ≥ 7/10
- Negative reviews: rating ≤ 4/10
- Neutral reviews (5-6/10) excluded from dataset

### Preprocessing
- **Minimal preprocessing**: Use raw text as provided
- **No stemming or lemmatization** (to allow embeddings to learn morphology)
- **No stopword removal** (BERT and Word2Vec benefit from function words)
- **HTML tags**: Present in raw data, handled by embedding methods

---

## Cross-Validation Protocol

### Stratified K-Fold Cross-Validation

**Configuration**:
- **K = 5 folds**: 20% held out for validation in each fold
- **Stratification**: Maintain class balance (50% pos/neg) in each fold
- **Multiple Seeds**: 4 random seeds for robustness
  - Seeds: `[42, 123, 456, 789]`
- **Total Evaluations**: 5 folds × 4 seeds = **20 evaluations** per configuration

### Why This Protocol?

1. **Stratification**: Ensures balanced validation sets, important for binary classification metrics
2. **5 folds**: Standard choice balancing bias-variance trade-off
   - More folds → lower bias, higher variance
   - 5 folds uses 80% for training, 20% for validation
3. **Multiple seeds**: Reduces variance in results due to random data splits
   - 4 seeds provides stable estimates without excessive computation
   - Enables statistical significance testing
4. **Total evaluations**: 20 evaluations provide robust performance estimates

### Implementation

```python
from sklearn.model_selection import StratifiedKFold

seeds = [42, 123, 456, 789]
n_folds = 5

for seed in seeds:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Train and evaluate
        ...
```

---

## Evaluation Metrics

### Primary Metrics

**1. Accuracy**
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

**2. F1 Score** (Binary, macro average)
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**3. Precision**
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**4. Recall**
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

where:
- TP = True Positives (correctly predicted positive)
- TN = True Negatives (correctly predicted negative)
- FP = False Positives (negative predicted as positive)
- FN = False Negatives (positive predicted as negative)

### Secondary Metrics

**5. Training Time** (seconds)
- Time to fit model on training fold

**6. Inference Time** (seconds)
- Time to predict on validation fold

**7. Embedding Time** (seconds)
- Time to generate embeddings (separate from model training)

### Aggregation

For each embedding + model combination:
- **Mean**: Average across 20 evaluations
- **Standard Deviation**: Measure of variability
- **Min/Max**: Best and worst case performance

---

## Hyperparameter Tuning

### Strategy

**Default Configuration**: Use sensible defaults without tuning
- Faster experiments
- Fair comparison across methods
- Baseline for tuning improvements

**Optional Tuning**: Enable via `--tune` flag
- **Method**: Grid search or random search
- **Inner CV**: 3-fold cross-validation within each training fold
- **Scoring**: F1 score (primary metric)

### Hyperparameter Grids

**Logistic Regression**:
- `C`: [0.01, 0.1, 1.0, 10.0]

**Random Forest**:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]

**AdaBoost**:
- `n_estimators`: [25, 50, 100]
- `learning_rate`: [0.5, 1.0, 1.5]

**LSTM**:
- `hidden_dim`: [64, 128, 256]
- `dropout`: [0.2, 0.3, 0.5]
- `learning_rate`: [0.0001, 0.001, 0.01]

---

## Computing Environment

### Hardware

**Target**: Google Colab (free tier)
- **CPU**: 2-core Intel Xeon @ 2.3 GHz
- **RAM**: 12-13 GB
- **GPU**: NVIDIA Tesla T4 (15 GB VRAM) - when available
- **Disk**: ~100 GB

**Local**: CPU-only machines supported
- Minimum: 8 GB RAM
- Recommended: 16 GB RAM for BERT models

### Software

**Python**: 3.8+

**Core Libraries**:
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `gensim >= 4.3.0`

See `requirements.txt` for complete list.

### Reproducibility

**Random Seeds**:
- NumPy: `np.random.seed(seed)`
- PyTorch: `torch.manual_seed(seed)`
- Scikit-learn: `random_state=seed` in all applicable functions

**Deterministic Operations**:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Experimental Workflow

### Phase 1: Data Loading
1. Load IMDB dataset from local directory
2. Verify train/test split sizes
3. Check label balance

### Phase 2: Embedding Generation
1. Initialize embedding model
2. Fit on training data (if needed)
3. Transform train and validation texts to vectors

### Phase 3: Model Training & Evaluation
1. Initialize classifier
2. (Optional) Hyperparameter tuning via inner CV
3. Train on training fold embeddings
4. Predict on validation fold
5. Compute metrics

### Phase 4: Result Aggregation
1. Collect results from all folds and seeds
2. Compute mean and std for each metric
3. Save to CSV files

### Phase 5: Reporting
1. Generate summary tables
2. Create markdown reports
3. (Optional) Visualizations

---

## Expected Runtimes (Colab CPU)

Approximate times for **one full CV run** (5 folds × 4 seeds):

| Embedding | Model | Time (min) |
|-----------|-------|------------|
| TF-IDF | Logistic Regression | 5-10 |
| TF-IDF | Random Forest | 15-30 |
| TF-IDF | AdaBoost | 10-20 |
| Word2Vec CBOW | Logistic Regression | 20-40 |
| Word2Vec Skip-gram | Logistic Regression | 20-40 |
| BERT-base (CPU) | Logistic Regression | 60-120 |
| DistilBERT (CPU) | Logistic Regression | 40-80 |
| LSTM | - | 30-60 |

**Note**: BERT models are **significantly faster with GPU** (5-10× speedup).

---

## Output Files

### Results Directory: `reports/results/`

**1. `results_long.csv`**
- Detailed results for every fold/seed combination
- Columns: `embedding`, `model`, `seed`, `fold`, `accuracy`, `f1`, `precision`, `recall`, `train_seconds`, `infer_seconds`

**2. `results_summary.csv`**
- Aggregated statistics per embedding+model
- Columns: `embedding`, `model`, `accuracy_mean`, `accuracy_std`, `f1_mean`, `f1_std`, etc.

**3. `results_table.md`**
- Markdown-formatted table for inclusion in reports
- Sorted by F1 score (descending)

---

## Statistical Significance Testing

To test if one method is significantly better than another:

**Paired t-test**: Compare metrics across 20 matched evaluations

```python
from scipy.stats import ttest_rel

# Example: Compare F1 scores of two methods
method1_f1 = results_df[results_df['model'] == 'logreg']['f1']
method2_f1 = results_df[results_df['model'] == 'rf']['f1']

t_stat, p_value = ttest_rel(method1_f1, method2_f1)
print(f"p-value: {p_value}")
# If p < 0.05, difference is statistically significant
```

---

## Best Practices

1. **Always run full CV protocol** (all 20 evaluations) for fair comparison
2. **Use same random seeds** across experiments for consistency
3. **Report both mean and std** to show variability
4. **Save intermediate results** to avoid recomputation
5. **Document any deviations** from standard protocol
6. **Version control results** (commit CSV files to git)

---

## References

- Maas, A. L., et al. (2011). "Learning Word Vectors for Sentiment Analysis." *ACL*.
- Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap for Accuracy Estimation." *IJCAI*.
- Bengio, Y., & Grandvalet, Y. (2004). "No Unbiased Estimator of the Variance of K-Fold Cross-Validation." *JMLR*.
