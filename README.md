# IMDB Sentiment Classification Benchmark

A reproducible benchmarking scaffold for comparing multiple text embeddings and classification models on the Stanford IMDB sentiment dataset.

## 🎯 Goals

- Compare various text embedding methods (TF-IDF, Word2Vec, BERT variants)
- Evaluate multiple classifiers (Logistic Regression, Random Forest, AdaBoost, LSTM)
- Rigorous evaluation with 5-fold cross-validation × 4 seeds = 20 evaluations
- Google Colab-friendly workflow (no local GPU required)
- Export results (metrics tables, plots) and commit back to GitHub

## 📁 Repository Structure

```
.
├── data/                          # Dataset directory (excluded from git)
│   ├── README.md                 # Dataset download instructions
│   ├── raw/                      # Raw IMDB dataset (not committed)
│   └── processed/                # Processed data (not committed)
├── src/
│   └── imdb_benchmark/           # Main package
│       ├── data_loader.py        # IMDB dataset loader
│       ├── embeddings/           # Embedding implementations
│       │   ├── tfidf.py         # TF-IDF vectorizer
│       │   ├── word2vec.py      # Word2Vec (CBOW/Skip-gram)
│       │   ├── bert.py          # BERT-based embeddings
│       │   └── registry.py      # Embedding factory
│       ├── models/               # Model implementations
│       │   ├── sklearn_models.py # Logistic Regression, RF, AdaBoost
│       │   ├── lstm.py          # PyTorch LSTM classifier
│       │   └── registry.py      # Model factory
│       └── tuning.py            # Cross-validation & hyperparameter tuning
├── scripts/
│   ├── run_experiment.py        # CLI for running experiments
│   └── build_results_table.py   # Generate summary tables
├── notebooks/
│   ├── 01_baseline_tfidf_logreg.ipynb   # Baseline experiment
│   └── 02_run_benchmark.ipynb           # Full benchmark suite
├── docs/
│   ├── theory_embeddings.md     # Embedding methods theory
│   ├── theory_models.md         # Model theory
│   └── experiment_protocol.md   # Experiment design
├── reports/
│   ├── results/                 # Experiment results (CSV, markdown)
│   └── final_report.md          # Final report template
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation
```

## 🚀 Quick Start

### 1. Dataset Setup

Download the IMDB dataset:

```bash
# Option 1: wget (Linux/Mac)
cd data/raw
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

# Option 2: Manual download
# Download from: http://ai.stanford.edu/~amaas/data/sentiment/
# Extract to data/raw/aclImdb/
```

See [data/README.md](data/README.md) for detailed instructions, including Google Colab setup.

### 2. Install Dependencies

```bash
# Local installation
pip install -r requirements.txt
pip install -e .

# Or on Google Colab (in notebook)
!pip install gensim transformers torch scikit-learn
```

### 3. Run an Experiment

#### Option A: Command Line

```bash
# Run TF-IDF + Logistic Regression
python scripts/run_experiment.py \
  --data /path/to/aclImdb \
  --embedding tfidf_bigram \
  --model logreg

# List available options
python scripts/run_experiment.py --list

# Run with hyperparameter tuning
python scripts/run_experiment.py \
  --data /path/to/aclImdb \
  --embedding w2v_cbow \
  --model rf \
  --tune
```

#### Option B: Notebooks

1. Open `notebooks/01_baseline_tfidf_logreg.ipynb` for a baseline experiment
2. Open `notebooks/02_run_benchmark.ipynb` to run multiple experiments

On Google Colab:
1. Upload notebooks to Colab
2. Mount Google Drive or upload dataset to `/content/`
3. Update `DATA_PATH` in the notebook
4. Run all cells

### 4. View Results

Results are saved to `reports/results/`:
- `results_long.csv` - Detailed results for each fold/seed
- `results_summary.csv` - Aggregated statistics
- `results_table.md` - Markdown table for reports

Generate summary table:

```bash
python scripts/build_results_table.py
```

## 📊 Available Embeddings

### TF-IDF
- `tfidf_unigram` - Unigrams only
- `tfidf_bigram` - Unigrams + bigrams
- `tfidf_trigram` - Unigrams + bigrams + trigrams

### Word2Vec (Gensim)
- `w2v_cbow` - CBOW (Continuous Bag of Words)
- `w2v_skipgram` - Skip-gram
- `w2v_cbow_300d` - CBOW with 300-dimensional vectors

### BERT-based (Transformers)
- `bert_base` - BERT base uncased
- `roberta_base` - RoBERTa base
- `distilbert_base` - DistilBERT base uncased
- `albert_base` - ALBERT base v2

## 🤖 Available Models

### Scikit-learn
- `logreg` - Logistic Regression
- `rf` - Random Forest
- `adaboost` - AdaBoost

### PyTorch
- `lstm` - LSTM text classifier (end-to-end with own tokenization)

## 🔬 Cross-Validation Protocol

- **5-fold StratifiedKFold** to maintain class balance
- **4 random seeds** (42, 123, 456, 789) for robustness
- **Total: 20 evaluations** per embedding+model combination

### Metrics Collected
- Accuracy
- F1 Score (binary)
- Precision
- Recall
- Training time
- Inference time

## 📝 Workflow for Colab Users

1. **Upload dataset to Google Drive** or `/content/` on Colab
2. **Clone this repository** in Colab:
   ```python
   !git clone https://github.com/rylex27-z/Embedding_models_with_Classification.git
   %cd Embedding_models_with_Classification
   ```
3. **Run experiments** using notebooks or CLI
4. **Download results** or save to Drive:
   ```python
   from google.colab import files
   files.download('reports/results/results_long.csv')
   ```
5. **Commit results back to GitHub**:
   - Download result files
   - Commit locally and push
   - OR use git commands in Colab (requires authentication)

## 🔧 Customization

### Add Custom Embedding

1. Create new file in `src/imdb_benchmark/embeddings/`
2. Implement `fit()`, `transform()`, and `fit_transform()` methods
3. Register in `embeddings/registry.py`

### Add Custom Model

1. Create classifier in `src/imdb_benchmark/models/`
2. Ensure it has `fit()` and `predict()` methods (scikit-learn compatible)
3. Register in `models/registry.py`

### Modify CV Protocol

Edit seeds and folds in experiments:

```python
evaluator = CVEvaluator(
    n_folds=10,  # Change number of folds
    seeds=[1, 2, 3, 4, 5],  # Change seeds
    tune_hyperparams=True,  # Enable tuning
)
```

## 📚 Documentation

- [Theory: Embeddings](docs/theory_embeddings.md) - TF-IDF, Word2Vec, BERT details
- [Theory: Models](docs/theory_models.md) - Classifier mathematics and intuitions
- [Experiment Protocol](docs/experiment_protocol.md) - Detailed experimental setup
- [Final Report](reports/final_report.md) - Results and analysis template

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run experiments and commit results
4. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Dataset**: Stanford IMDB Movie Review Dataset ([Maas et al., 2011](http://ai.stanford.edu/~amaas/data/sentiment/))
- **Libraries**: scikit-learn, Gensim, Hugging Face Transformers, PyTorch

## 📧 Contact

For questions or issues, please open a GitHub issue.