# IMDB Sentiment Classification Benchmark

A comprehensive benchmarking framework for comparing multiple text embedding methods and classification models on the IMDB movie review sentiment dataset.

## 🎯 Project Goals

This project provides a **standardized, reproducible** framework to:
- Compare different text embedding methods (TF-IDF, Word2Vec, BERT-family)
- Evaluate multiple classifiers (Logistic Regression, Random Forest, AdaBoost, LSTM)
- Run rigorous cross-validation (20-fold protocol by default)
- Generate publication-ready results and comparison tables
- Work seamlessly in **Google Colab** (no local GPU required)

## 📁 Repository Structure

```
.
├── configs/                    # Configuration files
│   ├── default_config.yaml     # Default hyperparameters and settings
│   └── experiment_configs/     # Individual experiment configurations
├── data/                       # Dataset directory (gitignored)
│   └── README.md              # Dataset download instructions
├── docs/                       # Documentation
│   ├── theory_embeddings.md   # Mathematical background on embeddings
│   ├── theory_models.md       # Model descriptions and theory
│   └── experiment_protocol.md # Compute environment and methodology
├── notebooks/                  # Jupyter notebooks for Colab
│   ├── getting_started.ipynb  # Quick start guide
│   └── full_benchmark.ipynb   # Complete benchmark workflow
├── reports/                    # Results and visualizations
│   ├── results/               # CSV and markdown tables (committed)
│   ├── figures/               # Plots and charts (gitignored)
│   └── final_report.md        # Final report template
├── scripts/                    # CLI scripts
│   ├── run_experiment.py      # Run single experiment
│   └── build_results_table.py # Aggregate results into tables
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_loader.py         # Load IMDB dataset
│   ├── embeddings.py          # Text embedding implementations
│   ├── models.py              # Classification models
│   ├── evaluation.py          # Cross-validation and metrics
│   └── utils.py               # Helper functions
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Quick Start (Google Colab)

### 1. Clone the Repository

```bash
!git clone https://github.com/rylex27-z/Embedding_models_with_Classification.git
%cd Embedding_models_with_Classification
```

### 2. Install Dependencies

```python
!pip install -r requirements.txt
```

### 3. Download IMDB Dataset

Download the IMDB dataset from [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) or Kaggle.

In Colab, upload the `aclImdb.zip` file to `/content/` or mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
# Place aclImdb.zip in /content/drive/MyDrive/data/
```

### 4. Run Your First Experiment

```python
# Example: TF-IDF + Logistic Regression
!python scripts/run_experiment.py \
    --embedding tfidf \
    --model logreg \
    --data-path /content/aclImdb.zip
```

### 5. View Results

Results are saved in `reports/results/`:
- `*_cv_long.csv` - Per-fold detailed results
- `*_summary.csv` - Aggregated mean±std statistics
- `*_table.md` - Markdown formatted table

## 🧪 Supported Embeddings and Models

### Embeddings
- **TF-IDF**: Configurable max_features and n-gram ranges
- **Word2Vec**: 
  - CBOW (Continuous Bag of Words)
  - Skip-gram
- **BERT-family**:
  - `bert-base-uncased`
  - `roberta-base`
  - `distilbert-base-uncased`
  - `albert-base-v2`
  - Extensible to other HuggingFace models

### Classifiers
- Logistic Regression (sklearn)
- Random Forest (sklearn)
- AdaBoost (sklearn)
- LSTM (PyTorch)

## 📊 Cross-Validation Protocol

By default, we use **StratifiedKFold with 5 splits repeated across 4 different seeds** = **20 total evaluations**.

This ensures robust statistical estimates of model performance with mean±std metrics.

Configuration in `configs/default_config.yaml`:
```yaml
cross_validation:
  n_splits: 5
  n_repeats: 4
  random_seeds: [42, 123, 456, 789]
```

## 🔧 Running Experiments

### CLI Interface

```bash
# TF-IDF + Logistic Regression
python scripts/run_experiment.py --embedding tfidf --model logreg

# Word2Vec Skip-gram + Random Forest
python scripts/run_experiment.py --embedding word2vec --variant skipgram --model randomforest

# BERT + Logistic Regression (with sampling for quick testing)
python scripts/run_experiment.py --embedding bert --variant bert-base-uncased --model logreg --sample 1000

# RoBERTa + LSTM
python scripts/run_experiment.py --embedding bert --variant roberta-base --model lstm
```

### Aggregating Results

After running multiple experiments:

```bash
python scripts/build_results_table.py
```

This generates:
- `reports/results/results_summary.csv`
- `reports/results/results_long.csv`
- `reports/results/results_table.md`

## 📓 Using Notebooks

Two Colab-ready notebooks are provided:

1. **`getting_started.ipynb`**: Quick introduction and single experiment
2. **`full_benchmark.ipynb`**: Complete benchmark across all combinations

Open in Colab and follow the step-by-step instructions.

## 💾 Dataset Instructions

See [`data/README.md`](data/README.md) for detailed instructions on:
- Downloading the IMDB dataset
- Uploading to Colab
- Using Google Drive for persistent storage

**Important**: The dataset is **NOT committed** to this repository. You must download it separately.

## 🔐 What Gets Committed

✅ **DO commit**:
- Source code (`src/`, `scripts/`)
- Configuration files (`configs/`)
- Documentation (`docs/`, `README.md`)
- Result tables (`reports/results/*.csv`, `*.md`)
- Notebooks (`notebooks/`)

❌ **DO NOT commit** (automatically gitignored):
- Raw dataset (`data/aclImdb/`, `*.zip`)
- Large model checkpoints (`*.pth`, `*.h5`)
- Processed data (`*.pkl`, `*.npy`)
- Figures (`reports/figures/*.png`)

## 🧮 Adding New Embeddings or Models

### Adding a New Embedding

1. Extend `src/embeddings.py` with a new class
2. Update `get_embedding()` factory function
3. Add configuration to `configs/default_config.yaml`
4. Update documentation

### Adding a New Classifier

1. Extend `src/models.py` with a new class
2. Update `get_model()` factory function
3. Add configuration to `configs/default_config.yaml`

## 📚 Documentation

- [`data/README.md`](data/README.md) - Dataset download and setup
- [`docs/theory_embeddings.md`](docs/theory_embeddings.md) - Mathematical background on embeddings
- [`docs/theory_models.md`](docs/theory_models.md) - Model descriptions
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md) - Experimental methodology
- [`reports/final_report.md`](reports/final_report.md) - Final report template

## ⚡ Performance Tips for Colab

### For BERT Models
- Use `--sample 1000` for quick testing
- Reduce `bert_max_length` to 256 if running out of memory
- Reduce `bert_batch_size` to 4 if needed

### For LSTM Models
- Use smaller `hidden_dim` and fewer `epochs` for faster training
- Batch size can be reduced if OOM errors occur

### Resource Monitoring
```python
# In Colab, check GPU usage
!nvidia-smi
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run experiments to validate
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- IMDB dataset: [Maas et al., 2011](https://ai.stanford.edu/~amaas/data/sentiment/)
- HuggingFace Transformers library
- scikit-learn, Gensim, PyTorch communities

## 📧 Contact

For questions or issues, please open a GitHub issue.