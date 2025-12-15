# Scripts Documentation

This directory contains CLI scripts for running experiments and processing results.

## Scripts Overview

### 1. `run_experiment.py`

Run a single experiment with specified embedding and model.

**Basic Usage:**
```bash
python scripts/run_experiment.py --embedding EMBEDDING_TYPE --model MODEL_TYPE
```

**Examples:**

```bash
# TF-IDF + Logistic Regression
python scripts/run_experiment.py --embedding tfidf --model logreg

# Word2Vec Skip-gram + Random Forest
python scripts/run_experiment.py --embedding word2vec --variant skipgram --model randomforest

# BERT + Logistic Regression (quick test with sampling)
python scripts/run_experiment.py --embedding bert --variant bert-base-uncased --model logreg --sample 1000

# RoBERTa + LSTM
python scripts/run_experiment.py --embedding bert --variant roberta-base --model lstm

# Specify custom data path
python scripts/run_experiment.py --embedding tfidf --model logreg --data-path /content/aclImdb.zip

# Custom CV configuration
python scripts/run_experiment.py --embedding tfidf --model logreg --cv-splits 10 --cv-repeats 2
```

**Options:**
- `--embedding`: `tfidf`, `word2vec`, or `bert`
- `--model`: `logreg`, `randomforest`, `adaboost`, or `lstm`
- `--variant`: Embedding variant (e.g., `skipgram`, `cbow`, `bert-base-uncased`, `roberta-base`)
- `--config`: Path to config YAML file (default: `configs/default_config.yaml`)
- `--data-path`: Path to IMDB zip or directory (overrides config)
- `--sample`: Sample size per class for testing (e.g., `1000`)
- `--output-prefix`: Prefix for output files
- `--no-cv`: Skip cross-validation (only run test evaluation)
- `--cv-splits`: Number of CV folds (default: 5)
- `--cv-repeats`: Number of CV repeats (default: 4)

**Outputs:**
- `{prefix}_cv_long.csv`: Per-fold detailed results
- `{prefix}_summary.csv`: Aggregated statistics

---

### 2. `build_results_table.py`

Aggregate experiment results into summary tables and markdown.

**Usage:**
```bash
python scripts/build_results_table.py
```

**Options:**
- `--results-dir`: Directory containing result CSV files (default: `reports/results`)
- `--output`: Output filename prefix (default: `results`)
- `--pattern`: Pattern for summary CSV files (default: `*_summary.csv`)

**Outputs:**
- `results_summary.csv`: Aggregated summary table
- `results_long.csv`: All per-fold results combined
- `results_table.md`: Markdown formatted table (ready for reports)

**Example:**
```bash
# Default: aggregate all results in reports/results/
python scripts/build_results_table.py

# Custom output name
python scripts/build_results_table.py --output my_results

# Custom results directory
python scripts/build_results_table.py --results-dir /content/drive/MyDrive/results
```

---

### 3. `test_workflow.py`

Test the end-to-end workflow using synthetic data (no IMDB dataset required).

**Usage:**
```bash
python scripts/test_workflow.py
```

**What it tests:**
1. TF-IDF + Logistic Regression
2. Word2Vec + Random Forest
3. CV protocol (20 evaluations)
4. Results format validation

**Purpose:**
- Validate installation and setup
- Test pipeline before running expensive experiments
- Quick sanity check after code changes

---

## Typical Workflow

### 1. Test Installation
```bash
python scripts/test_workflow.py
```

### 2. Run Individual Experiments
```bash
# Quick test with sampling
python scripts/run_experiment.py --embedding tfidf --model logreg --sample 1000

# Full experiment
python scripts/run_experiment.py --embedding tfidf --model logreg
```

### 3. Run Multiple Experiments
```bash
# Run several combinations
for embedding in tfidf word2vec; do
    for model in logreg randomforest adaboost; do
        python scripts/run_experiment.py --embedding $embedding --model $model
    done
done
```

### 4. Aggregate Results
```bash
python scripts/build_results_table.py
```

### 5. View Results
```bash
cat reports/results/results_table.md
```

---

## Tips

### For Google Colab

1. **Upload data first:**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload aclImdb.zip
   ```

2. **Run experiments:**
   ```bash
   !python scripts/run_experiment.py --embedding tfidf --model logreg --data-path /content/aclImdb.zip
   ```

3. **Save results to Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp reports/results/*.csv /content/drive/MyDrive/results/
   ```

### For Quick Testing

Use `--sample` to test with smaller dataset:
```bash
python scripts/run_experiment.py --embedding bert --variant distilbert-base-uncased --model logreg --sample 500
```

### For BERT Experiments

If you get OOM errors:
```bash
# Use DistilBERT (smaller, faster)
python scripts/run_experiment.py --embedding bert --variant distilbert-base-uncased --model logreg

# Or reduce sample size
python scripts/run_experiment.py --embedding bert --variant bert-base-uncased --model logreg --sample 1000
```

---

## Output Files

All outputs go to `reports/results/`:

**Per-experiment files:**
- `{embedding}_{variant}_{model}_{timestamp}_cv_long.csv`
- `{embedding}_{variant}_{model}_{timestamp}_summary.csv`

**Aggregated files (from build_results_table.py):**
- `results_summary.csv`
- `results_long.csv`
- `results_table.md`

---

## Troubleshooting

**"ModuleNotFoundError":**
```bash
pip install -r requirements.txt
```

**"File not found" (dataset):**
- Check `--data-path` points to correct location
- See `data/README.md` for dataset download instructions

**"Out of memory" (BERT):**
- Use `--sample` with smaller size
- Use DistilBERT instead of BERT
- Reduce batch size in config

**"CUDA out of memory":**
- Reduce `batch_size` in `configs/default_config.yaml`
- Use CPU (slower but works): BERT will auto-detect
- Use smaller model (DistilBERT)

---

## See Also

- Main documentation: `../README.md`
- Dataset setup: `../data/README.md`
- Experiment protocol: `../docs/experiment_protocol.md`
- Notebooks: `../notebooks/`
