# IMDB Dataset Setup Instructions

## Dataset Overview

The **IMDB Movie Review Dataset** contains 50,000 movie reviews for binary sentiment classification:
- **25,000 training reviews** (12,500 positive, 12,500 negative)
- **25,000 test reviews** (12,500 positive, 12,500 negative)

**Citation**: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL 2011.

## Download Sources

### Option 1: Direct Download (Recommended)

Download from Stanford AI Lab:
- **URL**: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
- **Size**: ~84 MB (compressed)

```bash
# Linux/Mac
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

# Convert to zip (optional, for consistency)
zip -r aclImdb.zip aclImdb/
```

### Option 2: Kaggle

Search for "IMDB Dataset" on Kaggle and download.

## Setup for Google Colab

### Method 1: Upload to Colab Session (Quick)

1. Download `aclImdb_v1.tar.gz` or create `aclImdb.zip`
2. In Colab, upload the file:

```python
from google.colab import files
uploaded = files.upload()  # Select your aclImdb.zip file
```

3. The file will be in `/content/aclImdb.zip`
4. Update your experiment config or use `--data-path /content/aclImdb.zip`

**Note**: Files uploaded this way are temporary and will be lost when the runtime disconnects.

### Method 2: Google Drive (Persistent, Recommended)

1. Upload `aclImdb.zip` to your Google Drive (e.g., `MyDrive/data/aclImdb.zip`)

2. In Colab, mount your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Use the Drive path in your experiments:

```python
!python scripts/run_experiment.py \
    --embedding tfidf \
    --model logreg \
    --data-path /content/drive/MyDrive/data/aclImdb.zip
```

**Advantages**:
- Persistent storage (no need to re-upload)
- Can reuse across multiple Colab sessions
- Share with collaborators

### Method 3: Download in Colab

```python
# Download and extract in Colab
!wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xzf aclImdb_v1.tar.gz

# Create zip for consistency
!apt-get install -y zip
!zip -r aclImdb.zip aclImdb/
```

## Expected Directory Structure

After extraction, you should have:

```
aclImdb/
├── train/
│   ├── pos/          # 12,500 positive reviews (*.txt files)
│   ├── neg/          # 12,500 negative reviews (*.txt files)
│   └── unsup/        # Unsupervised data (not used in this project)
├── test/
│   ├── pos/          # 12,500 positive test reviews
│   └── neg/          # 12,500 negative test reviews
├── imdb.vocab        # Vocabulary file
├── imdbEr.txt        # Additional info
└── README
```

## Configuration

Update `configs/default_config.yaml` with your data path:

```yaml
data:
  zip_path: "/content/aclImdb.zip"  # or "/content/drive/MyDrive/data/aclImdb.zip"
  data_dir: null  # Alternative: path to extracted aclImdb/ directory
  sample_size: null  # Set to integer for quick testing (e.g., 1000)
  preprocess: true
  random_seed: 42
```

Or use command-line arguments:

```bash
# From zip
python scripts/run_experiment.py --data-path /content/aclImdb.zip --embedding tfidf --model logreg

# From directory
python scripts/run_experiment.py --data-path /content/aclImdb --embedding tfidf --model logreg

# With sampling for quick testing
python scripts/run_experiment.py --data-path /content/aclImdb.zip --sample 1000 --embedding bert --model logreg
```

## Data Loading in Code

The `src/data_loader.py` module handles data loading:

```python
from src.data_loader import load_and_preprocess_imdb

# Load from zip
train_texts, train_labels, test_texts, test_labels = load_and_preprocess_imdb(
    zip_path="/content/aclImdb.zip"
)

# Load from directory
train_texts, train_labels, test_texts, test_labels = load_and_preprocess_imdb(
    data_dir="/content/aclImdb"
)

# Load with sampling (for quick testing)
train_texts, train_labels, test_texts, test_labels = load_and_preprocess_imdb(
    zip_path="/content/aclImdb.zip",
    sample_size=1000  # 1000 per class per split
)
```

## Storage Best Practices

### ✅ DO:
- Keep dataset in Google Drive for persistent access
- Use `sample_size` parameter for quick testing/debugging
- Document the dataset version you're using

### ❌ DON'T:
- Commit the dataset to Git (it's gitignored)
- Store dataset in temporary Colab `/content` without backup
- Mix different dataset versions

## Troubleshooting

### "File not found" Error

1. Check the file path is correct
2. If using Drive, ensure it's mounted: `drive.mount('/content/drive')`
3. Verify the file exists: `!ls -lh /content/aclImdb.zip`

### "Out of Memory" During Loading

1. Use `sample_size` parameter to load a subset:
   ```python
   load_and_preprocess_imdb(zip_path="...", sample_size=1000)
   ```

2. For BERT experiments, use smaller samples initially:
   ```bash
   python scripts/run_experiment.py --sample 500 --embedding bert --model logreg
   ```

### Extraction Issues

If zip extraction fails:
```python
import zipfile
with zipfile.ZipFile('/content/aclImdb.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')
```

## Dataset Statistics

- **Total reviews**: 50,000
- **Training set**: 25,000 (12,500 pos + 12,500 neg)
- **Test set**: 25,000 (12,500 pos + 12,500 neg)
- **Average review length**: ~230 words
- **Vocabulary size**: ~90,000 unique words

## References

- **Paper**: Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In *Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies* (pp. 142-150).
- **Dataset URL**: https://ai.stanford.edu/~amaas/data/sentiment/
- **Additional Info**: http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf
