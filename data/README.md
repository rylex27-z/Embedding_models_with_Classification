# IMDB Dataset Setup

## Overview
This directory contains the IMDB sentiment classification dataset used for benchmarking.

## Dataset Information
- **Source**: Stanford IMDB Movie Review Dataset
- **URL**: http://ai.stanford.edu/~amaas/data/sentiment/
- **Size**: ~80 MB (compressed)
- **Structure**: 
  - 25,000 training reviews (12,500 positive, 12,500 negative)
  - 25,000 test reviews (12,500 positive, 12,500 negative)

## Download Instructions

### Option 1: Manual Download
1. Download the dataset from: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
2. Extract the archive to get the `aclImdb/` folder
3. The expected folder structure:
   ```
   aclImdb/
   ├── train/
   │   ├── pos/  (12,500 positive reviews)
   │   ├── neg/  (12,500 negative reviews)
   │   └── unsup/ (50,000 unlabeled reviews - not used)
   └── test/
       ├── pos/  (12,500 positive reviews)
       └── neg/  (12,500 negative reviews)
   ```

### Option 2: Using wget (Linux/Mac)
```bash
cd data/raw
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

## Google Colab Setup

### Method 1: Upload to Colab Runtime (Temporary)
1. Upload `aclImdb_v1.tar.gz` to `/content/` on Colab
2. Extract with:
   ```python
   !tar -xzf /content/aclImdb_v1.tar.gz -C /content/
   ```
3. Use data path: `/content/aclImdb`

### Method 2: Google Drive (Persistent)
1. Upload `aclImdb_v1.tar.gz` to your Google Drive
2. Mount Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Extract from Drive:
   ```python
   !tar -xzf /content/drive/MyDrive/path/to/aclImdb_v1.tar.gz -C /content/
   ```
4. Use data path: `/content/aclImdb`

### Method 3: Direct Download in Colab
```python
!wget -P /content/ http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xzf /content/aclImdb_v1.tar.gz -C /content/
```

## Usage in Code

After extraction, configure your data path:

```python
from imdb_benchmark.data_loader import load_imdb_data

# Local development
data_path = "/path/to/aclImdb"

# Colab (after extraction)
data_path = "/content/aclImdb"

# Load data
X_train, y_train, X_test, y_test = load_imdb_data(data_path)
```

## Notes
- The raw dataset is **excluded from git** via `.gitignore` to keep the repository lightweight
- Each review is stored as a separate text file
- Ratings 1-4 are considered negative, 7-10 are positive (5-6 are excluded from the dataset)
- Original review texts are preserved without preprocessing
