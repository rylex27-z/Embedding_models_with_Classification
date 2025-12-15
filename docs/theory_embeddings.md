# Theory: Text Embeddings

This document provides theoretical background on the text embedding methods used in this benchmark.

## Overview

Text embeddings convert raw text into numerical vectors that capture semantic meaning. Different embedding methods make different trade-offs between computational cost, semantic richness, and downstream task performance.

---

## 1. TF-IDF (Term Frequency-Inverse Document Frequency)

### Intuition
TF-IDF weights words based on their importance in a document relative to a corpus. Words that appear frequently in a document but rarely across the corpus receive high weights.

### Mathematical Formulation

For a term $t$ in document $d$ from corpus $D$:

**Term Frequency (TF)**:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

where $f_{t,d}$ is the raw count of term $t$ in document $d$.

**Inverse Document Frequency (IDF)**:
$$\text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

where $|D|$ is the total number of documents, and the denominator counts documents containing term $t$.

**TF-IDF Score**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### N-grams
N-grams capture local word sequences:
- **Unigrams** (1-gram): individual words
- **Bigrams** (2-gram): word pairs (e.g., "not good")
- **Trigrams** (3-gram): word triples

N-grams help capture negations and multi-word expressions but increase feature dimensionality.

### Implementation Details
- **Normalization**: L2 normalization of TF-IDF vectors
- **Max features**: Limit vocabulary size (e.g., 10,000 most frequent terms)
- **Min/Max DF**: Filter very rare or very common terms

### Strengths
- Simple, interpretable, fast
- No training required
- Works well with linear models
- Captures word importance

### Weaknesses
- Bag-of-words: ignores word order (except n-grams)
- High-dimensional and sparse
- No semantic similarity between words
- Poor handling of out-of-vocabulary words

### References
- Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.

---

## 2. Word2Vec

Word2Vec learns dense word representations by predicting words from context (or vice versa).

### 2.1 CBOW (Continuous Bag of Words)

**Intuition**: Predict target word from context words.

**Objective**: Maximize probability of target word $w_t$ given context $C$:
$$\mathcal{L} = \sum_{t=1}^T \log P(w_t | C_t)$$

**Architecture**:
1. Input: One-hot encoded context words
2. Embedding layer: Maps words to dense vectors
3. Average context embeddings
4. Softmax layer: Predicts target word

**Mathematical Model**:
$$P(w_t | C_t) = \frac{\exp(v_{w_t}^T \cdot \bar{v}_C)}{\sum_{w \in V} \exp(v_w^T \cdot \bar{v}_C)}$$

where $\bar{v}_C = \frac{1}{|C|}\sum_{c \in C} v_c$ is the average context embedding.

### 2.2 Skip-gram

**Intuition**: Predict context words from target word.

**Objective**: Maximize probability of context given target:
$$\mathcal{L} = \sum_{t=1}^T \sum_{c \in C_t} \log P(w_c | w_t)$$

**Architecture**: Similar to CBOW but reversed direction.

**Mathematical Model**:
$$P(w_c | w_t) = \frac{\exp(v_{w_c}^T \cdot v_{w_t})}{\sum_{w \in V} \exp(v_w^T \cdot v_{w_t})}$$

### Training Techniques

**Negative Sampling**: Avoid computing full softmax by sampling negative examples:
$$\log \sigma(v_{w_O}^T v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v_{w_i}^T v_{w_I})]$$

where $\sigma$ is the sigmoid function and $P_n(w)$ is the negative sampling distribution.

**Hierarchical Softmax**: Use binary tree structure to reduce computational complexity from $O(|V|)$ to $O(\log |V|)$.

### Document Embeddings

Word2Vec produces word-level embeddings. For document classification, we use **mean pooling**:
$$v_d = \frac{1}{|d|} \sum_{w \in d} v_w$$

### Strengths
- Captures semantic similarity (e.g., "king" - "man" + "woman" ≈ "queen")
- Dense, low-dimensional representations
- Handles large vocabularies efficiently
- Pre-training transfers across tasks

### Weaknesses
- Requires training (or pre-trained embeddings)
- Mean pooling loses word order information
- Single embedding per word (no context sensitivity)

### References
- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space." *arXiv:1301.3781*.
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases." *NIPS*.

---

## 3. BERT and Transformer-based Embeddings

### 3.1 Transformer Architecture

**Self-Attention Mechanism**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, $V$ are query, key, value matrices, and $d_k$ is the key dimension.

**Multi-Head Attention**: Run multiple attention mechanisms in parallel and concatenate:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

### 3.2 BERT (Bidirectional Encoder Representations from Transformers)

**Pre-training Objectives**:

1. **Masked Language Modeling (MLM)**: Mask 15% of tokens and predict them:
   $$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i | w_{\setminus i})$$

2. **Next Sentence Prediction (NSP)**: Predict if sentence B follows sentence A.

**Architecture**:
- 12 layers (base) or 24 layers (large)
- Hidden size: 768 (base) or 1024 (large)
- 12 or 16 attention heads

**Document Representation**: Use [CLS] token embedding or mean pooling over all tokens.

### 3.3 Variants

**RoBERTa** (Robustly Optimized BERT):
- Remove NSP objective
- Dynamic masking
- Larger batch sizes and more data

**DistilBERT**:
- Knowledge distillation from BERT
- 40% smaller, 60% faster
- Retains 97% of BERT's performance

**ALBERT** (A Lite BERT):
- Factorized embedding parameterization
- Cross-layer parameter sharing
- Sentence-order prediction instead of NSP

### Strengths
- State-of-the-art on many NLP tasks
- Contextual embeddings (word meaning depends on context)
- Bidirectional context modeling
- Rich pre-trained representations

### Weaknesses
- Computationally expensive (especially for large models)
- Requires GPU for reasonable speed
- Large model sizes
- May overfit on small datasets

### References
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NIPS*.
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers." *arXiv:1810.04805*.
- Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT." *arXiv:1907.11692*.
- Sanh, V., et al. (2019). "DistilBERT." *arXiv:1910.01108*.
- Lan, Z., et al. (2019). "ALBERT." *arXiv:1909.11942*.

---

## Comparison Summary

| Method | Type | Dimensionality | Context-Aware | Training Required | Computational Cost |
|--------|------|----------------|---------------|-------------------|-------------------|
| TF-IDF | Sparse | High (10k+) | No | No | Low |
| Word2Vec CBOW | Dense | Low (100-300) | Window-based | Yes | Medium |
| Word2Vec Skip-gram | Dense | Low (100-300) | Window-based | Yes | Medium |
| BERT | Dense | Fixed (768) | Bidirectional | Pre-trained | High |
| RoBERTa | Dense | Fixed (768) | Bidirectional | Pre-trained | High |
| DistilBERT | Dense | Fixed (768) | Bidirectional | Pre-trained | Medium |
| ALBERT | Dense | Fixed (768) | Bidirectional | Pre-trained | Medium-High |

---

## Recommended Reading

1. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). Chapter 6: Vector Semantics and Embeddings.
2. Goldberg, Y. (2017). *Neural Network Methods for Natural Language Processing*.
3. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/
