# Theory: Text Embedding Methods

This document provides mathematical background and theoretical foundations for the text embedding methods used in this project.

## Overview

Text embeddings convert raw text into numerical vectors that can be processed by machine learning models. The quality of embeddings significantly impacts downstream task performance.

---

## 1. TF-IDF (Term Frequency-Inverse Document Frequency)

### Mathematical Formulation

For a term $t$ in document $d$ from corpus $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:

**Term Frequency (TF)**:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

- $f_{t,d}$ = raw count of term $t$ in document $d$

**Inverse Document Frequency (IDF)**:
$$\text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

- $|D|$ = total number of documents
- $|\{d \in D : t \in d\}|$ = number of documents containing term $t$

### Properties

- **Sparse representation**: Most entries are zero
- **High dimensionality**: Vocabulary size (e.g., 5000-10000 features)
- **No semantic understanding**: "good" and "excellent" are completely different
- **Efficient**: Fast computation and inference

### Hyperparameters

- `max_features`: Maximum vocabulary size
- `ngram_range`: Include unigrams, bigrams, etc. (e.g., (1,2))
- `min_df`: Minimum document frequency threshold
- `max_df`: Maximum document frequency threshold (remove common words)

### Implementation Notes

```python
# Document vector is concatenation of TF-IDF scores
doc_vector = [tfidf(t1, d), tfidf(t2, d), ..., tfidf(tn, d)]
```

---

## 2. Word2Vec

Word2Vec learns dense word embeddings using neural networks on large text corpora.

### 2.1 Continuous Bag of Words (CBOW)

**Objective**: Predict target word from context words

$$P(w_t | w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k})$$

**Architecture**:
1. Input: One-hot encoded context words
2. Hidden layer: Average of context word embeddings
3. Output: Softmax over vocabulary to predict target word

**Loss Function**:
$$L = -\log P(w_t | \text{context}(w_t))$$

### 2.2 Skip-gram

**Objective**: Predict context words from target word

$$P(w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k} | w_t)$$

**Independence Assumption**:
$$P(\text{context} | w_t) = \prod_{-k \leq j \leq k, j \neq 0} P(w_{t+j} | w_t)$$

**Loss Function**:
$$L = -\sum_{-k \leq j \leq k, j \neq 0} \log P(w_{t+j} | w_t)$$

### Softmax Approximation

Exact softmax is expensive:
$$P(w_O | w_I) = \frac{\exp(v_{w_O}^T v_{w_I})}{\sum_{w=1}^W \exp(v_w^T v_{w_I})}$$

**Negative Sampling**: Sample $k$ negative examples instead of full vocabulary:
$$\log \sigma(v_{w_O}^T v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v_{w_i}^T v_{w_I})]$$

### Document Embedding

Word2Vec produces word-level embeddings. For document embeddings:

**Simple Averaging**:
$$\mathbf{v}_{\text{doc}} = \frac{1}{|d|} \sum_{w \in d} \mathbf{v}_w$$

### Properties

- **Dense representation**: Typically 100-300 dimensions
- **Semantic relationships**: Similar words have similar vectors
- **Analogies**: "king - man + woman ≈ queen"
- **Context-independent**: One embedding per word (no disambiguation)

### Hyperparameters

- `vector_size`: Embedding dimensionality (e.g., 100, 300)
- `window`: Context window size (e.g., 5)
- `min_count`: Minimum word frequency threshold
- `sg`: 0 for CBOW, 1 for Skip-gram
- `epochs`: Training iterations

### CBOW vs Skip-gram

| Aspect | CBOW | Skip-gram |
|--------|------|-----------|
| **Speed** | Faster | Slower |
| **Data efficiency** | Better for large corpora | Better for small corpora |
| **Rare words** | Less effective | More effective |
| **Use case** | Frequent words | Rare words, small datasets |

---

## 3. BERT and Transformer-based Embeddings

### Architecture

BERT uses **Transformer encoder** with self-attention mechanism.

**Self-Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q$ = Queries, $K$ = Keys, $V$ = Values
- $d_k$ = dimension of keys

**Multi-Head Attention**:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Pre-training Objectives

**1. Masked Language Modeling (MLM)**:
- Randomly mask 15% of tokens
- Predict masked tokens from context
$$L_{\text{MLM}} = -\mathbb{E}_{\tilde{x} \sim \text{mask}(x)} \log P(x | \tilde{x})$$

**2. Next Sentence Prediction (NSP)** (BERT):
- Predict if sentence B follows sentence A
- Binary classification task

### Variants

| Model | Key Features |
|-------|-------------|
| **BERT** | Bidirectional, MLM+NSP |
| **RoBERTa** | No NSP, larger batches, more data |
| **DistilBERT** | Distilled (smaller, faster), 97% performance |
| **ALBERT** | Parameter sharing, factorized embeddings |

### Document Embedding from BERT

**[CLS] Token Pooling** (used in this project):
$$\mathbf{v}_{\text{doc}} = \mathbf{h}_{\text{[CLS]}}$$

**Mean Pooling**:
$$\mathbf{v}_{\text{doc}} = \frac{1}{|d|} \sum_{i=1}^{|d|} \mathbf{h}_i$$

Where $\mathbf{h}_i$ are contextualized token embeddings.

### Properties

- **Contextualized**: Same word has different embeddings in different contexts
- **Transfer learning**: Pre-trained on massive corpora
- **Bidirectional context**: Sees both left and right context
- **High quality**: State-of-the-art on many NLP tasks
- **Computationally expensive**: Requires GPU for practical use

### Hyperparameters

- `max_length`: Maximum sequence length (e.g., 512, 256)
- `batch_size`: Inference batch size (memory constraint)
- `pooling`: "cls" or "mean" pooling strategy
- `model_name`: Specific pre-trained model

---

## Comparison of Methods

| Method | Dimension | Context | Training | Inference Speed | Performance |
|--------|-----------|---------|----------|-----------------|-------------|
| **TF-IDF** | 1000-10000 (sparse) | No | None | Very Fast | Baseline |
| **Word2Vec CBOW** | 100-300 (dense) | Local window | Minutes | Fast | Good |
| **Word2Vec Skip-gram** | 100-300 (dense) | Local window | Minutes | Fast | Good |
| **BERT-base** | 768 (dense) | Full sequence | Pre-trained | Slow (GPU) | Best |
| **DistilBERT** | 768 (dense) | Full sequence | Pre-trained | Medium | Very Good |

---

## References

1. **TF-IDF**: Sparck Jones, K. (1972). "A statistical interpretation of term specificity and its application in retrieval." *Journal of Documentation*.

2. **Word2Vec**: Mikolov, T., et al. (2013). "Efficient estimation of word representations in vector space." *ICLR*.

3. **BERT**: Devlin, J., et al. (2019). "BERT: Pre-training of deep bidirectional transformers for language understanding." *NAACL*.

4. **RoBERTa**: Liu, Y., et al. (2019). "RoBERTa: A robustly optimized BERT pretraining approach." *arXiv preprint*.

5. **DistilBERT**: Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *NeurIPS Workshop*.

6. **ALBERT**: Lan, Z., et al. (2020). "ALBERT: A lite BERT for self-supervised learning of language representations." *ICLR*.

---

## Additional Reading

- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
