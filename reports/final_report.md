# IMDB Sentiment Classification: Final Report

**Authors**: [Your Name]  
**Date**: [Date]  
**Repository**: https://github.com/rylex27-z/Embedding_models_with_Classification

---

## Executive Summary

This report presents a comprehensive benchmark of text embeddings and classification models for binary sentiment analysis on the Stanford IMDB movie review dataset. We evaluate [X] embedding methods and [Y] classifiers using rigorous 5-fold cross-validation repeated across 4 random seeds (20 total evaluations per configuration).

**Key Findings**:
- [To be filled with top-performing combination]
- [Key insight about embedding comparison]
- [Key insight about model comparison]

---

## 1. Introduction

### 1.1 Motivation

Text classification is a fundamental NLP task with applications in sentiment analysis, spam detection, and content moderation. The choice of text representation (embedding) and classifier significantly impacts performance, computational cost, and interpretability.

This benchmark aims to:
1. Compare classical (TF-IDF, Word2Vec) and modern (BERT) embeddings
2. Evaluate traditional ML (Logistic Regression, Random Forest) vs. deep learning (LSTM) classifiers
3. Provide reproducible results with statistical confidence intervals
4. Enable informed model selection for sentiment analysis tasks

### 1.2 Dataset

**Stanford IMDB Movie Review Dataset**:
- 50,000 reviews (25,000 train, 25,000 test)
- Binary labels (positive/negative)
- Perfectly balanced
- Source: http://ai.stanford.edu/~amaas/data/sentiment/

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for details.

---

## 2. Methodology

### 2.1 Text Embeddings

We evaluate three categories of embeddings:

**1. TF-IDF** (sparse, count-based):
- Unigrams, bigrams, trigrams
- Max features: 10,000-20,000
- See [docs/theory_embeddings.md](docs/theory_embeddings.md#1-tf-idf)

**2. Word2Vec** (dense, predictive):
- CBOW and Skip-gram architectures
- 100-300 dimensional vectors
- Mean pooling for document vectors
- See [docs/theory_embeddings.md](docs/theory_embeddings.md#2-word2vec)

**3. BERT variants** (dense, contextualized):
- BERT-base, RoBERTa, DistilBERT, ALBERT
- 768-dimensional [CLS] token embeddings
- Pre-trained on large corpora
- See [docs/theory_embeddings.md](docs/theory_embeddings.md#3-bert-and-transformer-based-embeddings)

### 2.2 Classification Models

**Scikit-learn classifiers**:
- Logistic Regression (linear baseline)
- Random Forest (ensemble decision trees)
- AdaBoost (boosting ensemble)

**PyTorch classifier**:
- LSTM (recurrent neural network with built-in tokenization)

See [docs/theory_models.md](docs/theory_models.md) for mathematical details.

### 2.3 Evaluation Protocol

- **Cross-Validation**: 5-fold StratifiedKFold
- **Random Seeds**: 4 seeds (42, 123, 456, 789)
- **Total Evaluations**: 20 per configuration
- **Metrics**: Accuracy, F1, Precision, Recall, Training Time, Inference Time

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for complete protocol.

---

## 3. Results

### 3.1 Performance Summary

<!-- Include the auto-generated results table -->
<include src="results/results_table.md" />

**Alternative**: If the table is large, provide a link:
See [reports/results/results_table.md](reports/results/results_table.md) for complete results.

### 3.2 Top Performers

**Best Overall** (by F1 score):
1. [Embedding + Model]: F1 = [score] ± [std]
2. [Embedding + Model]: F1 = [score] ± [std]
3. [Embedding + Model]: F1 = [score] ± [std]

**Fastest** (by training + inference time):
1. [Embedding + Model]: [time] seconds
2. [Embedding + Model]: [time] seconds
3. [Embedding + Model]: [time] seconds

### 3.3 Embedding Comparison

**TF-IDF**:
- Performance: [Summary of accuracy/F1]
- Speed: [Fast/Medium/Slow]
- Best with: [Which model performed best]
- Observations: [Key insights]

**Word2Vec**:
- CBOW vs. Skip-gram: [Comparison]
- Performance: [Summary]
- Best with: [Which model]
- Observations: [Key insights]

**BERT Variants**:
- Performance: [Summary]
- Computational cost: [Discussion of speed/memory]
- Best variant: [bert/roberta/distilbert/albert]
- Observations: [Key insights]

### 3.4 Model Comparison

**Logistic Regression**:
- Works best with: [Which embeddings]
- Performance: [Summary]
- Observations: [Linear decision boundaries sufficient?]

**Random Forest**:
- Works best with: [Which embeddings]
- Performance: [Summary]
- Observations: [Benefit of non-linearity?]

**AdaBoost**:
- Works best with: [Which embeddings]
- Performance: [Summary]
- Observations: [Comparison with Random Forest]

**LSTM**:
- End-to-end performance: [Summary]
- Comparison with embedding + sklearn: [Discussion]
- Observations: [Benefit of sequence modeling?]

---

## 4. Analysis and Discussion

### 4.1 Why Do Embeddings Differ?

**Information Captured**:
- **TF-IDF**: Word importance in document vs. corpus
  - Pros: Interpretable, fast, works well with linear models
  - Cons: No semantic similarity, sparse, high-dimensional

- **Word2Vec**: Distributional semantics (words in similar contexts)
  - Pros: Dense, captures word similarity, generalizes better
  - Cons: Single vector per word (no context), requires training or pre-trained vectors

- **BERT**: Contextualized representations (word meaning depends on context)
  - Pros: State-of-the-art performance, rich pre-trained knowledge
  - Cons: Computationally expensive, may overfit small datasets

### 4.2 Trade-offs

**Performance vs. Speed**:
- [Discussion of accuracy/F1 vs. training/inference time]
- [When to use fast baselines vs. expensive BERT]

**Interpretability vs. Accuracy**:
- [Linear models with TF-IDF: feature importance]
- [Black-box models: higher accuracy, less interpretable]

**Training Data Requirements**:
- [TF-IDF/sklearn: work well with less data]
- [Word2Vec: needs corpus for training]
- [LSTM: needs more labeled data, prone to overfitting]

### 4.3 Practical Recommendations

**For Production Systems**:
- [Recommended embedding + model based on constraints]
- [Considerations: latency, throughput, memory, cost]

**For Research**:
- [Recommended approach for best performance]
- [Directions for improvement]

**For Resource-Constrained Settings**:
- [Best CPU-friendly combination]
- [Mobile/edge deployment considerations]

---

## 5. Limitations and Future Work

### 5.1 Limitations

- **Dataset**: Single domain (movie reviews), may not generalize
- **Metrics**: Binary classification, balanced data (real-world often imbalanced)
- **Hyperparameters**: Default settings used, tuning may improve results
- **Computational Resources**: Limited by Colab free tier
- **[Other limitations]**

### 5.2 Future Directions

- **Additional Embeddings**: FastText, ELMo, GPT-based embeddings
- **Fine-tuning**: Fine-tune BERT on IMDB instead of feature extraction
- **Ensemble Methods**: Combine multiple embeddings or models
- **Hyperparameter Optimization**: Bayesian optimization for tuning
- **Other Datasets**: Generalization to other sentiment datasets
- **Class Imbalance**: Evaluate on imbalanced splits
- **Interpretability Analysis**: LIME, SHAP for model explanations
- **[Other ideas]**

---

## 6. Conclusion

[Summary of key findings and takeaways]

[Reiterate best performing methods]

[Final recommendations for practitioners]

---

## 7. References

### Dataset
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. *ACL*.

### Embeddings
- Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv:1301.3781*.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv:1810.04805*.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT. *arXiv:1907.11692*.

### Models
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

### Additional Resources
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## Appendices

### Appendix A: Detailed Results

See `reports/results/results_long.csv` for all 20 evaluations per configuration.

### Appendix B: Code Repository

Full code available at: https://github.com/rylex27-z/Embedding_models_with_Classification

### Appendix C: Reproducibility

All experiments can be reproduced using:
```bash
python scripts/run_experiment.py --data /path/to/aclImdb --embedding [name] --model [name]
```

Random seeds: 42, 123, 456, 789

---

**End of Report**
