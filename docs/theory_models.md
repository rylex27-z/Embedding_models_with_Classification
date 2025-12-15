# Theory: Classification Models

This document provides theoretical background on the classification models used in this benchmark.

## Overview

Classification models learn decision boundaries to separate positive and negative sentiment reviews. Different models make different assumptions about the data and have varying capacities for modeling complex patterns.

---

## 1. Logistic Regression

### Intuition
Logistic regression models the probability of class membership as a linear combination of features passed through a sigmoid function.

### Mathematical Formulation

**Model**:
$$P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

where:
- $\mathbf{x}$ is the input feature vector (embedding)
- $\mathbf{w}$ is the weight vector
- $b$ is the bias term
- $\sigma$ is the sigmoid (logistic) function

**Decision Rule**:
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**Loss Function** (Binary Cross-Entropy):
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Regularization**: To prevent overfitting, add penalty term:
- **L2 (Ridge)**: $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\mathbf{w}\|_2^2$
- **L1 (Lasso)**: $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\mathbf{w}\|_1$

The regularization parameter $C = \frac{1}{\lambda}$ controls the inverse of regularization strength.

**Optimization**: Typically solved via gradient descent or L-BFGS:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}$$

### Strengths
- Simple, interpretable (feature importance via weights)
- Fast training and prediction
- Probabilistic outputs
- Works well with high-dimensional sparse features (TF-IDF)
- Well-studied, stable

### Weaknesses
- Linear decision boundary (limited expressiveness)
- Assumes feature independence
- Sensitive to feature scaling
- May underfit complex patterns

### References
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 4.
- Hastie, T., et al. (2009). *The Elements of Statistical Learning*. Chapter 4.

---

## 2. Random Forest

### Intuition
Random Forest is an ensemble of decision trees, each trained on a random subset of data and features. Predictions are aggregated by majority voting.

### Decision Trees

**Split Criterion** (for classification):

**Gini Impurity**:
$$G(t) = 1 - \sum_{k=1}^K p_k^2$$

where $p_k$ is the proportion of class $k$ in node $t$.

**Information Gain** (based on entropy):
$$H(t) = -\sum_{k=1}^K p_k \log p_k$$

$$\text{IG}(t, f) = H(t) - \sum_{c \in \{L, R\}} \frac{|t_c|}{|t|} H(t_c)$$

where $f$ is the feature to split on, and $L, R$ are left/right children.

### Random Forest Algorithm

1. **Bootstrap Sampling**: For each tree $i$, sample $N$ examples with replacement
2. **Random Feature Selection**: At each node, consider random subset of $m$ features
3. **Tree Growing**: Grow tree to maximum depth or minimum samples per leaf
4. **Aggregation**: Combine predictions by majority voting

**Prediction**:
$$\hat{y} = \text{mode}\{\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T\}$$

where $T$ is the number of trees.

### Hyperparameters

- **n_estimators**: Number of trees
- **max_depth**: Maximum tree depth (controls overfitting)
- **min_samples_split**: Minimum samples required to split a node
- **max_features**: Number of features to consider for splits

### Strengths
- Handles non-linear patterns
- Robust to outliers
- Provides feature importance scores
- Reduces overfitting via ensemble
- Works well with mixed feature types
- No feature scaling required

### Weaknesses
- Can overfit noisy data
- Large memory footprint
- Slower prediction than linear models
- Less interpretable than single trees
- Biased toward features with more levels

### References
- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
- Hastie, T., et al. (2009). *The Elements of Statistical Learning*. Chapter 15.

---

## 3. AdaBoost (Adaptive Boosting)

### Intuition
AdaBoost trains weak learners sequentially, with each learner focusing on examples misclassified by previous learners. Final prediction is a weighted vote.

### Algorithm

**Initialize**: Sample weights $w_i^{(1)} = \frac{1}{N}$ for $i = 1, \ldots, N$

**For** $t = 1$ **to** $T$:
1. Train weak learner $h_t$ on weighted dataset
2. Compute weighted error:
   $$\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i^{(t)}$$

3. Compute learner weight:
   $$\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$$

4. Update sample weights:
   $$w_i^{(t+1)} = w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))$$

5. Normalize weights: $w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_j w_j^{(t+1)}}$

**Final Prediction**:
$$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$$

### Weak Learners
Typically decision stumps (depth-1 trees), but can be any classifier with accuracy > 50%.

### Hyperparameters
- **n_estimators**: Number of boosting rounds
- **learning_rate**: Shrinkage parameter to prevent overfitting

### Strengths
- Improved accuracy via boosting
- Automatically focuses on hard examples
- Less prone to overfitting than single trees (with proper tuning)
- No feature scaling required

### Weaknesses
- Sensitive to noisy data and outliers
- Sequential training (cannot parallelize)
- Longer training time than Random Forest
- Can overfit if too many estimators

### References
- Freund, Y., & Schapire, R. E. (1997). "A Decision-Theoretic Generalization of On-Line Learning." *Journal of Computer and System Sciences*, 55(1), 119-139.
- Hastie, T., et al. (2009). *The Elements of Statistical Learning*. Chapter 10.

---

## 4. LSTM (Long Short-Term Memory)

### Intuition
LSTMs are recurrent neural networks designed to capture long-range dependencies in sequences. For text classification, they process word sequences to build document representations.

### LSTM Cell

**Gates** (at time step $t$):

**Forget Gate**:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate**:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell State**:
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update**:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate**:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State**:
$$h_t = o_t \odot \tanh(C_t)$$

where:
- $\sigma$ is the sigmoid function
- $\tanh$ is the hyperbolic tangent
- $\odot$ is element-wise multiplication
- $x_t$ is the input at time $t$
- $h_t$ is the hidden state
- $C_t$ is the cell state

### Text Classification Architecture

1. **Embedding Layer**: Maps words to dense vectors
2. **LSTM Layer**: Processes sequence, outputs hidden states
3. **Pooling**: Use final hidden state $h_T$ or mean pooling
4. **Dense Layer**: $\mathbf{z} = W h_T + b$
5. **Output**: Softmax for classification probabilities

**Loss Function** (Cross-Entropy):
$$\mathcal{L} = -\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik})$$

### Bidirectional LSTM

Process sequence in both directions and concatenate:
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

### Hyperparameters
- **embedding_dim**: Word embedding dimension
- **hidden_dim**: LSTM hidden state dimension
- **num_layers**: Number of stacked LSTM layers
- **dropout**: Dropout rate for regularization
- **learning_rate**: Step size for optimizer

### Strengths
- Captures sequential information and word order
- Handles variable-length sequences
- Learns task-specific word embeddings
- Models long-range dependencies

### Weaknesses
- Requires more data than traditional ML models
- Slower training and inference
- More hyperparameters to tune
- Needs GPU for efficient training
- Risk of overfitting on small datasets

### References
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
- Goodfellow, I., et al. (2016). *Deep Learning*. Chapter 10.

---

## Comparison Summary

| Model | Type | Complexity | Training Speed | Handles Sequences | Interpretability |
|-------|------|------------|----------------|------------------|------------------|
| Logistic Regression | Linear | Low | Fast | No | High |
| Random Forest | Ensemble (Trees) | Medium | Medium | No | Medium |
| AdaBoost | Ensemble (Boosting) | Medium | Medium-Slow | No | Low |
| LSTM | Neural Network | High | Slow (GPU helps) | Yes | Low |

---

## Model Selection Guidelines

**Use Logistic Regression when**:
- You need fast, interpretable baseline
- Working with high-dimensional sparse features (TF-IDF)
- Limited training data

**Use Random Forest when**:
- You need robust non-linear model
- Feature interactions are important
- You want feature importance

**Use AdaBoost when**:
- You want to boost weak learner performance
- You have clean data (sensitive to noise)

**Use LSTM when**:
- Word order is crucial
- You have sufficient training data
- You have GPU resources
- End-to-end learning is desired

---

## Recommended Reading

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
2. Hastie, T., et al. (2009). *The Elements of Statistical Learning*.
3. Goodfellow, I., et al. (2016). *Deep Learning*.
4. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.
