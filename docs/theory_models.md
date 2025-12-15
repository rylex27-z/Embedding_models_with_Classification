# Theory: Classification Models

This document provides mathematical background and theoretical foundations for the classification models used in this project.

---

## 1. Logistic Regression

### Mathematical Formulation

Binary classification using logistic (sigmoid) function.

**Model**:
$$P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where:
- $\mathbf{x}$ = input feature vector (text embedding)
- $\mathbf{w}$ = weight vector
- $b$ = bias term
- $\sigma$ = sigmoid function

**Decision Rule**:
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### Training: Maximum Likelihood Estimation

**Likelihood**:
$$L(\mathbf{w}, b) = \prod_{i=1}^n P(y_i | \mathbf{x}_i)^{y_i} (1 - P(y_i | \mathbf{x}_i))^{1-y_i}$$

**Log-Likelihood (Loss to minimize)**:
$$\ell(\mathbf{w}, b) = -\sum_{i=1}^n \left[ y_i \log P(y_i=1|\mathbf{x}_i) + (1-y_i) \log P(y_i=0|\mathbf{x}_i) \right]$$

This is the **binary cross-entropy loss**.

### Regularization

**L2 Regularization (Ridge)**:
$$\ell_{\text{reg}}(\mathbf{w}, b) = \ell(\mathbf{w}, b) + \lambda \|\mathbf{w}\|_2^2$$

**L1 Regularization (Lasso)**:
$$\ell_{\text{reg}}(\mathbf{w}, b) = \ell(\mathbf{w}, b) + \lambda \|\mathbf{w}\|_1$$

The hyperparameter $C = \frac{1}{\lambda}$ controls regularization strength.

### Properties

- **Linear decision boundary**
- **Probabilistic outputs**: Provides class probabilities
- **Fast training and inference**
- **Interpretable**: Feature importance from weights
- **Works well with high-dimensional sparse data** (e.g., TF-IDF)

### Hyperparameters

- `C`: Inverse regularization strength (smaller = stronger regularization)
- `solver`: Optimization algorithm ('lbfgs', 'liblinear', 'saga')
- `max_iter`: Maximum iterations for convergence

---

## 2. Random Forest

### Ensemble of Decision Trees

Random Forest builds multiple decision trees and aggregates their predictions.

**Prediction**:
$$\hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), ..., h_T(\mathbf{x})\}$$

Where $h_t$ is the prediction of tree $t$.

### Decision Tree Fundamentals

**Splitting Criterion - Gini Impurity**:
$$\text{Gini}(S) = 1 - \sum_{c=1}^C p_c^2$$

Where $p_c$ is the proportion of class $c$ in set $S$.

**Information Gain**:
$$\text{Gain}(S, A) = \text{Gini}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Gini}(S_v)$$

### Randomization

1. **Bootstrap Aggregating (Bagging)**:
   - Each tree trained on random sample (with replacement) of data
   - Reduces variance, prevents overfitting

2. **Feature Randomness**:
   - At each split, consider random subset of features
   - Typically $\sqrt{d}$ features for classification
   - Increases diversity among trees

### Properties

- **Non-linear decision boundaries**
- **Handles high-dimensional data well**
- **Feature importance**: Average decrease in impurity
- **Robust to outliers**
- **No probability calibration needed**
- **Can overfit if trees are too deep**

### Hyperparameters

- `n_estimators`: Number of trees (e.g., 100-500)
- `max_depth`: Maximum tree depth (None = unlimited)
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples in leaf node
- `max_features`: Number of features for split ('sqrt', 'log2')

---

## 3. AdaBoost (Adaptive Boosting)

### Boosting Framework

Sequential ensemble where each model corrects errors of previous models.

**Final Prediction**:
$$H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(\mathbf{x})\right)$$

Where:
- $h_t$ = weak learner at iteration $t$
- $\alpha_t$ = weight of learner $t$

### AdaBoost Algorithm

1. Initialize weights: $w_i^{(1)} = \frac{1}{n}$ for all samples

2. For $t = 1$ to $T$:
   
   a. Train weak learner $h_t$ on weighted data
   
   b. Compute weighted error:
   $$\epsilon_t = \sum_{i: h_t(\mathbf{x}_i) \neq y_i} w_i^{(t)}$$
   
   c. Compute learner weight:
   $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
   
   d. Update sample weights:
   $$w_i^{(t+1)} = w_i^{(t)} \exp(-\alpha_t y_i h_t(\mathbf{x}_i))$$
   
   e. Normalize weights

### Properties

- **Sequential training** (cannot parallelize)
- **Focuses on hard examples**
- **Can overfit with too many iterations**
- **Sensitive to noisy data and outliers**
- **Often uses decision stumps** (depth-1 trees) as weak learners

### Hyperparameters

- `n_estimators`: Number of boosting iterations
- `learning_rate`: Shrinks contribution of each classifier
- `base_estimator`: Weak learner type (default: decision stump)

---

## 4. LSTM (Long Short-Term Memory)

### Recurrent Neural Network for Sequences

LSTMs process sequential data with memory cells that maintain information over time.

### LSTM Cell

At each time step $t$, given input $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1}$:

**Forget Gate**:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input Gate**:
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate Cell State**:
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Cell State Update**:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output Gate**:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**Hidden State**:
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Where $\odot$ denotes element-wise multiplication.

### Bidirectional LSTM

Processes sequence in both forward and backward directions:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{forward}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{backward}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

### Classification with LSTM

For sentiment classification:

1. Input: Sequence of embeddings $[\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T]$
2. LSTM processes sequence: $[\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_T]$
3. Use final hidden state $\mathbf{h}_T$ (or pooling)
4. Fully connected layer + sigmoid: $\hat{y} = \sigma(\mathbf{W}\mathbf{h}_T + b)$

### Training

**Loss**: Binary cross-entropy
$$L = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)\right]$$

**Optimization**: Adam, SGD with backpropagation through time (BPTT)

### Properties

- **Captures long-term dependencies**
- **Handles variable-length sequences**
- **More parameters** than traditional ML models
- **Requires more training data**
- **GPU acceleration beneficial**
- **Can overfit on small datasets**

### Hyperparameters

- `hidden_dim`: Hidden state dimension (e.g., 128, 256)
- `num_layers`: Number of stacked LSTM layers (e.g., 1-3)
- `dropout`: Dropout rate for regularization (e.g., 0.3)
- `bidirectional`: Use bidirectional LSTM (True/False)
- `learning_rate`: Learning rate for optimizer (e.g., 0.001)
- `batch_size`: Training batch size (e.g., 32)
- `epochs`: Number of training epochs (e.g., 10)

---

## Model Comparison

| Model | Type | Complexity | Training Speed | Interpretability | Non-linear | Best For |
|-------|------|------------|----------------|------------------|------------|----------|
| **Logistic Regression** | Linear | Low | Very Fast | High | No | Baseline, sparse features |
| **Random Forest** | Ensemble (Trees) | Medium | Fast | Medium | Yes | Tabular data, robustness |
| **AdaBoost** | Ensemble (Boosting) | Medium | Medium | Low | Yes | When accuracy is priority |
| **LSTM** | Deep Learning | High | Slow | Low | Yes | Sequential patterns, large data |

---

## Evaluation Metrics

### Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (Sensitivity)
$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1 Score
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

Where:
- $TP$ = True Positives
- $TN$ = True Negatives
- $FP$ = False Positives
- $FN$ = False Negatives

---

## References

1. **Logistic Regression**: Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression*. Wiley.

2. **Random Forest**: Breiman, L. (2001). "Random forests." *Machine Learning*, 45(1), 5-32.

3. **AdaBoost**: Freund, Y., & Schapire, R. E. (1997). "A decision-theoretic generalization of on-line learning and an application to boosting." *Journal of Computer and System Sciences*, 55(1), 119-139.

4. **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780.

5. **Bidirectional LSTM**: Schuster, M., & Paliwal, K. K. (1997). "Bidirectional recurrent neural networks." *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.

---

## Additional Reading

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
