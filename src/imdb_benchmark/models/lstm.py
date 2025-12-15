"""
PyTorch LSTM Text Classifier

Lightweight LSTM classifier for sequence-based sentiment classification.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(self, texts: List[List[int]], labels: List[int]):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.texts[idx]), torch.LongTensor([self.labels[idx]])


class LSTMClassifier(nn.Module):
    """LSTM-based text classifier."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, 2)  # Binary classification
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output


class LSTMTextClassifier:
    """Wrapper for LSTM classifier with training and prediction."""
    
    def __init__(
        self,
        max_vocab_size: int = 10000,
        max_seq_length: int = 256,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        batch_size: int = 32,
        epochs: int = 5,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize LSTM text classifier.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_seq_length: Maximum sequence length
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to use ('cuda', 'cpu', or None for auto)
            random_state: Random seed
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
        
        self.vocab = None
        self.word2idx = None
        self.model = None
    
    def _build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Keep top vocab_size words
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # Build word2idx (0 for padding, 1 for unknown)
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in most_common:
            word2idx[word] = len(word2idx)
        
        logger.info(f"Built vocabulary with {len(word2idx)} words")
        return word2idx
    
    def _texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of indices."""
        sequences = []
        for text in texts:
            words = text.lower().split()
            seq = [self.word2idx.get(word, 1) for word in words]  # 1 = <UNK>
            
            # Truncate or pad
            if len(seq) > self.max_seq_length:
                seq = seq[:self.max_seq_length]
            else:
                seq = seq + [0] * (self.max_seq_length - len(seq))  # 0 = <PAD>
            
            sequences.append(seq)
        
        return sequences
    
    def fit(self, texts: List[str], labels: List[int]) -> 'LSTMTextClassifier':
        """Train the LSTM classifier."""
        logger.info(f"Training LSTM on {len(texts)} samples")
        
        # Build vocabulary
        self.word2idx = self._build_vocab(texts)
        vocab_size = len(self.word2idx)
        
        # Convert texts to sequences
        sequences = self._texts_to_sequences(texts)
        
        # Create dataset and dataloader
        dataset = TextDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert texts to sequences
        sequences = self._texts_to_sequences(texts)
        
        # Create dataset and dataloader
        labels_dummy = [0] * len(texts)  # Dummy labels
        dataset = TextDataset(sequences, labels_dummy)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_texts, _ in dataloader:
                batch_texts = batch_texts.to(self.device)
                outputs = self.model(batch_texts)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for texts."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert texts to sequences
        sequences = self._texts_to_sequences(texts)
        
        # Create dataset and dataloader
        labels_dummy = [0] * len(texts)
        dataset = TextDataset(sequences, labels_dummy)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Predict
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch_texts, _ in dataloader:
                batch_texts = batch_texts.to(self.device)
                outputs = self.model(batch_texts)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probabilities.extend(probs)
        
        return np.array(probabilities)
    
    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            "max_vocab_size": self.max_vocab_size,
            "max_seq_length": self.max_seq_length,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }


def get_default_params() -> Dict[str, Any]:
    """Get default LSTM parameters."""
    return {
        "max_vocab_size": 10000,
        "max_seq_length": 256,
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.3,
        "bidirectional": False,
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 0.001,
    }


def get_hyperparameter_grid() -> Dict[str, list]:
    """Get hyperparameter search space for LSTM."""
    return {
        "hidden_dim": [64, 128, 256],
        "dropout": [0.2, 0.3, 0.5],
        "learning_rate": [0.0001, 0.001, 0.01],
    }
