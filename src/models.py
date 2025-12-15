"""
Classification models: Logistic Regression, Random Forest, AdaBoost, LSTM.
"""
import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class LSTMClassifier(nn.Module):
    """LSTM classifier for text classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM classifier.
        
        Args:
            input_dim: Input dimension (embedding size)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        
        Returns:
            Output probabilities (batch_size, 1)
        """
        # If input is 2D (batch_size, input_dim), add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(last_output)
        
        return out


class LSTMClassifierWrapper:
    """Wrapper for LSTM classifier to match sklearn interface."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 10,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize LSTM classifier wrapper.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
            device: Device to use
            verbose: Whether to print training progress
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model
        self.model = None
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y):
        """
        Fit LSTM classifier.
        
        Args:
            X: Training features (n_samples, input_dim)
            y: Training labels (n_samples,)
        """
        # Initialize model
        self.model = LSTMClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 2 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, input_dim)
        
        Returns:
            Probabilities (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = outputs.cpu().numpy()
        
        # Return probabilities for both classes
        return np.hstack([1 - probs, probs])
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, input_dim)
        
        Returns:
            Predicted labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


def get_model(
    model_type: str,
    input_dim: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Factory function to get model object.
    
    Args:
        model_type: Type of model ("logreg", "randomforest", "adaboost", "lstm")
        input_dim: Input dimension (required for LSTM)
        config: Configuration dictionary for model parameters
    
    Returns:
        Model object
    """
    if config is None:
        config = {}
    
    if model_type.lower() in ["logreg", "logistic", "logisticregression"]:
        return LogisticRegression(
            max_iter=config.get('max_iter', 1000),
            random_state=config.get('random_state', 42),
            **{k: v for k, v in config.items() if k not in ['max_iter', 'random_state']}
        )
    
    elif model_type.lower() in ["rf", "randomforest"]:
        return RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            random_state=config.get('random_state', 42),
            **{k: v for k, v in config.items() if k not in ['n_estimators', 'random_state']}
        )
    
    elif model_type.lower() == "adaboost":
        return AdaBoostClassifier(
            n_estimators=config.get('n_estimators', 50),
            random_state=config.get('random_state', 42),
            **{k: v for k, v in config.items() if k not in ['n_estimators', 'random_state']}
        )
    
    elif model_type.lower() == "lstm":
        if input_dim is None:
            raise ValueError("input_dim is required for LSTM")
        return LSTMClassifierWrapper(
            input_dim=input_dim,
            **config
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
