"""
Market Scoring Unit for DeepTrader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for market scoring unit.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize temporal attention module.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Parameters for temporal attention
        # Change Ve to be 2D instead of 1D
        self.Ve = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.U1 = nn.Parameter(torch.Tensor(hidden_dim, 2 * hidden_dim))
        self.U2 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Use xavier_uniform_ for 2D parameters
        nn.init.xavier_uniform_(self.Ve)
        nn.init.xavier_uniform_(self.U1)
        nn.init.xavier_uniform_(self.U2)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Hidden states of shape [batch_size, seq_length, hidden_dim]
            x: Input features of shape [batch_size, seq_length, input_dim]
            
        Returns:
            Context vector of shape [batch_size, hidden_dim]
        """
        batch_size = h.size(0)
        seq_length = h.size(1)
        
        # Get last hidden state
        h_last = h[:, -1, :]  # [batch_size, hidden_dim]
        
        # Calculate attention weights
        e = []
        for t in range(seq_length):
            # Concatenate current hidden state with last hidden state
            h_concat = torch.cat([h[:, t, :], h_last], dim=1)  # [batch_size, 2*hidden_dim]
            
            # Apply attention mechanism
            # [batch_size, hidden_dim]
            u1 = torch.matmul(h_concat, self.U1.t())
            u2 = torch.matmul(x[:, t, :], self.U2.t())
            u = torch.tanh(u1 + u2)
            
            # Calculate attention score using Ve (need to adjust for 2D)
            # [batch_size]
            score = torch.matmul(u, self.Ve.squeeze(0))
            e.append(score)
        
        # Stack attention scores
        # [batch_size, seq_length]
        e = torch.stack(e, dim=1)
        
        # Apply softmax to get attention weights
        # [batch_size, seq_length]
        alpha = F.softmax(e, dim=1)
        
        # Calculate weighted sum of hidden states
        # [batch_size, hidden_dim]
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        
        return context


class MarketScoringUnit(nn.Module):
    """
    Market Scoring Unit as described in the paper.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        """
        Initialize Market Scoring Unit.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(MarketScoringUnit, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal attention layer
        self.temporal_attn = TemporalAttention(hidden_dim)
        
        # Output layer for Gaussian distribution parameters
        self.output_layer = nn.Linear(hidden_dim, 2)  # mu and sigma
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            Tuple of (mu, sigma) for Gaussian distribution
        """
        # Apply LSTM
        # output: [batch_size, seq_length, hidden_dim]
        # h_n: [num_layers, batch_size, hidden_dim]
        # c_n: [num_layers, batch_size, hidden_dim]
        output, (h_n, c_n) = self.lstm(x)
        
        # Apply temporal attention
        # [batch_size, hidden_dim]
        context = self.temporal_attn(output, x)
        
        # Apply dropout
        context = self.dropout(context)
        
        # Apply output layer
        # [batch_size, 2]
        params = self.output_layer(context)
        
        # Split into mu and sigma
        mu = params[:, 0]
        
        # Apply softplus to ensure sigma is positive
        sigma = F.softplus(params[:, 1]) + 1e-5
        
        return mu, sigma
    
    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Sample from Gaussian distribution.
        
        Args:
            mu: Mean of shape [batch_size]
            sigma: Standard deviation of shape [batch_size]
            
        Returns:
            Samples of shape [batch_size]
        """
        # Sample from standard normal distribution
        eps = torch.randn_like(mu)
        
        # Scale and shift by mu and sigma
        samples = mu + sigma * eps
        
        return samples