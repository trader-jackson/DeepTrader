"""
DeepTrader model combining Asset Scoring Unit, Market Scoring Unit, and Portfolio Generator.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .asset_scoring import AssetScoringUnit
from .market_scoring import MarketScoringUnit
from .portfolio_gen import PortfolioGenerator


class DeepTrader(nn.Module):
    """
    DeepTrader model as described in the paper.
    """
    
    def __init__(self, 
                 tickers,
                 stock_feature_dim: int,
                 market_feature_dim: int,
                 hidden_dim: int = 64,
                 num_layers_asset: int = 4,
                 num_layers_market: int = 2,
                 kernel_size: int = 3,
                 dilation_base: int = 2,
                 dropout: float = 0.1,
                 winner_size: int = 5,
                 adj_matrix: Optional[torch.Tensor] = None,
                 graph_type: str = 'causal'):
        """
        Initialize DeepTrader model.
        
        Args:
            num_stocks: Number of stocks
            stock_feature_dim: Dimension of stock features
            market_feature_dim: Dimension of market features
            hidden_dim: Hidden dimension
            num_layers_asset: Number of layers in Asset Scoring Unit
            num_layers_market: Number of layers in Market Scoring Unit
            kernel_size: Kernel size for TCN
            dilation_base: Dilation base for TCN
            dropout: Dropout rate
            winner_size: Number of stocks to select as winners/losers
            adj_matrix: Adjacency matrix for graph structure
            graph_type: Type of graph structure to use
        """
        super(DeepTrader, self).__init__()
        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.stock_feature_dim = stock_feature_dim
        self.market_feature_dim = market_feature_dim
        self.hidden_dim = hidden_dim
        
        # Asset Scoring Unit
        self.asset_scoring = AssetScoringUnit(
            num_stocks=self.num_stocks,
            input_dim=stock_feature_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers_asset,
            dilation_base=dilation_base,
            dropout=dropout,
            adj_matrix=adj_matrix,
            graph_type=graph_type
        )
        
        # Market Scoring Unit
        self.market_scoring = MarketScoringUnit(
            input_dim=market_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_market,
            dropout=dropout
        )
        
        # Portfolio Generator
        self.portfolio_gen = PortfolioGenerator(
            num_stocks=self.num_stocks,
            winner_size=winner_size
        )
    
    def forward(self, stock_features: torch.Tensor, market_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            stock_features: Stock features of shape [batch_size, num_stocks, stock_feature_dim, seq_length]
            market_features: Market features of shape [batch_size, seq_length, market_feature_dim]
            
        Returns:
            Dictionary with model outputs
        """
        # Asset Scoring Unit
        # [batch_size, num_stocks]
        asset_scores = self.asset_scoring(stock_features)
        
        # Market Scoring Unit
        # mu: [batch_size], sigma: [batch_size]
        mu, sigma = self.market_scoring(market_features)
        
        # Sample short ratio from Gaussian distribution during training
        if self.training:
            short_ratio = self.market_scoring.sample(mu, sigma)
        else:
            # Use mean as the short ratio during inference
            short_ratio = mu
        
        # Portfolio Generator
        # Dictionary with portfolio weights
        portfolio = self.portfolio_gen(asset_scores, short_ratio)
        
        # Add distribution parameters to the output
        portfolio["mu"] = mu
        portfolio["sigma"] = sigma
        portfolio["asset_scores"] = asset_scores
        
        return portfolio

    def act(self, stock_features: torch.Tensor, market_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get action (portfolio weights) for given state.
        
        Args:
            stock_features: Stock features of shape [batch_size, num_stocks, stock_feature_dim, seq_length]
            market_features: Market features of shape [batch_size, seq_length, market_feature_dim]
            
        Returns:
            Dictionary with portfolio weights
        """
        self.eval()
        with torch.no_grad():
            portfolio = self.forward(stock_features, market_features)
        
        return {
            "long_weights": portfolio["long_weights"],
            "short_weights": portfolio["short_weights"],
            "short_ratio": portfolio["short_ratio"],
        }

    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, **kwargs):
        """
        Load model from file.
        
        Args:
            path: Path to load the model from
            **kwargs: Additional arguments to initialize the model
            
        Returns:
            Loaded model
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model