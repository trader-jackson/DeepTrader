"""
Portfolio Generator for DeepTrader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PortfolioGenerator(nn.Module):
    """
    Portfolio Generator as described in the paper.
    
    This module generates the portfolio weights based on:
    1. Asset scores from the Asset Scoring Unit
    2. Short ratio from the Market Scoring Unit
    """
    
    def __init__(self, num_stocks: int, winner_size: int):
        """
        Initialize Portfolio Generator.
        
        Args:
            num_stocks: Number of stocks
            winner_size: Number of stocks to select as winners/losers (top-G/bottom-G)
        """
        super(PortfolioGenerator, self).__init__()
        
        self.num_stocks = num_stocks
        self.winner_size = winner_size
    
    def forward(self, asset_scores: torch.Tensor, short_ratio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            asset_scores: Asset scores from Asset Scoring Unit of shape [batch_size, num_stocks]
            short_ratio: Short ratio from Market Scoring Unit of shape [batch_size]
            
        Returns:
            Dictionary with portfolio weights
        """
        batch_size = asset_scores.size(0)
        
        # Clamp short ratio to [0, 1]
        short_ratio = torch.clamp(short_ratio, 0.0, 1.0)
        
        # Initialize portfolio weights
        long_weights = torch.zeros((batch_size, self.num_stocks), device=asset_scores.device)
        short_weights = torch.zeros((batch_size, self.num_stocks), device=asset_scores.device)
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Sort asset scores
            sorted_scores, sorted_indices = torch.sort(asset_scores[b], descending=True)
            
            # Select top-G stocks as winners for long position
            winners = sorted_indices[:self.winner_size]
            
            # Select bottom-G stocks as losers for short position
            losers = sorted_indices[-self.winner_size:]
            
            # Calculate long weights (proportional to asset scores)
            winner_scores = asset_scores[b, winners]
            long_weights[b, winners] = F.softmax(winner_scores, dim=0)
            
            # Calculate short weights (inversely proportional to asset scores)
            loser_scores = 1.0 - asset_scores[b, losers]
            short_weights[b, losers] = F.softmax(loser_scores, dim=0)
        
        return {
            "long_weights": long_weights,
            "short_weights": short_weights,
            "short_ratio": short_ratio
        }