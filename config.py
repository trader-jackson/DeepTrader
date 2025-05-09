"""
Configuration parameters for DeepTrader model.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"

TRAIN_START_DATE = "2010-01-01"  
TRAIN_END_DATE = "2018-12-31"

EVAL_START_DATE = "2019-01-01"
EVAL_END_DATE = "2021-12-31"

TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2023-12-31"
# Dataset configurations
DATASETS = {
    "dow30": {
        "train_period": (TRAIN_START_DATE, EVAL_END_DATE),
        "test_period": (TEST_START_DATE, TEST_END_DATE),
        "num_stocks": 30,
    },
    "nasdaq100": {
        "train_period": ("1990-01-01", "2006-12-31"),
        "test_period": ("2007-01-01", "2019-12-31"),
        "num_stocks": 100,
    },
    "sse50": {
        "train_period": ("2005-01-01", "2012-12-31"),
        "test_period": ("2013-01-01", "2019-12-31"),
        "num_stocks": 50,
    },
}

# Model hyperparameters
MODEL_CONFIG = {
    # Asset scoring unit
    "asset_scoring": {
        "input_dim": 10,          # Number of stock indicators
        "hidden_dim": 64,         # Hidden dimension
        "num_layers": 4,          # Number of spatial-TCN blocks
        "dilation_base": 2,       # Dilation base for TCN
        "graph_type": "causal",   # Graph type: ["industry", "correlation", "partial_correlation", "causal"]
        "dropout": 0.1,           # Dropout rate
    },
    
    # Market scoring unit
    "market_scoring": {
        "input_dim": 5,           # Number of market indicators
        "hidden_dim": 64,         # Hidden dimension
        "num_layers": 2,          # Number of LSTM layers
        "dropout": 0.1,           # Dropout rate
    },
    
    # Training parameters
    "train": {
        "batch_size": 32,         # Batch size for training
        "sequence_length": 50,    # Length of input sequence
        "learning_rate": 1e-4,    # Learning rate
        "gamma": 0.99,            # Discount factor for RL
        "entropy_weight": 0.001,  # Entropy weight for exploration
        "reward_type": "mdd",     # Reward function type: ["ror", "sr", "mdd", "cr"]
        "winner_size": 5,         # Top/bottom k stocks to invest
        "market_reward_weight": 0.5, # Weight for market scoring reward
    },
}

# Stock indicators (asset scoring unit inputs)
STOCK_INDICATORS = [
    "open",             # Opening price
    "high",             # Highest price
    "low",              # Lowest price
    "close",            # Closing price
    "volume",           # Trading volume
    "ma5",              # 5-day moving average
    "ma10",             # 10-day moving average
    "rsi",              # Relative Strength Index
    "macd",             # Moving Average Convergence Divergence
    "boll_upper",       # Bollinger Bands upper
    "boll_lower",       # Bollinger Bands lower
]

# Market indicators (market scoring unit inputs)
MARKET_INDICATORS = [
    "index_return",     # Market index return
    "index_ma5",        # 5-day moving average of index
    "index_ma10",       # 10-day moving average of index
    "vol_ratio",        # Ratio of rising stocks
    "advance_decline",  # Advance-decline ratio
]

# Trading parameters
TRADING_CONFIG = {
    "initial_capital": 10000,     # Initial capital
    "transaction_cost": 0.001,    # Transaction cost ratio
    "short_fee": 0.001,           # Fee for short selling
}