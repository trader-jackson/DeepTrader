DeepTrader
Implementation of the paper "DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding" (AAAI 2021).
Overview
DeepTrader is a deep reinforcement learning method for portfolio management that balances risk and return effectively. It consists of two main components:

Asset Scoring Unit: Evaluates individual stocks and learns dynamic patterns from historical data, with the price rising rate as the reward function. It captures both temporal and spatial dependencies between assets through a graph-based architecture.
Market Scoring Unit: Embeds market conditions as an indicator to dynamically adjust the proportion between long and short funds, using negative maximum drawdown as the reward function.

The model captures interrelationships between assets hierarchically through different types of graph structures, finding that causal structure works best compared to industry classification and correlation.
Features

Deep reinforcement learning-based portfolio management
Risk-return balanced trading strategy with dynamic adjustment
Support for both long and short positions
Hierarchical graph-based asset relationship modeling
Multiple graph structure types: industry, correlation, partial correlation, causal
Implementation for three major stock indices: Dow 30, NASDAQ 100, SSE 50

Installation
bash# Clone the repository
git clone https://github.com/yourusername/deeptrader.git
cd deeptrader

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Project Structure
deeptrader/
├── config.py                   # Configuration parameters
├── data/                       # Data directory
│   ├── dow30/                  # Dow Jones 30 data
│   ├── nasdaq100/              # NASDAQ 100 data
│   └── sse50/                  # SSE 50 data
├── main.py                     # Main entry point
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
└── src/
    ├── data_processor.py       # Data loading and preprocessing
    ├── environment.py          # Trading environment
    ├── evaluation.py           # Model evaluation metrics
    ├── models/
    │   ├── asset_scoring.py    # Asset scoring unit
    │   ├── market_scoring.py   # Market scoring unit 
    │   ├── portfolio_gen.py    # Portfolio generator
    │   └── deeptrader.py       # Complete DeepTrader model
    ├── train.py                # Training loop
    └── utils/
        ├── graph_utils.py      # Graph construction utilities
        └── metrics.py          # Performance metrics
Usage
Downloading Data
bash# Download data for Dow Jones 30
python main.py train --dataset dow30 --download --num-epochs 1

# Download data for NASDAQ 100
python main.py train --dataset nasdaq100 --download --num-epochs 1

# Download data for SSE 50
python main.py train --dataset sse50 --download --num-epochs 1
Training the Model
bash# Train DeepTrader on Dow 30 with different graph types
python main.py train --dataset dow30 --graph-type causal --reward-type mdd --num-epochs 100
python main.py train --dataset dow30 --graph-type correlation --reward-type mdd --num-epochs 100
python main.py train --dataset dow30 --graph-type industry --reward-type mdd --num-epochs 100

# Train with different reward functions
python main.py train --dataset dow30 --graph-type causal --reward-type ror --num-epochs 100
python main.py train --dataset dow30 --graph-type causal --reward-type sr --num-epochs 100
python main.py train --dataset dow30 --graph-type causal --reward-type cr --num-epochs 100

# Compare with baseline strategies
python main.py train --dataset dow30 --graph-type causal --reward-type mdd --num-epochs 100 --baselines
Evaluating the Model
bash# Evaluate a trained model
python main.py evaluate --dataset dow30 --model-path ./results/dow30_YYYYMMDD_HHMMSS/checkpoints/best_model.pth --baselines

# Perform ablation study
python main.py ablation --dataset dow30 --model-path ./results/dow30_YYYYMMDD_HHMMSS/checkpoints/best_model.pth
Command-line Arguments
Training Arguments

--dataset: Dataset to use (dow30, nasdaq100, sse50)
--graph-type: Graph type for asset scoring unit (industry, correlation, partial_correlation, causal)
--reward-type: Reward type for market scoring unit (ror, sr, mdd, cr)
--num-epochs: Number of training epochs
--batch-size: Batch size for training
--learning-rate: Learning rate
--market-reward-weight: Weight for market scoring reward
--winner-size: Number of stocks to select as winners/losers
--eval-freq: Frequency of evaluation during training
--save-freq: Frequency of saving checkpoints
--baselines: Compare with baseline strategies

Evaluation Arguments

--dataset: Dataset to use
--model-path: Path to saved model checkpoint
--graph-type: Graph type for asset scoring unit
--baselines: Evaluate baseline strategies

Performance Metrics
The model's performance is evaluated using these metrics:

Annual Rate of Return (ARR): The annualized return of the portfolio
Annual Volatility (AVol): Annualized standard deviation of returns
Sharpe Ratio (ASR): Annualized return divided by annualized volatility
Sortino Ratio (SoR): Similar to Sharpe ratio but only considers downside risk
Maximum Drawdown (MDD): Maximum observed loss from peak to trough
Calmar Ratio (CR): Annualized return divided by maximum drawdown

Key Findings
As demonstrated in the original paper, DeepTrader:

Graph Structures: Causal graph structure consistently outperforms industry classification and correlation in capturing cross-asset relationships.
Market Conditions: The market scoring unit significantly improves performance by dynamically adjusting the proportion between long and short funds based on market conditions.
Crisis Performance: DeepTrader shows superior risk-return balance especially during market downturns like the 2008 financial crisis.

Citation
If you use this code in your research, please cite the original paper:
@inproceedings{wang2021deeptrader,
  title={DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding},
  author={Wang, Zhicheng and Huang, Biwei and Tu, Shikui and Zhang, Kun and Xu, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={1},
  pages={643--650},
  year={2021}
}
License
This project is licensed under the MIT License.

requirements.txt
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
torch>=1.9.0
tqdm>=4.62.0
networkx>=2.6.0
scikit-learn>=0.24.0
yfinance>=0.1.70
pgmpy>=0.1.17
scipy>=1.7.0
python-dateutil>=2.8.0
seaborn>=0.11.0
This requirements.txt file includes all necessary dependencies for running the DeepTrader implementation. Make sure to install them using pip install -r requirements.txt before running the code.