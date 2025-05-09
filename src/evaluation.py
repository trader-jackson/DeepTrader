"""
Evaluation module for DeepTrader model.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.environment import TradingEnvironment
from src.utils.metrics import calculate_metrics
from src.models.deeptrader import DeepTrader
from config import DATASETS


class Evaluator:
    """
    Evaluator for DeepTrader model.
    """
    
    def __init__(self, model: DeepTrader, tickers: List[str], device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: DeepTrader model
            tickers: List of ticker symbols
            device: Device to use for evaluation
        """
        self.model = model
        self.tickers = tickers
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate(self, dataset: List[Dict], env: TradingEnvironment, output_dir: str = None) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Dataset for evaluation
            env: Trading environment
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Reset environment
        state = env.reset()
        
        # Track portfolio values
        portfolio_values = [env._get_portfolio_value()]
        returns = []
        dates = []
        actions = []
        asset_scores = []
        
        # Evaluate model on dataset
        for i, data in enumerate(dataset):
            # Extract features
            stock_features = torch.tensor(data["stock_features"], dtype=torch.float32).to(self.device)
            market_features = torch.tensor(data["market_features"], dtype=torch.float32).to(self.device)
            
            # Add batch dimension if needed
            if stock_features.dim() == 3:
                stock_features = stock_features.unsqueeze(0)
            if market_features.dim() == 2:
                market_features = market_features.unsqueeze(0)
            
            # Get action from model
            with torch.no_grad():
                action = self.model.act(stock_features, market_features)
                
                # Get asset scores for analysis
                output = self.model(stock_features, market_features)
                scores = output["asset_scores"].cpu().numpy()
                asset_scores.append(scores)
            
            # Convert to numpy
            action_np = {
                "long_weights": action["long_weights"].cpu().numpy()[0],
                "short_weights": action["short_weights"].cpu().numpy()[0],
                "short_ratio": action["short_ratio"].cpu().numpy()[0],
            }
            
            # Execute action in environment
            prices = np.array(data["target_returns"]) + 1.0  # Convert returns to price relatives
            next_state, reward, done, info = env.step(action_np, prices)
            
            # Track results
            portfolio_values.append(info["portfolio_value"])
            returns.append(info["return"])
            dates.append(data["date"])
            actions.append(action_np)
        
        # Calculate performance metrics
        metrics = calculate_metrics(portfolio_values)
        
        # Add additional metrics
        metrics["cumulative_return"] = portfolio_values[-1] / portfolio_values[0] - 1
        metrics["final_portfolio_value"] = portfolio_values[-1]
        
        # Save results if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save portfolio values
            pd.DataFrame({
                "date": [dataset[0]["date"]] + [d for d in dates],
                "portfolio_value": portfolio_values
            }).to_csv(os.path.join(output_dir, "portfolio_values.csv"), index=False)
            
            # Save metrics
            pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(
                os.path.join(output_dir, "metrics.csv"), index=False
            )
            
            # Save actions
            action_df = []
            for i, action in enumerate(actions):
                date = dates[i]
                for j, ticker in enumerate(self.tickers):
                    action_df.append({
                        "date": date,
                        "ticker": ticker,
                        "long_weight": action["long_weights"][j],
                        "short_weight": action["short_weights"][j],
                        "asset_score": asset_scores[i][0][j]
                    })
            
            pd.DataFrame(action_df).to_csv(
                os.path.join(output_dir, "actions.csv"), index=False
            )
            
            # Generate plots
            self._generate_plots(portfolio_values, dates, output_dir)
        
        return metrics
    
    def _generate_plots(self, portfolio_values: List[float], dates: List, output_dir: str):
        """
        Generate plots for evaluation results.
        
        Args:
            portfolio_values: Portfolio values
            dates: Dates
            output_dir: Directory to save plots
        """
        # Convert dates to datetime if they're not already
        if not isinstance(dates[0], pd.Timestamp):
            dates = [pd.to_datetime(d) for d in dates]
        
        # Create plot directory
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(dates, portfolio_values[1:])
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "portfolio_value.png"))
        plt.close()
        
        # Plot cumulative returns
        cumulative_returns = [v / portfolio_values[0] - 1 for v in portfolio_values]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, cumulative_returns[1:])
        plt.title("Cumulative Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "cumulative_returns.png"))
        plt.close()
        
        # Plot drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (running_max - portfolio_values) / running_max
        
        plt.figure(figsize=(12, 6))
        plt.plot([pd.to_datetime(dates[0])] + dates, drawdown)
        plt.title("Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "drawdown.png"))
        plt.close()


class ComparativeEvaluator:
    """
    Evaluator for comparing multiple models.
    """
    
    def __init__(self, tickers: List[str], device: torch.device):
        """
        Initialize comparative evaluator.
        
        Args:
            tickers: List of ticker symbols
            device: Device to use for evaluation
        """
        self.tickers = tickers
        self.device = device
    
    def evaluate_models(self, models: Dict[str, DeepTrader], dataset: List[Dict], 
                        initial_capital: float = 10000, output_dir: str = None) -> Dict:
        """
        Evaluate multiple models on dataset.
        
        Args:
            models: Dictionary mapping model name to model
            dataset: Dataset for evaluation
            initial_capital: Initial capital for environment
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating model: {name}")
            
            # Create environment
            env = TradingEnvironment(
                tickers=self.tickers,
                initial_capital=initial_capital
            )
            
            # Create evaluator
            evaluator = Evaluator(model, self.tickers, self.device)
            
            # Evaluate model
            if output_dir is not None:
                model_output_dir = os.path.join(output_dir, name)
            else:
                model_output_dir = None
            
            metrics = evaluator.evaluate(dataset, env, model_output_dir)
            results[name] = metrics
        
        # Compare models if output directory is provided
        if output_dir is not None:
            self._compare_models(results, output_dir)
        
        return results
    
    def _compare_models(self, results: Dict[str, Dict], output_dir: str):
        """
        Compare models and generate comparative plots.
        
        Args:
            results: Dictionary with evaluation metrics for each model
            output_dir: Directory to save results
        """
        # Create comparison directory
        compare_dir = os.path.join(output_dir, "comparison")
        os.makedirs(compare_dir, exist_ok=True)
        
        # Create comparison table
        table = []
        for name, metrics in results.items():
            row = {"model": name}
            row.update(metrics)
            table.append(row)
        
        pd.DataFrame(table).to_csv(
            os.path.join(compare_dir, "comparison.csv"), index=False
        )
        
        # Create bar charts for key metrics
        key_metrics = [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
        ]
        
        for metric in key_metrics:
            plt.figure(figsize=(10, 6))
            values = [metrics[metric] for _, metrics in results.items()]
            plt.bar(list(results.keys()), values)
            plt.title(f"Comparison of {metric}")
            plt.ylabel(metric)
            plt.grid(True, axis="y")
            plt.savefig(os.path.join(compare_dir, f"{metric}.png"))
            plt.close()


def evaluate_baseline_strategies(dataset: List[Dict], tickers: List[str], 
                                initial_capital: float = 10000, output_dir: str = None) -> Dict:
    """
    Evaluate baseline strategies on dataset.
    
    Args:
        dataset: Dataset for evaluation
        tickers: List of ticker symbols
        initial_capital: Initial capital for environment
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics for each strategy
    """
    results = {}
    
    # Define baseline strategies
    strategies = {
        "Market": market_strategy,
        "BLSW": blsw_strategy,
        "CSM": csm_strategy,
    }
    
    for name, strategy_fn in strategies.items():
        print(f"Evaluating baseline strategy: {name}")
        
        # Create environment
        env = TradingEnvironment(
            tickers=tickers,
            initial_capital=initial_capital
        )
        
        # Reset environment
        state = env.reset()
        
        # Track portfolio values
        portfolio_values = [env._get_portfolio_value()]
        returns = []
        dates = []
        
        # Evaluate strategy on dataset
        for i, data in enumerate(dataset):
            # Get action from strategy
            action = strategy_fn(data, tickers)
            
            # Execute action in environment
            prices = np.array(data["target_returns"]) + 1.0  # Convert returns to price relatives
            next_state, reward, done, info = env.step(action, prices)
            
            # Track results
            portfolio_values.append(info["portfolio_value"])
            returns.append(info["return"])
            dates.append(data["date"])
        
        # Calculate performance metrics
        metrics = calculate_metrics(portfolio_values)
        
        # Add additional metrics
        metrics["cumulative_return"] = portfolio_values[-1] / portfolio_values[0] - 1
        metrics["final_portfolio_value"] = portfolio_values[-1]
        
        results[name] = metrics
        
        # Save results if output directory is provided
        if output_dir is not None:
            strategy_dir = os.path.join(output_dir, name)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Save portfolio values
            pd.DataFrame({
                "date": [dataset[0]["date"]] + [d for d in dates],
                "portfolio_value": portfolio_values
            }).to_csv(os.path.join(strategy_dir, "portfolio_values.csv"), index=False)
            
            # Save metrics
            pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(
                os.path.join(strategy_dir, "metrics.csv"), index=False
            )
            
            # Generate plots
            plot_dir = os.path.join(strategy_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot portfolio value over time
            plt.figure(figsize=(12, 6))
            plt.plot(dates, portfolio_values[1:])
            plt.title(f"{name} - Portfolio Value Over Time")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value")
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "portfolio_value.png"))
            plt.close()
    
    return results


def market_strategy(data: Dict, tickers: List[str]) -> Dict[str, np.ndarray]:
    """
    Market (Buy and Hold) strategy.
    
    Args:
        data: Data for current step
        tickers: List of ticker symbols
        
    Returns:
        Action dictionary
    """
    n_assets = len(tickers)
    
    # Equal weight for all assets, no short selling
    long_weights = np.ones(n_assets) / n_assets
    short_weights = np.zeros(n_assets)
    short_ratio = 0.0
    
    return {
        "long_weights": long_weights,
        "short_weights": short_weights,
        "short_ratio": short_ratio
    }


def blsw_strategy(data: Dict, tickers: List[str]) -> Dict[str, np.ndarray]:
    """
    Buying-Loser-Selling-Winner (BLSW) strategy.
    
    Args:
        data: Data for current step
        tickers: List of ticker symbols
        
    Returns:
        Action dictionary
    """
    n_assets = len(tickers)
    returns = np.array(data["target_returns"])
    
    # Sort returns in ascending order (lowest returns first)
    sorted_indices = np.argsort(returns)
    
    # Select bottom 30% for long and top 30% for short
    num_select = max(1, int(n_assets * 0.3))
    
    long_indices = sorted_indices[:num_select]
    short_indices = sorted_indices[-num_select:]
    
    # Initialize weights
    long_weights = np.zeros(n_assets)
    short_weights = np.zeros(n_assets)
    
    # Equal weights for selected assets
    long_weights[long_indices] = 1.0 / num_select
    short_weights[short_indices] = 1.0 / num_select
    
    # Allocate 50% to short positions
    short_ratio = 0.5
    
    return {
        "long_weights": long_weights,
        "short_weights": short_weights,
        "short_ratio": short_ratio
    }


def csm_strategy(data: Dict, tickers: List[str]) -> Dict[str, np.ndarray]:
    """
    Cross-Sectional Momentum (CSM) strategy.
    
    Args:
        data: Data for current step
        tickers: List of ticker symbols
        
    Returns:
        Action dictionary
    """
    n_assets = len(tickers)
    
    # Use historical returns as momentum indicator
    # Extract returns from asset features (assuming last feature is return)
    features = data["stock_features"]
    returns = np.mean(features[:, -1, -5:], axis=1)  # Average return over last 5 days
    
    # Sort returns in descending order (highest returns first)
    sorted_indices = np.argsort(-returns)
    
    # Select top 30% for long and bottom 30% for short
    num_select = max(1, int(n_assets * 0.3))
    
    long_indices = sorted_indices[:num_select]
    short_indices = sorted_indices[-num_select:]
    
    # Initialize weights
    long_weights = np.zeros(n_assets)
    short_weights = np.zeros(n_assets)
    
    # Equal weights for selected assets
    long_weights[long_indices] = 1.0 / num_select
    short_weights[short_indices] = 1.0 / num_select
    
    # Allocate 50% to short positions
    short_ratio = 0.5
    
    return {
        "long_weights": long_weights,
        "short_weights": short_weights,
        "short_ratio": short_ratio
    }