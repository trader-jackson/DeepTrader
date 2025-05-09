"""
Trading environment for DeepTrader.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from config import TRADING_CONFIG


class TradingEnvironment:
    """
    Trading environment for portfolio management with DeepTrader.
    
    This environment simulates the trading process described in the paper,
    allowing both long and short positions.
    """
    
    def __init__(self, tickers: List[str], initial_capital: float = None, 
                 transaction_cost: float = None, short_fee: float = None):
        """
        Initialize the trading environment.
        
        Args:
            tickers: List of stock tickers
            initial_capital: Initial capital
            transaction_cost: Transaction cost ratio
            short_fee: Fee for short selling
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
        
        # Trading parameters
        self.initial_capital = initial_capital or TRADING_CONFIG["initial_capital"]
        self.transaction_cost = transaction_cost or TRADING_CONFIG["transaction_cost"]
        self.short_fee = short_fee or TRADING_CONFIG["short_fee"]
        
        # Portfolio state
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state
        """
        # Portfolio state
        self.cash = self.initial_capital  # Risk-free assets (cash)
        self.long_positions = np.zeros(self.n_assets)  # Long positions
        self.short_positions = np.zeros(self.n_assets)  # Short positions (borrowed stocks)
        self.prices = None  # Current prices
        self.prev_prices = None  # Previous prices
        self.step_count = 0
        
        # Performance tracking
        self.portfolio_values = [self.initial_capital]
        self.returns = []
        self.max_drawdown = 0
        
        return self._get_state()
    
    def step(self, action: Dict[str, np.ndarray], prices: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Dictionary with actions
                - long_weights: Long portfolio weights
                - short_weights: Short portfolio weights
                - short_ratio: Ratio of assets used for short position
            prices: Current closing prices
                
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        long_weights = action["long_weights"]  # Shape: [n_assets]
        short_weights = action["short_weights"]  # Shape: [n_assets]
        short_ratio = action["short_ratio"]  # Scalar
        
        # Update prices
        self.prev_prices = self.prices
        self.prices = prices
        
        if self.prev_prices is None:
            self.prev_prices = self.prices
        
        # Execute trading steps as described in the paper
        portfolio_value_before = self._get_portfolio_value()
        
        # 1) Sell all stocks on long position and get cash
        proceeds_from_long = np.sum(self.long_positions * self.prices)
        self.cash += proceeds_from_long * (1 - self.transaction_cost)
        self.long_positions = np.zeros(self.n_assets)
        
        # 2) Buy back the borrowed stocks and return to broker
        cost_to_cover = np.sum(self.short_positions * self.prices)
        self.cash -= cost_to_cover * (1 + self.transaction_cost)
        self.short_positions = np.zeros(self.n_assets)
        
        # 3) Mortgage stocks and sell them (short selling)
        current_portfolio_value = self.cash
        short_value = current_portfolio_value * short_ratio
        
        # Calculate shares to short for each asset
        short_shares = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if short_weights[i] > 0:
                short_shares[i] = (short_value * short_weights[i]) / self.prices[i]
        
        # Update short positions and cash
        self.short_positions = short_shares
        self.cash += np.sum(short_shares * self.prices) * (1 - self.transaction_cost - self.short_fee)
        
        # 4) Purchase stocks based on long weights
        long_value = self.cash * (1 - short_ratio)
        
        # Calculate shares to buy for each asset
        long_shares = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if long_weights[i] > 0:
                long_shares[i] = (long_value * long_weights[i]) / self.prices[i]
        
        # Update long positions and cash
        self.long_positions = long_shares
        self.cash -= np.sum(long_shares * self.prices) * (1 + self.transaction_cost)
        
        # Calculate portfolio value after trading
        portfolio_value_after = self._get_portfolio_value()
        
        # Calculate return
        period_return = (portfolio_value_after / portfolio_value_before) - 1
        self.returns.append(period_return)
        self.portfolio_values.append(portfolio_value_after)
        
        # Update maximum drawdown
        self.max_drawdown = self._calculate_max_drawdown()
        
        # Increment step counter
        self.step_count += 1
        
        # Calculate reward (daily return)
        reward = period_return
        
        # Check if done (episode ends)
        done = False
        
        # Prepare info dictionary
        info = {
            "portfolio_value": portfolio_value_after,
            "return": period_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "sortino_ratio": self._calculate_sortino_ratio(),
            "calmar_ratio": self._calculate_calmar_ratio(),
        }
        
        return self._get_state(), reward, done, info
    
    def _get_portfolio_value(self) -> float:
        """
        Calculate current portfolio value.
        
        Returns:
            Portfolio value
        """
        if self.prices is None:
            return self.cash
        
        long_value = np.sum(self.long_positions * self.prices)
        short_value = np.sum(self.short_positions * self.prices)
        
        return self.cash + long_value - short_value
    
    def _get_state(self) -> Dict[str, Any]:
        """
        Get current state of the environment.
        
        Returns:
            State dictionary
        """
        return {
            "cash": self.cash,
            "long_positions": self.long_positions,
            "short_positions": self.short_positions,
            "prices": self.prices,
            "portfolio_value": self._get_portfolio_value(),
            "step_count": self.step_count
        }
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown
        """
        # Convert to numpy array for calculations
        values = np.array(self.portfolio_values)
        
        # Calculate the running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate the drawdown
        drawdown = (running_max - values) / running_max
        
        # Get the maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.
        
        Returns:
            Sharpe ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0
        sharpe = mean_return / std_return
        
        # Annualize (assuming 252 trading days)
        sharpe = sharpe * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino_ratio(self) -> float:
        """
        Calculate Sortino ratio.
        
        Returns:
            Sortino ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.std(downside_returns)
        
        # Assuming risk-free rate of 0
        sortino = mean_return / downside_deviation
        
        # Annualize (assuming 252 trading days)
        sortino = sortino * np.sqrt(252)
        
        return sortino
    
    def _calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.
        
        Returns:
            Calmar ratio
        """
        if len(self.returns) < 2 or self.max_drawdown == 0:
            return 0.0
        
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        
        # Annualize return (assuming 252 trading days)
        annual_return = mean_return * 252
        
        # Calmar ratio
        calmar = annual_return / self.max_drawdown
        
        return calmar
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.portfolio_values) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0
            }
        
        # Calculate total return
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        
        # Calculate annualized return
        n_days = len(self.portfolio_values) - 1
        annualized_return = ((1 + total_return) ** (252 / n_days)) - 1
        
        # Calculate annualized volatility
        returns = np.array(self.returns)
        annualized_volatility = np.std(returns) * np.sqrt(252)
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "sortino_ratio": self._calculate_sortino_ratio(),
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self._calculate_calmar_ratio()
        }